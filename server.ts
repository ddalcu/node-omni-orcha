import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { createModel } from './src/index.ts';
import type { LlmModel, TtsModel, ImageModel, ChatMessage } from './src/types.ts';

// --- Config ---
const PORT = Number(process.env.PORT ?? 3333);
const MODELS_DIR = process.env.MODELS_DIR ?? `${process.env.HOME}/.orcha/workspace/.models`;

const LLM_PATH = process.env.LLM_MODEL ?? `${MODELS_DIR}/Qwen3.5-4B-IQ4_NL.gguf`;
const TTS_PATH = process.env.TTS_MODEL ?? `${MODELS_DIR}/qwen3-tts`;

// FLUX 2 Klein (image generation)
const IMAGE_DIR = `${MODELS_DIR}/flux2-klein`;
const IMAGE_PATH = process.env.IMAGE_MODEL ?? `${IMAGE_DIR}/flux-2-klein-4b-Q4_K_M.gguf`;
const IMAGE_VAE_PATH = process.env.IMAGE_VAE ?? `${IMAGE_DIR}/flux2-vae.safetensors`;
const IMAGE_LLM_PATH = process.env.IMAGE_LLM ?? `${IMAGE_DIR}/Qwen3-4B-Q4_K_M.gguf`;

// WAN video generation — locked to 480p presets
const VIDEO_RESOLUTIONS: Record<string, [number, number]> = {
  '16:9': [832, 480],
  '9:16': [480, 832],
  '1:1':  [624, 624],
};

const WAN_NEGATIVE_PROMPT = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走';

type VideoVariant = '5b' | 'turbo-a14b';

interface VideoVariantConfig {
  label: string;
  dir: string;
  model: string;
  vae: string;
  t5xxl: string;
  highNoiseModel?: string;
  defaults: { steps: number; cfgScale: number; flowShift: number };
  highNoiseDefaults?: { steps: number; cfgScale: number; sampleMethod: string };
}

const VIDEO_VARIANTS: Record<VideoVariant, VideoVariantConfig> = {
  '5b': {
    label: 'WAN 2.2 TI2V 5B',
    dir: `${MODELS_DIR}/wan22-5b`,
    model: 'Wan2.2-TI2V-5B-Q8_0.gguf',
    vae: 'Wan2.2_VAE.safetensors',
    t5xxl: 'umt5-xxl-encoder-Q8_0.gguf',
    defaults: { steps: 30, cfgScale: 6.0, flowShift: 3.0 },
  },
  'turbo-a14b': {
    label: 'WAN 2.2 T2V A14B',
    dir: `${MODELS_DIR}/wan22-turbo`,
    model: 'Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf',
    highNoiseModel: 'Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf',
    vae: 'Wan2.1_VAE.safetensors',
    t5xxl: 'umt5-xxl-encoder-Q8_0.gguf',
    defaults: { steps: 10, cfgScale: 3.5, flowShift: 3.0 },
    highNoiseDefaults: { steps: 8, cfgScale: 3.5, sampleMethod: 'euler' },
  },
};

// --- Model state ---
const models: {
  llm: LlmModel | null;
  tts: TtsModel | null;
  image: ImageModel | null;
  video: ImageModel | null;
} = { llm: null, tts: null, image: null, video: null };

let videoVariant: VideoVariant | null = null;

const loading: Record<string, Promise<void> | null> = {
  llm: null, tts: null, image: null, video: null,
};

// --- Helpers ---

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

function json(res: ServerResponse, data: unknown, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

function errorResponse(res: ServerResponse, msg: string, status = 500) {
  json(res, { error: msg }, status);
}

// --- Lazy model loaders ---

function loadLlm(): Promise<void> {
  if (models.llm) return Promise.resolve();
  if (loading.llm) return loading.llm;
  if (!LLM_PATH) return Promise.reject(new Error('LLM_MODEL not configured'));

  console.log(`Loading LLM: ${LLM_PATH}`);
  loading.llm = (async () => {
    const m = createModel(LLM_PATH, 'llm');
    await m.load({ contextSize: 4096 });
    models.llm = m;
    console.log('LLM ready');
  })().catch(e => { loading.llm = null; throw e; });
  return loading.llm;
}

function loadTts(): Promise<void> {
  if (models.tts) return Promise.resolve();
  if (loading.tts) return loading.tts;
  if (!TTS_PATH) return Promise.reject(new Error('TTS_MODEL not configured'));

  console.log(`Loading TTS (qwen3): ${TTS_PATH}`);
  loading.tts = (async () => {
    const m = createModel(TTS_PATH, 'tts');
    await m.load({ engine: 'qwen3' });
    models.tts = m;
    console.log('TTS ready');
  })().catch(e => { loading.tts = null; throw e; });
  return loading.tts;
}

function loadImage(): Promise<void> {
  if (models.image) return Promise.resolve();
  if (loading.image) return loading.image;
  if (!IMAGE_PATH) return Promise.reject(new Error('IMAGE_MODEL not configured'));

  console.log(`Loading Image (FLUX 2 Klein): ${IMAGE_PATH}`);
  loading.image = (async () => {
    const m = createModel(IMAGE_PATH, 'image');
    await m.load({
      vaePath: IMAGE_VAE_PATH,
      llmPath: IMAGE_LLM_PATH,
    });
    models.image = m;
    console.log('Image ready');
  })().catch(e => { loading.image = null; throw e; });
  return loading.image;
}

async function loadVideo(variant: VideoVariant = '5b'): Promise<void> {
  if (models.video && videoVariant === variant) return;
  if (loading.video) await loading.video.catch(() => {});

  if (models.video && videoVariant !== variant) {
    console.log(`Unloading Video (${videoVariant})...`);
    await models.video.unload();
    models.video = null;
    videoVariant = null;
  }

  if (models.video) return;

  const v = VIDEO_VARIANTS[variant];
  const modelPath = `${v.dir}/${v.model}`;
  console.log(`Loading Video (${v.label}): ${modelPath}`);

  loading.video = (async () => {
    const m = createModel(modelPath, 'image');
    await m.load({
      vaePath: `${v.dir}/${v.vae}`,
      t5xxlPath: `${v.dir}/${v.t5xxl}`,
      ...(v.highNoiseModel ? { highNoiseDiffusionModelPath: `${v.dir}/${v.highNoiseModel}` } : {}),
      flashAttn: true,
      vaeDecodeOnly: true,
    });
    models.video = m;
    videoVariant = variant;
    console.log(`Video ready (${v.label})`);
  })().catch(e => { loading.video = null; throw e; });
  return loading.video;
}

// --- Routes ---

async function handleLoad(req: IncomingMessage, res: ServerResponse) {
  const body = JSON.parse(await readBody(req));
  const type = body.type as string;

  const loaders: Record<string, () => Promise<void>> = {
    llm: loadLlm,
    tts: loadTts,
    image: loadImage,
    video: () => loadVideo((body.variant ?? '5b') as VideoVariant),
  };

  const loader = loaders[type];
  if (!loader) return errorResponse(res, `Unknown model type: ${type}`, 400);

  try {
    await loader();
    json(res, { ok: true });
  } catch (e: any) {
    errorResponse(res, e.message);
  }
}

async function handleChat(req: IncomingMessage, res: ServerResponse) {
  if (!models.llm) return errorResponse(res, 'LLM not loaded. Click the Chat tab to load it.', 503);

  const body = JSON.parse(await readBody(req));
  const messages: ChatMessage[] = body.messages ?? [];

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  try {
    for await (const chunk of models.llm.stream(messages, {
      temperature: body.temperature ?? 0.7,
      maxTokens: body.maxTokens ?? 2048,
    })) {
      if (chunk.content) {
        res.write(`data: ${JSON.stringify({ content: chunk.content })}\n\n`);
      }
      if (chunk.done) {
        res.write(`data: ${JSON.stringify({ done: true, usage: chunk.usage })}\n\n`);
      }
    }
  } catch (e: any) {
    res.write(`data: ${JSON.stringify({ error: e.message })}\n\n`);
  }
  res.end();
}

async function handleTts(req: IncomingMessage, res: ServerResponse) {
  if (!models.tts) return errorResponse(res, 'TTS not loaded. Click the TTS tab to load it.', 503);

  const body = JSON.parse(await readBody(req));
  const text = body.text ?? '';
  if (!text) return errorResponse(res, 'Missing "text" field', 400);

  try {
    const speakOpts: Record<string, unknown> = {
      temperature: body.temperature ?? undefined,
    };

    if (models.tts.engine === 'qwen3') {
      speakOpts.referenceAudioPath = body.referenceAudioPath ?? '';
    } else {
      speakOpts.voice = body.voice ?? '';
      speakOpts.speed = body.speed ?? 1.0;
    }

    const wav = await models.tts.speak(text, speakOpts as any);
    res.writeHead(200, {
      'Content-Type': 'audio/wav',
      'Content-Length': wav.length,
    });
    res.end(wav);
  } catch (e: any) {
    errorResponse(res, e.message);
  }
}

async function handleImage(req: IncomingMessage, res: ServerResponse) {
  if (!models.image) return errorResponse(res, 'Image model not loaded. Click the Image tab to load it.', 503);

  const body = JSON.parse(await readBody(req));
  const prompt = body.prompt ?? '';
  if (!prompt) return errorResponse(res, 'Missing "prompt" field', 400);

  try {
    const png = await models.image.generate(prompt, {
      width: body.width ?? 512,
      height: body.height ?? 512,
      steps: body.steps,
      cfgScale: body.cfgScale ?? 1.0,
      seed: body.seed,
    });
    res.writeHead(200, {
      'Content-Type': 'image/png',
      'Content-Length': png.length,
    });
    res.end(png);
  } catch (e: any) {
    errorResponse(res, e.message);
  }
}

async function handleVideo(req: IncomingMessage, res: ServerResponse) {
  const body = JSON.parse(await readBody(req));
  const variant = (body.variant ?? '5b') as VideoVariant;
  const prompt = body.prompt ?? '';
  if (!prompt) return errorResponse(res, 'Missing "prompt" field', 400);

  try {
    if (!models.video || videoVariant !== variant) await loadVideo(variant);
    const v = VIDEO_VARIANTS[variant];
    const [width, height] = VIDEO_RESOLUTIONS[body.aspect ?? '16:9'] ?? VIDEO_RESOLUTIONS['16:9'];
    const frames = await models.video!.generateVideo(prompt, {
      width,
      height,
      negativePrompt: WAN_NEGATIVE_PROMPT,
      videoFrames: body.videoFrames ?? 33,
      steps: body.steps ?? v.defaults.steps,
      cfgScale: body.cfgScale ?? v.defaults.cfgScale,
      flowShift: body.flowShift ?? v.defaults.flowShift,
      seed: body.seed,
      ...(v.highNoiseDefaults ? {
        highNoiseSteps: v.highNoiseDefaults.steps,
        highNoiseCfgScale: v.highNoiseDefaults.cfgScale,
        highNoiseSampleMethod: v.highNoiseDefaults.sampleMethod,
      } : {}),
    });
    const b64Frames = frames.map(f => f.toString('base64'));
    json(res, { frames: b64Frames });
  } catch (e: any) {
    errorResponse(res, e.message);
  }
}

async function handleStatus(_req: IncomingMessage, res: ServerResponse) {
  json(res, {
    llm: models.llm?.loaded ?? false,
    llmLoading: loading.llm !== null && !models.llm,
    tts: models.tts ? { loaded: models.tts.loaded, engine: models.tts.engine } : false,
    ttsLoading: loading.tts !== null && !models.tts,
    image: models.image?.loaded ?? false,
    imageLoading: loading.image !== null && !models.image,
    video: models.video ? { loaded: models.video.loaded, variant: videoVariant } : false,
    videoLoading: loading.video !== null && !models.video,
  });
}

// --- Router ---

async function router(req: IncomingMessage, res: ServerResponse) {
  const url = req.url ?? '/';
  const method = req.method ?? 'GET';

  if (method === 'GET' && url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(HTML);
    return;
  }
  if (method === 'GET' && url === '/api/status') return handleStatus(req, res);
  if (method === 'POST' && url === '/api/load') return handleLoad(req, res);
  if (method === 'POST' && url === '/api/chat') return handleChat(req, res);
  if (method === 'POST' && url === '/api/tts') return handleTts(req, res);
  if (method === 'POST' && url === '/api/image') return handleImage(req, res);
  if (method === 'POST' && url === '/api/video') return handleVideo(req, res);

  res.writeHead(404);
  res.end('Not found');
}

// --- Boot (no model loading — on-demand only) ---

const server = createServer(router);
server.listen(PORT, () => {
  console.log(`node-omni-orcha server on http://localhost:${PORT}`);
  console.log('Models load on-demand when you click a tab');
});

// --- HTML ---

const HTML = /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>node-omni-orcha</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; max-width: 720px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 1.1rem; margin-bottom: 12px; color: #888; font-weight: 400; }

  .tabs { display: flex; gap: 4px; margin-bottom: 12px; }
  .tab { padding: 6px 14px; border: 1px solid #333; background: transparent; color: #888; cursor: pointer; border-radius: 4px; font-size: 0.85rem; }
  .tab.active { background: #1a1a1a; color: #fff; border-color: #555; }
  .tab.loading { border-color: #4a9eff; color: #4a9eff; animation: pulse 1.5s infinite; }
  .tab.ready { border-color: #4a4; }
  .tab.error { border-color: #a33; }

  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  .panel { display: none; }
  .panel.active { display: block; }

  .input-row { display: flex; gap: 8px; margin-bottom: 8px; }
  .input-row textarea, .input-row input[type="text"] {
    flex: 1; padding: 8px; background: #111; border: 1px solid #333; color: #e0e0e0;
    border-radius: 4px; font-size: 0.9rem; resize: vertical; font-family: inherit;
  }
  .input-row textarea { min-height: 60px; }
  .btn { padding: 8px 16px; background: #2a2a2a; border: 1px solid #444; color: #e0e0e0; border-radius: 4px; cursor: pointer; font-size: 0.85rem; white-space: nowrap; }
  .btn:hover { background: #333; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .progress { height: 3px; background: #222; border-radius: 2px; margin-bottom: 8px; overflow: hidden; display: none; }
  .progress.active { display: block; }
  .progress-bar { height: 100%; background: #4a9eff; width: 30%; animation: slide 1.2s infinite ease-in-out; }
  @keyframes slide { 0% { transform: translateX(-100%); width: 30%; } 50% { width: 50%; } 100% { transform: translateX(340%); width: 30%; } }

  .output { background: #111; border: 1px solid #222; border-radius: 4px; padding: 12px; min-height: 80px; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; line-height: 1.5; }
  .output img { max-width: 100%; border-radius: 4px; margin-top: 8px; }
  .output audio { margin-top: 8px; width: 100%; }
  .output .error { color: #f44; }

  .status { font-size: 0.75rem; color: #555; margin-top: 12px; }
  .status .on { color: #4a4; }
  .status .off { color: #633; }
  .status .loading-text { color: #4a9eff; }

  .chat-messages { background: #111; border: 1px solid #222; border-radius: 4px; padding: 12px; min-height: 200px; max-height: 400px; overflow-y: auto; margin-bottom: 8px; font-size: 0.9rem; line-height: 1.5; }
  .msg { margin-bottom: 8px; }
  .msg-user { color: #7ab; }
  .msg-assistant { color: #ccc; }
  .msg-label { font-size: 0.75rem; color: #666; margin-bottom: 2px; }

  .video-player { text-align: center; }
  .video-controls { margin-top: 8px; display: flex; gap: 8px; justify-content: center; align-items: center; }
  .video-controls span { font-size: 0.8rem; color: #888; }

  .options-row { display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
  .options-row label { font-size: 0.8rem; color: #888; }
  .options-row input, .options-row select { width: 70px; padding: 4px; background: #111; border: 1px solid #333; color: #e0e0e0; border-radius: 4px; font-size: 0.8rem; }
  .options-row select { width: auto; }
  .options-row input.wide { width: 200px; }
  .hidden { display: none !important; }
</style>
</head>
<body>
<h1>node-omni-orcha</h1>

<div class="tabs">
  <button class="tab active" data-panel="chat" data-model="llm">Chat</button>
  <button class="tab" data-panel="tts" data-model="tts">TTS</button>
  <button class="tab" data-panel="image" data-model="image">Image</button>
  <button class="tab" data-panel="video" data-model="video">Video</button>
</div>

<!-- Chat -->
<div class="panel active" id="chat">
  <div class="chat-messages" id="chat-messages"></div>
  <div class="progress" id="chat-progress"><div class="progress-bar"></div></div>
  <div class="input-row">
    <textarea id="chat-input" placeholder="Type a message..."></textarea>
    <button class="btn" id="chat-send">Send</button>
  </div>
</div>

<!-- TTS -->
<div class="panel" id="tts">
  <div class="input-row">
    <textarea id="tts-input" placeholder="Text to speak..."></textarea>
    <button class="btn" id="tts-send">Speak</button>
  </div>
  <div class="options-row" id="tts-options-kokoro">
    <label>Voice <input type="text" id="tts-voice" placeholder="af_bella"></label>
    <label>Speed <input type="text" id="tts-speed" placeholder="1.0"></label>
  </div>
  <div class="options-row" id="tts-options-qwen3">
    <label>Reference WAV <input type="text" id="tts-ref-audio" placeholder="/path/to/voice.wav" class="wide"></label>
  </div>
  <div class="progress" id="tts-progress"><div class="progress-bar"></div></div>
  <div class="output" id="tts-output"></div>
</div>

<!-- Image -->
<div class="panel" id="image">
  <div class="input-row">
    <textarea id="image-input" placeholder="Describe an image..."></textarea>
    <button class="btn" id="image-send">Generate</button>
  </div>
  <div class="options-row">
    <label>W <input type="text" id="image-w" placeholder="512"></label>
    <label>H <input type="text" id="image-h" placeholder="512"></label>
    <label>Steps <input type="text" id="image-steps" placeholder="20"></label>
  </div>
  <div class="progress" id="image-progress"><div class="progress-bar"></div></div>
  <div class="output" id="image-output"></div>
</div>

<!-- Video -->
<div class="panel" id="video">
  <div class="input-row">
    <textarea id="video-input" placeholder="Describe a video..."></textarea>
    <button class="btn" id="video-send">Generate</button>
  </div>
  <div class="options-row">
    <label>Model <select id="video-variant"><option value="5b">5B</option><option value="turbo-a14b">Turbo A14B</option></select></label>
    <label>Aspect <select id="video-aspect"><option value="16:9">16:9 (832×480)</option><option value="9:16">9:16 (480×832)</option><option value="1:1">1:1 (624×624)</option></select></label>
    <label>Frames <select id="video-frames"><option value="17">17 (~1s)</option><option value="33" selected>33 (~2s)</option><option value="49">49 (~3s)</option><option value="65">65 (~4s)</option><option value="81">81 (~5s)</option></select></label>
    <label>Steps <input type="number" id="video-steps" min="10" max="50" value="30"></label>
  </div>
  <div class="progress" id="video-progress"><div class="progress-bar"></div></div>
  <div class="output" id="video-output"></div>
</div>

<div class="status" id="status"></div>

<script>
// --- Model state tracking (image and video are separate models now) ---
const modelState = { llm: 'off', tts: 'off', image: 'off', video: 'off' };

// --- Tabs ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');

    const modelType = tab.dataset.model;
    if (modelState[modelType] === 'off') {
      loadModel(modelType, modelType === 'video' ? { variant: document.getElementById('video-variant').value } : {});
    }
  });
});

async function loadModel(type, extra = {}) {
  modelState[type] = 'loading';
  updateTabs();
  updateStatus();

  try {
    const res = await fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, ...extra }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    modelState[type] = 'on';
  } catch (e) {
    modelState[type] = 'error';
    console.error('Load failed:', type, e.message);
  }
  updateTabs();
  updateStatus();
  await checkStatus();
}

function updateTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    const modelType = tab.dataset.model;
    const state = modelState[modelType];
    tab.classList.remove('loading', 'ready', 'error');
    if (state === 'loading') tab.classList.add('loading');
    else if (state === 'on') tab.classList.add('ready');
    else if (state === 'error') tab.classList.add('error');
  });
}

// --- Status ---
async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();

    const ttsEngine = s.tts && s.tts.engine ? s.tts.engine : null;
    const ttsOn = s.tts && (s.tts.loaded || s.tts === true);

    if (s.llm) modelState.llm = 'on';
    if (ttsOn) modelState.tts = 'on';
    if (s.image) modelState.image = 'on';
    if (s.video && s.video.loaded) modelState.video = 'on';
    if (s.llmLoading) modelState.llm = 'loading';
    if (s.ttsLoading) modelState.tts = 'loading';
    if (s.imageLoading) modelState.image = 'loading';
    if (s.videoLoading) modelState.video = 'loading';

    updateTabs();

    if (ttsEngine === 'qwen3') {
      document.getElementById('tts-options-kokoro').classList.add('hidden');
      document.getElementById('tts-options-qwen3').classList.remove('hidden');
    } else {
      document.getElementById('tts-options-kokoro').classList.remove('hidden');
      document.getElementById('tts-options-qwen3').classList.add('hidden');
    }
  } catch {}
  updateStatus();
}

function updateStatus() {
  const el = document.getElementById('status');
  el.innerHTML = ['llm', 'tts', 'image', 'video'].map(k => {
    const state = modelState[k];
    const cls = state === 'on' ? 'on' : state === 'loading' ? 'loading-text' : 'off';
    const label = state === 'loading' ? 'loading...' : state;
    return '<span class="' + cls + '">' + k.toUpperCase() + ': ' + label + '</span>';
  }).join(' &middot; ');
}

checkStatus();
setInterval(checkStatus, 3000);

function progress(id, on) {
  document.getElementById(id).classList.toggle('active', on);
}

// --- Chat ---
const chatMessages = [];
const chatEl = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');

function renderChat() {
  chatEl.innerHTML = chatMessages.map(m =>
    '<div class="msg msg-' + m.role + '"><div class="msg-label">' + m.role + '</div>' + escHtml(m.content) + '</div>'
  ).join('');
  chatEl.scrollTop = chatEl.scrollHeight;
}

chatSend.addEventListener('click', async () => {
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = '';

  chatMessages.push({ role: 'user', content: text });
  chatMessages.push({ role: 'assistant', content: '' });
  renderChat();
  chatSend.disabled = true;
  progress('chat-progress', true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: chatMessages.slice(0, -1) }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const d = JSON.parse(line.slice(6));
        if (d.error) { chatMessages[chatMessages.length - 1].content += '\\n[Error: ' + d.error + ']'; }
        if (d.content) { chatMessages[chatMessages.length - 1].content += d.content; }
        renderChat();
      }
    }
  } catch (e) {
    chatMessages[chatMessages.length - 1].content += '\\n[Error: ' + e.message + ']';
    renderChat();
  }
  progress('chat-progress', false);
  chatSend.disabled = false;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); chatSend.click(); }
});

// --- TTS ---
document.getElementById('tts-send').addEventListener('click', async () => {
  const text = document.getElementById('tts-input').value.trim();
  if (!text) return;
  const btn = document.getElementById('tts-send');
  btn.disabled = true;
  progress('tts-progress', true);
  document.getElementById('tts-output').innerHTML = '';

  const body = { text };
  const refAudio = document.getElementById('tts-ref-audio').value.trim();
  const voice = document.getElementById('tts-voice').value.trim();
  const speed = document.getElementById('tts-speed').value.trim();

  if (refAudio) body.referenceAudioPath = refAudio;
  if (voice) body.voice = voice;
  if (speed) body.speed = parseFloat(speed);

  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('tts-output').innerHTML = '<audio controls autoplay src="' + url + '"></audio>';
  } catch (e) {
    document.getElementById('tts-output').innerHTML = '<span class="error">' + escHtml(e.message) + '</span>';
  }
  progress('tts-progress', false);
  btn.disabled = false;
});

// --- Image ---
document.getElementById('image-send').addEventListener('click', async () => {
  const prompt = document.getElementById('image-input').value.trim();
  if (!prompt) return;
  const btn = document.getElementById('image-send');
  btn.disabled = true;
  progress('image-progress', true);
  document.getElementById('image-output').innerHTML = '';

  try {
    const res = await fetch('/api/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        width: parseInt(document.getElementById('image-w').value) || undefined,
        height: parseInt(document.getElementById('image-h').value) || undefined,
        steps: parseInt(document.getElementById('image-steps').value) || undefined,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('image-output').innerHTML = '<img src="' + url + '" alt="Generated">';
  } catch (e) {
    document.getElementById('image-output').innerHTML = '<span class="error">' + escHtml(e.message) + '</span>';
  }
  progress('image-progress', false);
  btn.disabled = false;
});

// --- Video ---
const videoDefaults = { '5b': { steps: 30, min: 10, max: 50 }, 'turbo-a14b': { steps: 10, min: 4, max: 30 } };
document.getElementById('video-variant').addEventListener('change', (e) => {
  const d = videoDefaults[e.target.value] || videoDefaults['5b'];
  const el = document.getElementById('video-steps');
  el.min = d.min; el.max = d.max; el.value = d.steps;
});

let videoFrames = [];
let videoIdx = 0;
let videoTimer = null;

document.getElementById('video-send').addEventListener('click', async () => {
  const prompt = document.getElementById('video-input').value.trim();
  if (!prompt) return;
  const btn = document.getElementById('video-send');
  btn.disabled = true;
  progress('video-progress', true);
  document.getElementById('video-output').innerHTML = '';
  if (videoTimer) { clearInterval(videoTimer); videoTimer = null; }

  try {
    const res = await fetch('/api/video', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        variant: document.getElementById('video-variant').value,
        aspect: document.getElementById('video-aspect').value,
        videoFrames: parseInt(document.getElementById('video-frames').value),
        steps: parseInt(document.getElementById('video-steps').value),
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error);
    }
    const data = await res.json();
    videoFrames = data.frames;
    videoIdx = 0;

    const out = document.getElementById('video-output');
    out.innerHTML = '<div class="video-player"><img id="video-canvas" src="data:image/png;base64,' + videoFrames[0] + '"><div class="video-controls"><button class="btn" id="video-play">Play</button><button class="btn" id="video-pause">Pause</button><span id="video-counter">1/' + videoFrames.length + '</span></div></div>';

    document.getElementById('video-play').addEventListener('click', () => {
      if (videoTimer) return;
      videoTimer = setInterval(() => {
        videoIdx = (videoIdx + 1) % videoFrames.length;
        document.getElementById('video-canvas').src = 'data:image/png;base64,' + videoFrames[videoIdx];
        document.getElementById('video-counter').textContent = (videoIdx + 1) + '/' + videoFrames.length;
      }, 100);
    });
    document.getElementById('video-pause').addEventListener('click', () => {
      if (videoTimer) { clearInterval(videoTimer); videoTimer = null; }
    });
  } catch (e) {
    document.getElementById('video-output').innerHTML = '<span class="error">' + escHtml(e.message) + '</span>';
  }
  progress('video-progress', false);
  btn.disabled = false;
});

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>`;
