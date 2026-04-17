import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { existsSync, statSync, mkdirSync, createWriteStream, renameSync, unlinkSync } from 'node:fs';
import * as path from 'node:path';
import { createModel, getSystemStatus } from './src/index.ts';
import type { LlmModel, TtsModel, ImageModel, ChatMessage } from './src/types.ts';

// --- Config ---
const PORT = Number(process.env.PORT ?? 3333);
const MODELS_DIR = process.env.MODELS_DIR ?? `${process.env.HOME}/.orcha/workspace/.models`;

const LLM_PATH = process.env.LLM_MODEL ?? `${MODELS_DIR}/qwen3-5-4b/Qwen3.5-4B-IQ4_NL.gguf`;
const LLM_MMPROJ_PATH = process.env.LLM_MMPROJ ?? `${MODELS_DIR}/qwen3-5-4b/mmproj-F16.gguf`;
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

// --- Model file registry (for detection & download UI) ---

interface ModelFileInfo { path: string; label: string; expectedSize: string; url: string }
interface ModelGroup { id: string; label: string; files: ModelFileInfo[] }

const MODEL_GROUPS: ModelGroup[] = [
  {
    id: 'base', label: 'Base (LLM)',
    files: [
      { path: `${MODELS_DIR}/qwen3-5-4b/Qwen3.5-4B-IQ4_NL.gguf`, label: 'Qwen3.5-4B IQ4_NL', expectedSize: '~2.5 GB', url: 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-IQ4_NL.gguf' },
      { path: `${MODELS_DIR}/qwen3-5-4b/mmproj-F16.gguf`, label: 'Qwen3.5-4B mmproj F16 (vision)', expectedSize: '~641 MB', url: 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf' },
      { path: `${MODELS_DIR}/tinyllama/tinyllama.gguf`, label: 'TinyLlama 1.1B (test)', expectedSize: '~250 MB', url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf' },
    ],
  },
  {
    id: 'tts', label: 'TTS (Qwen3-TTS, voice cloning)',
    files: [
      { path: `${MODELS_DIR}/qwen3-tts/qwen3-tts-0.6b-f16.gguf`, label: 'Qwen3-TTS 0.6B F16', expectedSize: '~1 GB', url: 'https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-0.6b-f16.gguf' },
      { path: `${MODELS_DIR}/qwen3-tts/qwen3-tts-tokenizer-f16.gguf`, label: 'Qwen3-TTS Tokenizer', expectedSize: '~200 MB', url: 'https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-tokenizer-f16.gguf' },
    ],
  },
  {
    id: 'image', label: 'Image (FLUX 2 Klein)',
    files: [
      { path: `${MODELS_DIR}/flux2-klein/flux-2-klein-4b-Q4_K_M.gguf`, label: 'FLUX 2 Klein 4B', expectedSize: '~2.5 GB', url: 'https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf' },
      { path: `${MODELS_DIR}/flux2-klein/Qwen3-4B-Q4_K_M.gguf`, label: 'Qwen3-4B (FLUX LLM)', expectedSize: '~2.2 GB', url: 'https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf' },
      { path: `${MODELS_DIR}/flux2-klein/flux2-vae.safetensors`, label: 'FLUX 2 VAE', expectedSize: '~300 MB', url: 'https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors' },
    ],
  },
  {
    id: 'video', label: 'Video (WAN 2.2 5B)',
    files: [
      { path: `${MODELS_DIR}/wan22-5b/Wan2.2-TI2V-5B-Q4_K_M.gguf`, label: 'WAN 2.2 TI2V 5B', expectedSize: '~3.5 GB', url: 'https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q4_K_M.gguf' },
      { path: `${MODELS_DIR}/wan22-5b/Wan2.2_VAE.safetensors`, label: 'WAN 2.2 VAE', expectedSize: '~300 MB', url: 'https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors' },
      { path: `${MODELS_DIR}/wan22-5b/umt5-xxl-encoder-Q8_0.gguf`, label: 'UMT5-XXL Encoder', expectedSize: '~6 GB', url: 'https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q8_0.gguf' },
    ],
  },
];

let activeDownload = false;

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

  const hasVisionProj = existsSync(LLM_MMPROJ_PATH);
  console.log(`Loading LLM: ${LLM_PATH}${hasVisionProj ? ` + mmproj: ${LLM_MMPROJ_PATH}` : ''}`);
  loading.llm = (async () => {
    const m = createModel(LLM_PATH, 'llm');
    await m.load({
      contextSize: hasVisionProj ? 8192 : 4096,
      ...(hasVisionProj ? {
        mmprojPath: LLM_MMPROJ_PATH,
        imageMaxTokens: 1024,
      } : {}),
    });
    models.llm = m;
    console.log(`LLM ready (vision: ${m.hasVision})`);
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
    await m.load();
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
  // Convert base64 image data to Buffers for the native binding
  const messages: ChatMessage[] = (body.messages ?? []).map((m: any) => {
    if (typeof m.content === 'string') return m;
    return {
      ...m,
      content: m.content.map((part: any) => {
        if (part.type === 'image' && typeof part.data === 'string') {
          return { type: 'image', data: Buffer.from(part.data, 'base64') };
        }
        return part;
      }),
    };
  });

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  try {
    for await (const chunk of models.llm.stream(messages, {
      temperature: body.temperature ?? 0.7,
      maxTokens: body.maxTokens ?? 2048,
      thinkingBudget: body.thinkingBudget ?? 0,
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
  if (!models.tts) return errorResponse(res, 'TTS not loaded. Load Qwen3-TTS first.', 503);

  const body = JSON.parse(await readBody(req));
  const text = body.text ?? '';
  if (!text) return errorResponse(res, 'Missing "text" field', 400);

  try {
    const speakOpts: Record<string, unknown> = {
      referenceAudioPath: body.referenceAudioPath ?? '',
      temperature: body.temperature ?? undefined,
    };

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
      width: body.width ?? 1024,
      height: body.height ?? 1024,
      steps: body.steps ?? 4,
      cfgScale: body.cfgScale ?? 1.5,
      strength: body.strength ?? 1.0,
      sampleMethod: body.sampleMethod ?? 'euler',
      scheduler: body.scheduler ?? 'discrete',
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
  const system = getSystemStatus();

  const modelStatuses = [];
  if (models.llm) {
    modelStatuses.push(models.llm.getStatus());
  }
  if (models.tts) {
    modelStatuses.push(models.tts.getStatus());
  }
  if (models.image) {
    modelStatuses.push(models.image.getStatus());
  }
  if (models.video) {
    modelStatuses.push({ ...models.video.getStatus(), variant: videoVariant });
  }

  json(res, {
    system,
    models: {
      llm: models.llm ? models.llm.getStatus() : { loaded: false, loading: loading.llm !== null },
      tts: models.tts ? models.tts.getStatus() : { loaded: false, loading: loading.tts !== null },
      image: models.image ? models.image.getStatus() : { loaded: false, loading: loading.image !== null },
      video: models.video
        ? { ...models.video.getStatus(), variant: videoVariant }
        : { loaded: false, loading: loading.video !== null },
    },
    activeInference: modelStatuses.filter(m => m.busy).map(m => m.type),
    loadedCount: modelStatuses.filter(m => m.loaded).length,
    totalModels: 4,
  });
}

function handleModelsApi(_req: IncomingMessage, res: ServerResponse) {
  const groups = MODEL_GROUPS.map(group => ({
    id: group.id,
    label: group.label,
    files: group.files.map(file => {
      let exists = false;
      let bytes = 0;
      try {
        const s = statSync(file.path);
        exists = true;
        bytes = s.size;
      } catch {}
      return { label: file.label, expectedSize: file.expectedSize, exists, bytes };
    }),
  }));
  json(res, { modelsDir: MODELS_DIR, groups });
}

async function downloadFile(
  url: string,
  dest: string,
  onProgress: (text: string) => void,
): Promise<void> {
  mkdirSync(path.dirname(dest), { recursive: true });
  const tmpDest = dest + '.download';

  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  if (!res.body) throw new Error('No response body');

  const totalBytes = Number(res.headers.get('content-length') || 0);
  let downloadedBytes = 0;
  let lastPct = -1;

  const fileStream = createWriteStream(tmpDest);
  const reader = res.body.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fileStream.write(value);
      downloadedBytes += value.byteLength;

      if (totalBytes > 0) {
        const pct = Math.floor((downloadedBytes / totalBytes) * 100);
        if (pct !== lastPct) {
          lastPct = pct;
          const dlMB = (downloadedBytes / 1048576).toFixed(0);
          const totalMB = (totalBytes / 1048576).toFixed(0);
          onProgress(`\r  ${pct}% (${dlMB}/${totalMB} MB)`);
        }
      }
    }

    await new Promise<void>((resolve, reject) => {
      fileStream.end(() => resolve());
      fileStream.on('error', reject);
    });

    renameSync(tmpDest, dest);
  } catch (e) {
    fileStream.destroy();
    try { unlinkSync(tmpDest); } catch {}
    throw e;
  }
}

async function handleDownloadApi(req: IncomingMessage, res: ServerResponse) {
  const reqUrl = new URL(req.url!, `http://${req.headers.host}`);
  const groupId = reqUrl.searchParams.get('group') ?? 'base';

  const groups = groupId === 'all'
    ? MODEL_GROUPS
    : MODEL_GROUPS.filter(g => g.id === groupId);
  if (groups.length === 0) return errorResponse(res, `Unknown group: ${groupId}`, 400);
  if (activeDownload) return errorResponse(res, 'Download already in progress', 409);

  activeDownload = true;

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  });

  const send = (text: string) => {
    try { res.write(`data: ${JSON.stringify({ text })}\n\n`); } catch {}
  };

  let exitCode = 0;
  try {
    for (const group of groups) {
      send(`=== ${group.label} ===\n`);
      for (const file of group.files) {
        if (existsSync(file.path)) {
          send(`Already exists: ${file.label}\n`);
          continue;
        }
        send(`Downloading ${file.label} (${file.expectedSize})...\n`);
        await downloadFile(file.url, file.path, send);
        send(`\nDownloaded: ${file.label}\n`);
      }
    }
    send(`\nDone. Models directory: ${MODELS_DIR}\n`);
  } catch (e: any) {
    send(`\nError: ${e.message}\n`);
    exitCode = 1;
  }

  activeDownload = false;
  try {
    res.write(`data: ${JSON.stringify({ done: true, code: exitCode })}\n\n`);
    res.end();
  } catch {}
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
  if (method === 'GET' && url === '/status') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(STATUS_HTML);
    return;
  }
  if (method === 'GET' && url === '/api/status') return handleStatus(req, res);
  if (method === 'GET' && url === '/api/models') return handleModelsApi(req, res);
  if (method === 'GET' && url.startsWith('/api/models/download')) return handleDownloadApi(req, res);
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

server.listen(PORT, '0.0.0.0', () => {
  console.log(`node-omni-orcha server on http://localhost:${PORT}`);
  console.log(`Status dashboard: http://localhost:${PORT}/status`);
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
  .status-link { font-size: 0.75rem; color: #4a9eff; text-decoration: none; margin-left: 8px; font-weight: 400; }
  .status-link:hover { text-decoration: underline; }

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
  .chat-messages img { max-width: 200px; max-height: 200px; border-radius: 4px; margin-top: 4px; display: block; }

  .chat-attachments { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
  .chat-attachment { position: relative; border: 1px solid #333; border-radius: 4px; overflow: hidden; }
  .chat-attachment img { display: block; width: 80px; height: 80px; object-fit: cover; }
  .chat-attachment-remove { position: absolute; top: 2px; right: 2px; width: 18px; height: 18px; background: rgba(0,0,0,0.7); border: none; color: #fff; border-radius: 50%; cursor: pointer; font-size: 12px; line-height: 18px; text-align: center; padding: 0; }
  .chat-attachment-remove:hover { background: #c93027; }
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

  .models-grid { display: flex; flex-direction: column; gap: 8px; }
  .model-group { border: 1px solid #222; border-radius: 4px; padding: 10px; background: #111; }
  .model-group.complete { border-color: #2a4a2a; }
  .model-group.partial { border-color: #4a3a1a; }
  .model-group-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .model-group-title { font-size: 0.9rem; font-weight: 500; }
  .model-files { display: flex; flex-direction: column; gap: 3px; }
  .model-file { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #888; }
  .model-file-status.ok { color: #4a4; }
  .model-file-status.missing { color: #633; }
  .model-file-size { margin-left: auto; font-size: 0.75rem; color: #555; }
  .btn-sm { padding: 4px 10px; font-size: 0.8rem; }
  .btn.downloading { border-color: #4a9eff; color: #4a9eff; animation: pulse 1.5s infinite; }
  .models-actions { display: flex; gap: 8px; margin-top: 10px; }
  .download-log { background: #0a0a0a; border: 1px solid #222; border-radius: 4px; padding: 8px; margin-top: 8px; font-family: monospace; font-size: 0.75rem; color: #888; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; display: none; }
  .download-log.active { display: block; }
  .models-dir { font-size: 0.75rem; color: #555; margin-bottom: 8px; word-break: break-all; }

</style>
</head>
<body>
<h1>node-omni-orcha <a href="/status" class="status-link">status</a></h1>

<div class="tabs">
  <button class="tab active" data-panel="models">Models</button>
  <button class="tab" data-panel="chat" data-model="llm">Chat</button>
  <button class="tab" data-panel="tts" data-model="tts">TTS</button>
  <button class="tab" data-panel="image" data-model="image">Image</button>
  <button class="tab" data-panel="video" data-model="video">Video</button>
</div>

<!-- Models -->
<div class="panel active" id="models">
  <div class="models-dir" id="models-dir"></div>
  <div class="models-grid" id="models-grid"></div>
  <div class="models-actions">
    <button class="btn btn-sm dl-btn" data-group="all">Download All</button>
    <button class="btn btn-sm" id="models-refresh">Refresh</button>
  </div>
  <pre class="download-log" id="download-log"></pre>
</div>

<!-- Chat -->
<div class="panel" id="chat">
  <div class="chat-messages" id="chat-messages"></div>
  <div class="progress" id="chat-progress"><div class="progress-bar"></div></div>
  <div class="chat-attachments" id="chat-attachments"></div>
  <div class="input-row">
    <label class="btn hidden" id="chat-browse-label" title="Attach image (PNG, JPEG, GIF, WebP, BMP)">&#128206;
      <input type="file" id="chat-file" accept="image/png,image/jpeg,image/gif,image/webp,image/bmp" multiple class="hidden">
    </label>
    <textarea id="chat-input" placeholder="Type a message... (attach images with the clip button)"></textarea>
    <button class="btn" id="chat-send">Send</button>
  </div>
</div>

<!-- TTS -->
<div class="panel" id="tts">
  <div class="input-row">
    <textarea id="tts-input" placeholder="Text to speak..."></textarea>
    <button class="btn" id="tts-send">Speak</button>
  </div>
  <div class="options-row">
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
    <label>Resolution <select id="image-res"><option value="1024x1024" selected>1024×1024</option><option value="896x1152">896×1152</option><option value="1152x896">1152×896</option><option value="1536x1024">1536×1024</option></select></label>
    <label>Steps <input type="number" id="image-steps" min="1" max="50" value="4"></label>
    <label>Sampler <select id="image-sampler"><option value="euler" selected>Euler</option><option value="euler_a">Euler A</option><option value="heun">Heun</option><option value="dpm2">DPM2</option><option value="dpmpp2s_a">DPM++ 2S A</option><option value="dpmpp2m">DPM++ 2M</option><option value="dpmpp2mv2">DPM++ 2M v2</option><option value="lcm">LCM</option></select></label>
    <label>Scheduler <select id="image-scheduler"><option value="discrete">Normal</option><option value="karras">Karras</option><option value="exponential">Exponential</option><option value="ays">AYS</option><option value="sgm_uniform">SGM Uniform</option><option value="simple">Simple</option></select></label>
    <label>CFG <input type="number" id="image-cfg" min="0" max="30" step="0.5" value="1.5"></label>
    <label>Denoise <input type="number" id="image-denoise" min="0" max="1" step="0.05" value="1.0"></label>
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
// --- Models panel ---

function fmtBytes(b) {
  if (!b) return '';
  var u = ['B', 'KB', 'MB', 'GB'];
  var i = b > 0 ? Math.floor(Math.log(b) / Math.log(1024)) : 0;
  return (b / Math.pow(1024, i)).toFixed(1) + ' ' + u[i];
}

function renderModels(data) {
  document.getElementById('models-dir').textContent = 'Models directory: ' + data.modelsDir;
  var grid = document.getElementById('models-grid');
  grid.innerHTML = data.groups.map(function(group) {
    var allOk = group.files.every(function(f) { return f.exists; });
    var someOk = group.files.some(function(f) { return f.exists; });
    var cls = allOk ? 'complete' : someOk ? 'partial' : '';

    return '<div class="model-group ' + cls + '">' +
      '<div class="model-group-header">' +
        '<span class="model-group-title">' + escHtml(group.label) + '</span>' +
        '<button class="btn btn-sm dl-btn" data-group="' + group.id + '"' + (allOk ? ' disabled' : '') + '>' +
          (allOk ? 'Downloaded' : 'Download') +
        '</button>' +
      '</div>' +
      '<div class="model-files">' +
        group.files.map(function(f) {
          return '<div class="model-file">' +
            '<span class="model-file-status ' + (f.exists ? 'ok' : 'missing') + '">' + (f.exists ? '\\u25cf' : '\\u25cb') + '</span>' +
            '<span class="model-file-label">' + escHtml(f.label) + '</span>' +
            '<span class="model-file-size">' + (f.exists ? fmtBytes(f.bytes) : f.expectedSize) + '</span>' +
          '</div>';
        }).join('') +
      '</div>' +
    '</div>';
  }).join('');
}

async function loadModelsStatus() {
  try {
    var res = await fetch('/api/models');
    var data = await res.json();
    renderModels(data);
  } catch (e) {
    document.getElementById('models-grid').innerHTML = '<span class="error">Failed to load model status</span>';
  }
}

var downloadLogText = '';

async function downloadGroup(groupId) {
  var logEl = document.getElementById('download-log');
  downloadLogText = '';
  logEl.textContent = '';
  logEl.classList.add('active');

  document.querySelectorAll('.dl-btn').forEach(function(b) { b.disabled = true; });
  var btn = document.querySelector('.dl-btn[data-group="' + groupId + '"]');
  if (btn) { btn.textContent = 'Downloading...'; btn.classList.add('downloading'); }

  try {
    var res = await fetch('/api/models/download?group=' + groupId);
    if (!res.ok) {
      var err = await res.json();
      throw new Error(err.error);
    }

    var reader = res.body.getReader();
    var dec = new TextDecoder();
    var buf = '';

    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      var lines = buf.split('\\n');
      buf = lines.pop();
      for (var i = 0; i < lines.length; i++) {
        if (!lines[i].startsWith('data: ')) continue;
        var d = JSON.parse(lines[i].slice(6));
        if (d.text) {
          downloadLogText += d.text;
          var logLines = downloadLogText.split('\\n');
          var processed = logLines.map(function(line) {
            var parts = line.split('\\r');
            return parts[parts.length - 1];
          });
          logEl.textContent = processed.join('\\n');
          logEl.scrollTop = logEl.scrollHeight;
        }
        if (d.done) {
          logEl.textContent += '\\n' + (d.code === 0 ? 'Download complete.' : 'Download failed (exit ' + d.code + ').');
          logEl.scrollTop = logEl.scrollHeight;
        }
      }
    }
  } catch (e) {
    logEl.textContent += '\\nError: ' + e.message;
  }

  await loadModelsStatus();
}

document.getElementById('models-grid').addEventListener('click', function(e) {
  var btn = e.target.closest('.dl-btn');
  if (btn && !btn.disabled) downloadGroup(btn.dataset.group);
});

document.querySelector('.models-actions').addEventListener('click', function(e) {
  var btn = e.target.closest('.dl-btn');
  if (btn && !btn.disabled) downloadGroup(btn.dataset.group);
});

document.getElementById('models-refresh').addEventListener('click', loadModelsStatus);

loadModelsStatus();

// --- Model state tracking (image and video are separate models now) ---
const modelState = { llm: 'off', tts: 'off', image: 'off', video: 'off' };

// --- Tabs ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');

    if (tab.dataset.panel === 'models') { loadModelsStatus(); return; }
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
    const m = s.models || {};

    if (m.llm && m.llm.loaded) modelState.llm = 'on';
    if (m.tts && m.tts.loaded) modelState.tts = 'on';
    if (m.image && m.image.loaded) modelState.image = 'on';
    if (m.video && m.video.loaded) modelState.video = 'on';
    if (m.llm && m.llm.loading) modelState.llm = 'loading';
    if (m.tts && m.tts.loading) modelState.tts = 'loading';
    if (m.image && m.image.loading) modelState.image = 'loading';
    if (m.video && m.video.loading) modelState.video = 'loading';

    updateTabs();

    // Toggle image upload based on vision support
    const hasVision = m.llm && m.llm.hasVision;
    visionEnabled = !!hasVision;
    const browseLabel = document.getElementById('chat-browse-label');
    if (hasVision) {
      browseLabel.classList.remove('hidden');
      chatInput.placeholder = 'Type a message... (attach images with the clip button, paste, or drag & drop)';
    } else {
      browseLabel.classList.add('hidden');
      chatInput.placeholder = m.llm && m.llm.loaded
        ? 'Type a message... (load a vision model with mmproj for image support)'
        : 'Type a message...';
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
const chatFile = document.getElementById('chat-file');
const chatAttachments = document.getElementById('chat-attachments');
let pendingImages = []; // { dataUrl: string, base64: string, mime: string }
let visionEnabled = false;

function renderChat() {
  chatEl.innerHTML = chatMessages.map(function(m) {
    let html = '<div class="msg msg-' + m.role + '"><div class="msg-label">' + m.role + '</div>';
    if (typeof m.content === 'string') {
      html += escHtml(m.content);
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (part.type === 'text') html += escHtml(part.text);
        if (part.type === 'image' && part.dataUrl) html += '<img src="' + part.dataUrl + '">';
      }
    }
    html += '</div>';
    return html;
  }).join('');
  chatEl.scrollTop = chatEl.scrollHeight;
}

function renderPendingAttachments() {
  chatAttachments.innerHTML = pendingImages.map(function(img, i) {
    return '<div class="chat-attachment">'
      + '<img src="' + img.dataUrl + '">'
      + '<button class="chat-attachment-remove" data-idx="' + i + '">x</button>'
      + '</div>';
  }).join('');
}

chatAttachments.addEventListener('click', function(e) {
  if (e.target.classList.contains('chat-attachment-remove')) {
    pendingImages.splice(Number(e.target.dataset.idx), 1);
    renderPendingAttachments();
  }
});

chatFile.addEventListener('change', function() {
  Array.from(chatFile.files).forEach(function(file) {
    const reader = new FileReader();
    reader.onload = function() {
      const dataUrl = reader.result;
      const base64 = dataUrl.split(',')[1];
      const mime = file.type;
      pendingImages.push({ dataUrl: dataUrl, base64: base64, mime: mime });
      renderPendingAttachments();
    };
    reader.readAsDataURL(file);
  });
  chatFile.value = '';
});

// Allow drag-and-drop onto the chat input
chatInput.addEventListener('dragover', function(e) { if (visionEnabled) e.preventDefault(); });
chatInput.addEventListener('drop', function(e) {
  if (!visionEnabled) return;
  e.preventDefault();
  var files = Array.from(e.dataTransfer.files).filter(function(f) { return f.type.startsWith('image/'); });
  files.forEach(function(file) {
    var reader = new FileReader();
    reader.onload = function() {
      var dataUrl = reader.result;
      var base64 = dataUrl.split(',')[1];
      pendingImages.push({ dataUrl: dataUrl, base64: base64, mime: file.type });
      renderPendingAttachments();
    };
    reader.readAsDataURL(file);
  });
});

// Allow paste images
chatInput.addEventListener('paste', function(e) {
  if (!visionEnabled) return;
  var items = Array.from(e.clipboardData.items).filter(function(i) { return i.type.startsWith('image/'); });
  items.forEach(function(item) {
    var file = item.getAsFile();
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function() {
      var dataUrl = reader.result;
      var base64 = dataUrl.split(',')[1];
      pendingImages.push({ dataUrl: dataUrl, base64: base64, mime: file.type });
      renderPendingAttachments();
    };
    reader.readAsDataURL(file);
  });
});

chatSend.addEventListener('click', async () => {
  const text = chatInput.value.trim();
  if (!text && pendingImages.length === 0) return;
  chatInput.value = '';

  // Build user message content
  let userContent;
  if (pendingImages.length > 0) {
    userContent = [];
    for (const img of pendingImages) {
      userContent.push({ type: 'image', dataUrl: img.dataUrl, base64: img.base64, mime: img.mime });
    }
    if (text) userContent.push({ type: 'text', text: text });
    pendingImages = [];
    renderPendingAttachments();
  } else {
    userContent = text;
  }

  chatMessages.push({ role: 'user', content: userContent });
  chatMessages.push({ role: 'assistant', content: '' });
  renderChat();
  chatSend.disabled = true;
  progress('chat-progress', true);

  // Build messages for API — convert image parts to { type: 'image', data: base64 }
  const apiMessages = chatMessages.slice(0, -1).map(function(m) {
    if (typeof m.content === 'string') return { role: m.role, content: m.content };
    // Multimodal content — strip dataUrl (display-only), send base64 data
    return {
      role: m.role,
      content: m.content.map(function(p) {
        if (p.type === 'image') return { type: 'image', data: p.base64 };
        return p;
      }),
    };
  });

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: apiMessages }),
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

  if (refAudio) body.referenceAudioPath = refAudio;

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
    const [iw, ih] = document.getElementById('image-res').value.split('x').map(Number);
    const res = await fetch('/api/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        width: iw,
        height: ih,
        steps: parseInt(document.getElementById('image-steps').value) || 4,
        sampleMethod: document.getElementById('image-sampler').value,
        scheduler: document.getElementById('image-scheduler').value,
        cfgScale: parseFloat(document.getElementById('image-cfg').value) || 1.5,
        strength: parseFloat(document.getElementById('image-denoise').value) || 1.0,
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

// --- Status Dashboard HTML ---

const STATUS_HTML = /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>node-omni-orcha — Status</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 24px; }

  .container { max-width: 960px; margin: 0 auto; }

  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
  .header h1 { font-size: 1.2rem; font-weight: 500; color: #999; }
  .header .uptime { font-size: 0.8rem; color: #555; }
  .header a { color: #4a9eff; text-decoration: none; font-size: 0.85rem; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }

  .card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; font-weight: 500; }

  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #1a1a1a; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { font-size: 0.85rem; color: #888; }
  .stat-value { font-size: 0.85rem; color: #e0e0e0; font-variant-numeric: tabular-nums; }

  .bar-container { height: 6px; background: #1a1a1a; border-radius: 3px; margin-top: 4px; overflow: hidden; }
  .bar { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  .bar-ok { background: #2d8a4e; }
  .bar-warn { background: #c9a227; }
  .bar-danger { background: #c93027; }

  .models-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }

  .model-card { background: #0d0d0d; border: 1px solid #222; border-radius: 8px; padding: 14px; position: relative; }
  .model-card .model-type { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #666; margin-bottom: 6px; }
  .model-card .model-name { font-size: 0.85rem; color: #ccc; word-break: break-all; margin-bottom: 8px; min-height: 1.4em; }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
  .badge-loaded { background: #1a3d2a; color: #4ade80; border: 1px solid #2d5a3d; }
  .badge-loading { background: #1a2d4d; color: #4a9eff; border: 1px solid #2d4a6d; animation: pulse 1.5s infinite; }
  .badge-busy { background: #4d3a1a; color: #f5a623; border: 1px solid #6d5a2d; animation: pulse 1s infinite; }
  .badge-off { background: #1a1a1a; color: #555; border: 1px solid #2a2a2a; }

  .badges { display: flex; gap: 6px; flex-wrap: wrap; }

  .meta-row { font-size: 0.75rem; color: #555; margin-top: 6px; }

  .inference-banner { background: #1a2d1a; border: 1px solid #2d5a3d; border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; display: flex; align-items: center; gap: 10px; }
  .inference-banner.idle { background: #111; border-color: #222; }
  .inference-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: pulse 1s infinite; flex-shrink: 0; }
  .inference-dot.idle { background: #333; animation: none; }
  .inference-text { font-size: 0.85rem; color: #ccc; }
  .inference-text.idle { color: #555; }

  .footer { text-align: center; margin-top: 32px; font-size: 0.75rem; color: #333; }

  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  .refresh-indicator { width: 6px; height: 6px; border-radius: 50%; background: #333; display: inline-block; margin-left: 8px; transition: background 0.2s; }
  .refresh-indicator.active { background: #4a9eff; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>node-omni-orcha <span class="refresh-indicator" id="refresh-dot"></span></h1>
    <div>
      <span class="uptime" id="uptime"></span>
      <a href="/">main ui</a>
    </div>
  </div>

  <div id="inference-banner" class="inference-banner idle">
    <div id="inference-dot" class="inference-dot idle"></div>
    <div id="inference-text" class="inference-text idle">No active inference</div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>System</h2>
      <div id="system-info"></div>
    </div>
    <div class="card">
      <h2>CPU</h2>
      <div id="cpu-info"></div>
    </div>
    <div class="card">
      <h2>System Memory</h2>
      <div id="mem-info"></div>
    </div>
    <div class="card">
      <h2>Process Memory</h2>
      <div id="proc-mem-info"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom: 24px;">
    <h2>Models (<span id="model-count">0</span> / <span id="model-total">4</span> loaded)</h2>
    <div class="models-grid" id="models-grid"></div>
  </div>

  <div class="card">
    <h2>GPU</h2>
    <div id="gpu-info"></div>
  </div>

  <div class="footer">
    Polling every 2s &middot; <span id="last-update"></span>
  </div>
</div>

<script>
function fmtBytes(b) {
  if (b == null) return '—';
  if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
  if (b >= 1048576) return (b / 1048576).toFixed(0) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KB';
  return b + ' B';
}

function fmtUptime(s) {
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const parts = [];
  if (d > 0) parts.push(d + 'd');
  if (h > 0) parts.push(h + 'h');
  parts.push(m + 'm');
  return parts.join(' ');
}

function barClass(pct) {
  if (pct < 70) return 'bar-ok';
  if (pct < 90) return 'bar-warn';
  return 'bar-danger';
}

function statRow(label, value) {
  return '<div class="stat-row"><span class="stat-label">' + label + '</span><span class="stat-value">' + value + '</span></div>';
}

function barRow(label, value, pct) {
  return '<div class="stat-row"><span class="stat-label">' + label + '</span><span class="stat-value">' + value + '</span></div>'
    + '<div class="bar-container"><div class="bar ' + barClass(pct) + '" style="width:' + Math.min(pct, 100) + '%"></div></div>';
}

function modelName(path) {
  if (!path) return '—';
  const parts = path.split('/');
  return parts[parts.length - 1] || parts[parts.length - 2] || path;
}

function renderModels(models) {
  const grid = document.getElementById('models-grid');
  const types = ['llm', 'tts', 'image', 'video'];
  let html = '';

  for (const t of types) {
    const m = models[t];
    if (!m) continue;

    const badges = [];
    if (m.busy) badges.push('<span class="badge badge-busy">inferring</span>');
    else if (m.loaded) badges.push('<span class="badge badge-loaded">loaded</span>');
    else if (m.loading) badges.push('<span class="badge badge-loading">loading</span>');
    else badges.push('<span class="badge badge-off">off</span>');

    if (m.hasVision) badges.push('<span class="badge badge-loaded">vision</span>');

    let meta = '';
    if (m.metadata) {
      const md = m.metadata;
      meta += '<div class="meta-row">' + md.architecture;
      if (md.contextLength) meta += ' &middot; ctx ' + md.contextLength;
      if (md.fileSizeBytes) meta += ' &middot; ' + fmtBytes(md.fileSizeBytes);
      meta += '</div>';
    }
    if (m.variant) {
      meta += '<div class="meta-row">variant: ' + m.variant + '</div>';
    }

    html += '<div class="model-card">'
      + '<div class="model-type">' + t.toUpperCase() + '</div>'
      + '<div class="model-name">' + (m.modelPath ? modelName(m.modelPath) : '—') + '</div>'
      + '<div class="badges">' + badges.join('') + '</div>'
      + meta
      + '</div>';
  }

  grid.innerHTML = html;
}

async function poll() {
  const dot = document.getElementById('refresh-dot');
  dot.classList.add('active');
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    const sys = s.system;

    // System info
    document.getElementById('system-info').innerHTML =
      statRow('Platform', sys.platform + ' ' + sys.arch)
      + statRow('Hostname', sys.hostname)
      + statRow('Node.js', sys.nodeVersion)
      + statRow('OS Uptime', fmtUptime(sys.osUptimeSeconds));

    // CPU
    document.getElementById('cpu-info').innerHTML =
      statRow('Model', sys.cpu.model)
      + statRow('Cores / Threads', sys.cpu.cores + ' / ' + sys.cpu.threads)
      + statRow('Speed', sys.cpu.speed + ' MHz');

    // System memory
    const memPct = sys.memory.usagePercent;
    document.getElementById('mem-info').innerHTML =
      barRow('Used', fmtBytes(sys.memory.usedBytes) + ' / ' + fmtBytes(sys.memory.totalBytes), memPct)
      + statRow('Free', fmtBytes(sys.memory.freeBytes))
      + statRow('Usage', memPct.toFixed(1) + '%');

    // Process memory
    const pm = sys.processMemory;
    document.getElementById('proc-mem-info').innerHTML =
      statRow('RSS', fmtBytes(pm.rssBytes))
      + statRow('Heap Used', fmtBytes(pm.heapUsedBytes) + ' / ' + fmtBytes(pm.heapTotalBytes))
      + statRow('External (native)', fmtBytes(pm.externalBytes));

    // GPU
    const g = sys.gpu;
    let gpuHtml = statRow('Backend', g.backend.toUpperCase());
    if (g.name) gpuHtml += statRow('Name', g.name);
    if (g.vramBytes) gpuHtml += statRow('VRAM', fmtBytes(g.vramBytes));
    document.getElementById('gpu-info').innerHTML = gpuHtml;

    // Uptime
    document.getElementById('uptime').textContent = 'process up ' + fmtUptime(sys.uptimeSeconds);

    // Models
    document.getElementById('model-count').textContent = s.loadedCount;
    document.getElementById('model-total').textContent = s.totalModels;
    renderModels(s.models);

    // Inference banner
    const active = s.activeInference || [];
    const banner = document.getElementById('inference-banner');
    const infDot = document.getElementById('inference-dot');
    const infText = document.getElementById('inference-text');
    if (active.length > 0) {
      banner.classList.remove('idle');
      infDot.classList.remove('idle');
      infText.classList.remove('idle');
      infText.textContent = 'Active: ' + active.map(function(t) { return t.toUpperCase(); }).join(', ');
    } else {
      banner.classList.add('idle');
      infDot.classList.add('idle');
      infText.classList.add('idle');
      infText.textContent = 'No active inference';
    }

    document.getElementById('last-update').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('last-update').textContent = 'fetch error: ' + e.message;
  }
  setTimeout(function() { dot.classList.remove('active'); }, 300);
}

poll();
setInterval(poll, 2000);
</script>
</body>
</html>`;
