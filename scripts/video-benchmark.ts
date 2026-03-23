import { createModel } from '../src/index.ts';
import type { ImageModel } from '../src/types.ts';
import { writeFileSync, mkdirSync, statSync, existsSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join } from 'node:path';

const MODELS_DIR = process.env.MODELS_DIR ?? `${process.env.HOME}/.orcha/workspace/.models`;
const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const OUTPUT = join(import.meta.dirname, '..', 'test-output', ts);

const WAN_NEGATIVE_PROMPT = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走';

const VARIANTS = {
  '5b': {
    dir: `${MODELS_DIR}/wan22-5b`,
    model: 'Wan2.2-TI2V-5B-Q8_0.gguf',
    vae: 'Wan2.2_VAE.safetensors',
    t5xxl: 'umt5-xxl-encoder-Q8_0.gguf',
  },
  'a14b': {
    dir: `${MODELS_DIR}/wan22-turbo`,
    model: 'Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf',
    highNoise: 'Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf',
    vae: 'Wan2.1_VAE.safetensors',
    t5xxl: 'umt5-xxl-encoder-Q8_0.gguf',
  },
} as const;

type VariantKey = keyof typeof VARIANTS;

interface TestCase {
  name: string;
  variant: VariantKey;
  frames: number;
  steps: number;
  cfgScale: number;
  flowShift: number;
  highNoiseSteps?: number;
  highNoiseCfgScale?: number;
}

const TESTS: TestCase[] = [
  //{ name: 'quick-preview-5b-17f-15s',    variant: '5b',   frames: 17, steps: 15, cfgScale: 6.0, flowShift: 3.0 },
  //{ name: 'quick-preview-a14b-17f-10s',  variant: 'a14b', frames: 17, steps: 10, cfgScale: 3.5, flowShift: 3.0, highNoiseSteps: 8, highNoiseCfgScale: 3.5 },
  { name: 'quick-preview-a14b-81f-5s',  variant: 'a14b', frames: 81, steps: 3, cfgScale: 3.5, flowShift: 3.0, highNoiseSteps: 3, highNoiseCfgScale: 3.5 },
  //{ name: 'balanced-5b-33f-30s',          variant: '5b',   frames: 33, steps: 30, cfgScale: 6.0, flowShift: 3.0 },
  //{ name: 'balanced-a14b-33f-10s',        variant: 'a14b', frames: 33, steps: 10, cfgScale: 3.5, flowShift: 3.0, highNoiseSteps: 8, highNoiseCfgScale: 3.5 },
  //{ name: 'high-quality-5b-49f-40s',      variant: '5b',   frames: 49, steps: 40, cfgScale: 6.0, flowShift: 3.0 },
  //{ name: 'high-quality-a14b-49f-15s',    variant: 'a14b', frames: 49, steps: 15, cfgScale: 3.5, flowShift: 3.0, highNoiseSteps: 8, highNoiseCfgScale: 3.5 },
  //{ name: 'maximum-5b-81f-30s',           variant: '5b',   frames: 81, steps: 30, cfgScale: 6.0, flowShift: 3.0 },
  //{ name: 'maximum-a14b-81f-10s',         variant: 'a14b', frames: 81, steps: 10, cfgScale: 3.5, flowShift: 3.0, highNoiseSteps: 8, highNoiseCfgScale: 3.5 },
];

const PROMPT = 'border collie dog playing with a frisbee, cinematic lighting';
const WIDTH = 832;
const HEIGHT = 480;
const SEED = 42;

function saveFrames(frames: Buffer[], name: string) {
  const outDir = join(OUTPUT, name);
  mkdirSync(outDir, { recursive: true });

  for (let i = 0; i < frames.length; i++) {
    writeFileSync(join(outDir, `frame_${String(i).padStart(4, '0')}.png`), frames[i]);
  }

  const mp4Path = join(OUTPUT, `${name}.mp4`);
  try {
    execSync(
      `ffmpeg -y -framerate 16 -i "${outDir}/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "${mp4Path}"`,
      { stdio: 'pipe' },
    );
    const size = (statSync(mp4Path).size / 1024).toFixed(0);
    console.log(`  MP4: ${mp4Path} (${size}KB)`);
  } catch {
    console.log('  ffmpeg not available, skipping MP4');
  }
}

const filter = process.argv[2];
const testsToRun = filter
  ? TESTS.filter(t => t.name.includes(filter) || t.variant === filter)
  : TESTS;

const grouped = new Map<VariantKey, TestCase[]>();
for (const t of testsToRun) {
  const list = grouped.get(t.variant) ?? [];
  list.push(t);
  grouped.set(t.variant, list);
}

mkdirSync(OUTPUT, { recursive: true });
console.log(`\nVideo Benchmark — ${testsToRun.length} tests, prompt: "${PROMPT}"\n`);

const results: { name: string; frames: number; steps: number; elapsed: string; avgFrameKB: string }[] = [];

for (const [variant, tests] of grouped) {
  const v = VARIANTS[variant];
  const modelPath = `${v.dir}/${v.model}`;

  if (!existsSync(modelPath)) {
    console.log(`Skipping ${variant} — model not found: ${modelPath}\n`);
    continue;
  }

  const highNoisePath = 'highNoise' in v ? `${v.dir}/${v.highNoise}` : undefined;
  if (highNoisePath && !existsSync(highNoisePath)) {
    console.log(`Skipping ${variant} — high-noise model not found: ${highNoisePath}\n`);
    continue;
  }

  for (const t of tests) {
    console.log(`Loading ${variant}: ${modelPath}`);
    const model = createModel(modelPath, 'image') as ImageModel;
    await model.load({
      vaePath: `${v.dir}/${v.vae}`,
      t5xxlPath: `${v.dir}/${v.t5xxl}`,
      ...(highNoisePath ? { highNoiseDiffusionModelPath: highNoisePath } : {}),
      flashAttn: true,
      vaeDecodeOnly: true,
    });

    console.log(`▶ ${t.name} (${t.frames} frames, ${t.steps} steps)`);
    const start = Date.now();

    const frames = await model.generateVideo(PROMPT, {
      width: WIDTH, height: HEIGHT,
      negativePrompt: WAN_NEGATIVE_PROMPT,
      videoFrames: t.frames, steps: t.steps,
      cfgScale: t.cfgScale, flowShift: t.flowShift,
      seed: SEED,
      ...(t.highNoiseSteps != null ? {
        highNoiseSteps: t.highNoiseSteps,
        highNoiseCfgScale: t.highNoiseCfgScale,
        highNoiseSampleMethod: 'euler',
      } : {}),
    });

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const avgKB = (frames.reduce((s, f) => s + f.length, 0) / frames.length / 1024).toFixed(0);

    console.log(`  ${frames.length} frames in ${elapsed}s (avg ${avgKB}KB/frame)`);
    saveFrames(frames, t.name);
    results.push({ name: t.name, frames: frames.length, steps: t.steps, elapsed: `${elapsed}s`, avgFrameKB: `${avgKB}KB` });

    await model.unload();
    console.log();
  }
}

console.log('='.repeat(70));
console.log('SUMMARY');
console.log('='.repeat(70));
console.log(`${'Test'.padEnd(35)} ${'Frames'.padEnd(8)} ${'Steps'.padEnd(7)} ${'Time'.padEnd(10)} Avg Size`);
console.log('-'.repeat(70));
for (const r of results) {
  console.log(`${r.name.padEnd(35)} ${String(r.frames).padEnd(8)} ${String(r.steps).padEnd(7)} ${r.elapsed.padEnd(10)} ${r.avgFrameKB}`);
}
console.log();
