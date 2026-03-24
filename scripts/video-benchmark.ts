import { createModel } from '../src/index.ts';
import type { ImageModel } from '../src/types.ts';
import { existsSync } from 'node:fs';
import { saveVideoFrames } from '../test/test-output-helper.ts';

const MODELS_DIR = process.env.MODELS_DIR ?? `${process.env.HOME}/.orcha/workspace/.models`;

const WAN_NEGATIVE_PROMPT = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走';

const VARIANTS = {
  '5b': {
    dir: `${MODELS_DIR}/wan22-5b`,
    model: 'Wan2.2-TI2V-5B-Q8_0.gguf',
    vae: 'Wan2.2_VAE.safetensors',
    t5xxl: 'umt5-xxl-encoder-Q8_0.gguf',
  },
  'turbo': {
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
  { name: 'quick-preview-turbo-81f-5s', variant: 'turbo', frames: 9, steps: 1, cfgScale: 3.5, flowShift: 3.0, highNoiseSteps: 2, highNoiseCfgScale: 3.5 },
];

const PROMPT = 'angle shot of a red car outside';
const WIDTH = 832;
const HEIGHT = 480;
const SEED = 42;

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

    const params = {
      width: WIDTH,
      height: HEIGHT,
      videoFrames: t.frames,
      steps: t.steps,
      cfgScale: t.cfgScale,
      flowShift: t.flowShift,
      seed: SEED,
      ...(t.highNoiseSteps != null ? {
        highNoiseSteps: t.highNoiseSteps,
        highNoiseCfgScale: t.highNoiseCfgScale,
        highNoiseSampleMethod: 'euler',
      } : {}),
    };

    console.log(`▶ ${t.name} (${t.frames} frames, ${t.steps} steps)`);
    const start = Date.now();

    const frames = await model.generateVideo(PROMPT, params);

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const avgKB = (frames.reduce((s, f) => s + f.length, 0) / frames.length / 1024).toFixed(0);

    console.log(`  ${frames.length} frames in ${elapsed}s (avg ${avgKB}KB/frame)`);

    const modelSlug = v.model.replace('.gguf', '');
    const outDir = saveVideoFrames(modelSlug, params, frames);
    console.log(`  Output: ${outDir}`);
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
