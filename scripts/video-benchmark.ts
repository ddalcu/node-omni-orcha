/**
 * Video generation benchmark — generates videos with WAN 2.2 models.
 * Outputs PNG frames + MP4 (via ffmpeg) to test-output/.
 *
 * Usage:
 *   node scripts/video-benchmark.ts              # run all tests
 *   node scripts/video-benchmark.ts 5b           # filter by variant
 *   node scripts/video-benchmark.ts hq           # filter by test name
 */

import { createModel } from '../src/index.ts';
import type { ImageModel } from '../src/types.ts';
import { existsSync } from 'node:fs';
import { saveVideoFrames } from '../test/test-output-helper.ts';

const MODELS_DIR = process.env.MODELS_DIR ?? `${process.env.HOME || process.env.USERPROFILE}/.orcha/workspace/.models`;

const VARIANTS = {
  '5b': {
    dir: `${MODELS_DIR}/wan22-5b`,
    model: 'Wan2.2-TI2V-5B-Q4_K_M.gguf',
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
  prompt: string;
  width: number;
  height: number;
  frames: number;
  steps: number;
  cfgScale: number;
  flowShift: number;
  seed: number;
  highNoiseSteps?: number;
  highNoiseCfgScale?: number;
}

const TESTS: TestCase[] = [
  // Quick preview — low res, few frames, few steps
  {
    name: 'quick-5b',
    variant: '5b',
    prompt: 'a red sports car driving fast on a mountain road at sunset, cinematic',
    width: 480, height: 288,
    frames: 5, steps: 6,
    cfgScale: 3.5, flowShift: 3.0, seed: 42,
  },
  // High quality — more frames, more steps, higher resolution
  {
    name: 'hq-5b',
    variant: '5b',
    prompt: 'a red sports car driving fast on a mountain road at sunset, cinematic',
    width: 832, height: 480,
    frames: 9, steps: 20,
    cfgScale: 3.5, flowShift: 3.0, seed: 42,
  },
];

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

console.log(`\nVideo Benchmark — ${testsToRun.length} test(s)\n`);

const results: { name: string; prompt: string; res: string; frames: number; steps: number; elapsed: string; avgFrameKB: string }[] = [];

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
      width: t.width,
      height: t.height,
      videoFrames: t.frames,
      steps: t.steps,
      cfgScale: t.cfgScale,
      flowShift: t.flowShift,
      seed: t.seed,
      ...(t.highNoiseSteps != null ? {
        highNoiseSteps: t.highNoiseSteps,
        highNoiseCfgScale: t.highNoiseCfgScale,
        highNoiseSampleMethod: 'euler',
      } : {}),
    };

    console.log(`\n  ${t.name}: "${t.prompt.slice(0, 60)}..."`);
    console.log(`  ${t.width}x${t.height}, ${t.frames} frames, ${t.steps} steps, cfg=${t.cfgScale}, seed=${t.seed}`);
    const start = Date.now();

    const frames = await model.generateVideo(t.prompt, params);

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const avgKB = (frames.reduce((s, f) => s + f.length, 0) / frames.length / 1024).toFixed(0);

    console.log(`  ${frames.length} frames in ${elapsed}s (avg ${avgKB}KB/frame)`);

    const modelSlug = v.model.replace('.gguf', '');
    const outDir = saveVideoFrames(modelSlug, params, frames);
    console.log(`  Frames: ${outDir}`);
    results.push({
      name: t.name,
      prompt: t.prompt.slice(0, 40),
      res: `${t.width}x${t.height}`,
      frames: frames.length,
      steps: t.steps,
      elapsed: `${elapsed}s`,
      avgFrameKB: `${avgKB}KB`,
    });

    await model.unload();
    console.log();
  }
}

console.log('='.repeat(90));
console.log('SUMMARY');
console.log('='.repeat(90));
console.log(`${'Test'.padEnd(15)} ${'Resolution'.padEnd(12)} ${'Frames'.padEnd(8)} ${'Steps'.padEnd(7)} ${'Time'.padEnd(10)} ${'Avg Size'.padEnd(10)} Prompt`);
console.log('-'.repeat(90));
for (const r of results) {
  console.log(`${r.name.padEnd(15)} ${r.res.padEnd(12)} ${String(r.frames).padEnd(8)} ${String(r.steps).padEnd(7)} ${r.elapsed.padEnd(10)} ${r.avgFrameKB.padEnd(10)} ${r.prompt}`);
}
console.log();
