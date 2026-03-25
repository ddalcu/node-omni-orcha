/**
 * Image-to-Video test — generates videos from init images using WAN 2.2 models.
 * Tests both TI2V-5B (text+image→video) and I2V-A14B (image→video with MoE).
 *
 * Usage: node scripts/i2v-test.ts
 */

import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import * as path from 'node:path';
import { createModel } from '../src/index.ts';
import { saveVideoFrames } from '../test/test-output-helper.ts';
import type { ImageModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || `${process.env['HOME'] || process.env['USERPROFILE']}/.orcha/workspace/.models`;
const FIXTURES = path.resolve(import.meta.dirname!, '..', 'test', 'fixtures');

// Generate a simple test image using the FLUX model if no meme images exist
async function getTestImage(name: string): Promise<Buffer | null> {
  // Check for meme images first
  const memePath = path.join(FIXTURES, 'memes', name);
  if (existsSync(memePath)) return readFile(memePath);

  // Check for any PNG in test-output we can reuse (from previous FLUX generations)
  const testOutput = path.resolve(import.meta.dirname!, '..', 'test-output');
  if (existsSync(testOutput)) {
    const { readdirSync } = await import('node:fs');
    const pngs = readdirSync(testOutput).filter(f => f.startsWith('image_') && f.endsWith('.png'));
    if (pngs.length > 0) return readFile(path.join(testOutput, pngs[0]!));
  }

  return null;
}

let passed = 0, failed = 0, skipped = 0;

async function run(name: string, fn: () => Promise<void>) {
  process.stdout.write(`\n[${'='.repeat(60)}]\n${name}\n[${'='.repeat(60)}]\n`);
  const start = Date.now();
  try {
    await fn();
    console.log(`PASSED (${((Date.now() - start) / 1000).toFixed(1)}s)\n`);
    passed++;
  } catch (err) {
    console.error(`FAILED (${((Date.now() - start) / 1000).toFixed(1)}s): ${(err as Error).message}\n`);
    failed++;
  }
}

// ─── TI2V-5B: Text + Image → Video ───

const ti2vModel = path.join(MODELS_DIR, 'wan22-5b', 'Wan2.2-TI2V-5B-Q4_K_M.gguf');
const ti2vVae = path.join(MODELS_DIR, 'wan22-5b', 'Wan2.2_VAE.safetensors');
const ti2vT5 = path.join(MODELS_DIR, 'wan22-5b', 'umt5-xxl-encoder-Q8_0.gguf');

if (existsSync(ti2vModel) && existsSync(ti2vVae) && existsSync(ti2vT5)) {
  const initImage = await getTestImage('meme-1.png');

  if (initImage) {
    await run('TI2V-5B — animate init image', async () => {
      const model = createModel(ti2vModel, 'image') as ImageModel;
      await model.load({ t5xxlPath: ti2vT5, vaePath: ti2vVae, flashAttn: true, vaeDecodeOnly: true });

      const params = {
        width: 832, height: 480,
        videoFrames: 9, steps: 20,
        cfgScale: 3.5, flowShift: 3.0, seed: 42,
        initImage,
      };

      console.log('  Generating 9 frames from init image...');
      const frames = await model.generateVideo(
        'cinematic motion, camera slowly panning right, dramatic lighting',
        params,
      );

      console.log(`  ${frames.length} frames, avg ${(frames.reduce((s, f) => s + f.length, 0) / frames.length / 1024).toFixed(0)}KB/frame`);
      const outDir = saveVideoFrames('TI2V-5B-Q4_K_M', params, frames);
      console.log(`  Output: ${outDir}`);

      await model.unload();
    });
  } else {
    console.log('\nSKIP: TI2V-5B — no init image available (generate one with FLUX first or add test/fixtures/memes/)\n');
    skipped++;
  }
} else {
  console.log('\nSKIP: TI2V-5B — model files not found\n');
  skipped++;
}

// ─── I2V-A14B: Image → Video (MoE dual-expert) ───

const i2vDir = path.join(MODELS_DIR, 'wan22-i2v');
const i2vLowNoise = path.join(i2vDir, 'Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf');
const i2vHighNoise = path.join(i2vDir, 'Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf');
const i2vVae = path.join(i2vDir, 'Wan2.1_VAE.safetensors');

if (existsSync(i2vLowNoise) && existsSync(i2vHighNoise) && existsSync(i2vVae) && existsSync(ti2vT5)) {
  const initImage = await getTestImage('meme-1.png');

  if (initImage) {
    await run('I2V-A14B — MoE image-to-video', async () => {
      const model = createModel(i2vLowNoise, 'image') as ImageModel;
      await model.load({
        highNoiseDiffusionModelPath: i2vHighNoise,
        t5xxlPath: ti2vT5,
        vaePath: i2vVae,
        flashAttn: true,
        vaeDecodeOnly: false, // I2V needs full VAE to encode init image
      });

      const params = {
        width: 832, height: 480,
        videoFrames: 9, steps: 10,
        cfgScale: 3.5, flowShift: 3.0, seed: 42,
        initImage,
        highNoiseSteps: 8,
        highNoiseCfgScale: 3.5,
        highNoiseSampleMethod: 'euler' as const,
      };

      console.log('  Generating 9 frames from init image (MoE)...');
      const frames = await model.generateVideo(
        'cinematic motion, dramatic camera movement',
        params,
      );

      console.log(`  ${frames.length} frames, avg ${(frames.reduce((s, f) => s + f.length, 0) / frames.length / 1024).toFixed(0)}KB/frame`);
      const outDir = saveVideoFrames('I2V-A14B-Q2_K', params, frames);
      console.log(`  Output: ${outDir}`);

      await model.unload();
    });
  } else {
    console.log('\nSKIP: I2V-A14B — no init image available\n');
    skipped++;
  }
} else {
  console.log('\nSKIP: I2V-A14B — model files not found\n');
  skipped++;
}

// ─── Summary ───
console.log('\n' + '='.repeat(60));
console.log(`RESULTS: ${passed} passed, ${failed} failed, ${skipped} skipped`);
console.log('='.repeat(60) + '\n');

if (failed > 0) process.exit(1);
