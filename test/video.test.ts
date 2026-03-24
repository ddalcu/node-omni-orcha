import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import { createModel } from '../src/index.ts';
import type { ImageModel } from '../src/types.ts';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { saveVideoFrames } from './test-output-helper.ts';

const FIXTURES = join(import.meta.dirname, 'fixtures');

const WAN_MODEL = join(FIXTURES, 'Wan2.2-TI2V-5B-Q8_0.gguf');
const WAN_TURBO = join(FIXTURES, 'Wan2.2-TI2V-5B-Turbo-Q8_0.gguf');
const WAN_VAE   = join(FIXTURES, 'wan2.2_vae.safetensors');
const UMT5_PATH = join(FIXTURES, 'umt5-xxl-encoder-Q3_K_S.gguf');

const PROMPT = 'a woman walking on a beach';
const MIN_FRAME_KB = 200;

function assertValidFrames(frames: Buffer[], label: string) {
  assert.ok(Array.isArray(frames), `${label}: should return array`);
  assert.ok(frames.length > 0, `${label}: should have frames`);

  for (let i = 0; i < frames.length; i++) {
    assert.ok(Buffer.isBuffer(frames[i]), `${label} frame ${i}: should be Buffer`);
    assert.equal(frames[i][0], 0x89, `${label} frame ${i}: PNG magic byte 0`);
    assert.equal(frames[i][1], 0x50, `${label} frame ${i}: PNG magic byte 1`);
    const sizeKB = frames[i].length / 1024;
    assert.ok(sizeKB > MIN_FRAME_KB, `${label} frame ${i}: ${sizeKB.toFixed(0)}KB too small (likely gray/noise)`);
  }
}

// --- WAN 2.2 Base Model ---

const hasBase = existsSync(WAN_MODEL) && existsSync(WAN_VAE) && existsSync(UMT5_PATH);

describe('WAN 2.2 TI2V 5B (base)', { skip: !hasBase && 'WAN 2.2 base model files not found' }, () => {
  let model: ImageModel;

  before(async () => {
    model = createModel(WAN_MODEL, 'image') as ImageModel;
    await model.load({ t5xxlPath: UMT5_PATH, vaePath: WAN_VAE, flashAttn: true, vaeDecodeOnly: true });
  });

  after(async () => {
    await model?.unload();
  });

  it('generates 5 video frames', async () => {
    const params = { width: 832, height: 480, videoFrames: 5, steps: 10, cfgScale: 6.0, flowShift: 3.0, seed: 42 };
    const start = Date.now();
    const frames = await model.generateVideo(PROMPT, params);
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    assertValidFrames(frames, 'base');
    console.log(`    Base: ${frames.length} frames in ${elapsed}s`);
    console.log(`    Sizes: ${frames.map(f => `${(f.length / 1024).toFixed(0)}KB`).join(', ')}`);

    const outDir = saveVideoFrames('wan22-TI2V-5B-Q8_0', params, frames);
    console.log(`    Saved: ${outDir}`);
  });
});

// --- WAN 2.2 Turbo Model ---

const hasTurbo = existsSync(WAN_TURBO) && existsSync(WAN_VAE) && existsSync(UMT5_PATH);

describe('WAN 2.2 TI2V 5B Turbo', { skip: !hasTurbo && 'WAN 2.2 Turbo model files not found' }, () => {
  let model: ImageModel;

  before(async () => {
    model = createModel(WAN_TURBO, 'image') as ImageModel;
    await model.load({ t5xxlPath: UMT5_PATH, vaePath: WAN_VAE, flashAttn: true, vaeDecodeOnly: true });
  });

  after(async () => {
    await model?.unload();
  });

  it('generates 9 video frames at 8 steps', async () => {
    const params = { width: 832, height: 480, videoFrames: 9, steps: 8, cfgScale: 1.0, flowShift: 3.0, seed: 42 };
    const start = Date.now();
    const frames = await model.generateVideo(PROMPT, params);
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    assertValidFrames(frames, 'turbo');
    console.log(`    Turbo: ${frames.length} frames in ${elapsed}s`);
    console.log(`    Sizes: ${frames.map(f => `${(f.length / 1024).toFixed(0)}KB`).join(', ')}`);

    const outDir = saveVideoFrames('wan22-TI2V-5B-Turbo-Q8_0', params, frames);
    console.log(`    Saved: ${outDir}`);
  });
});
