import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { loadModel, createModel } from '../src/index.ts';
import type { ImageModel } from '../src/types.ts';

// --- Basic SD model test (single file) ---
const SD_MODEL_PATH = new URL('./fixtures/sd_turbo.safetensors', import.meta.url).pathname;
const hasSDModel = existsSync(SD_MODEL_PATH);

describe('ImageModel (SD)', { skip: !hasSDModel ? 'No SD model at test/fixtures/sd_turbo.safetensors' : undefined }, () => {
  let model: ImageModel;

  before(async () => {
    model = createModel(SD_MODEL_PATH, 'image');
    await model.load();
  });

  after(async () => {
    await model?.unload();
  });

  it('loads and reports as loaded', () => {
    assert.equal(model.type, 'image');
    assert.equal(model.loaded, true);
  });

  it('generates a small image', async () => {
    const png = await model.generate('a red circle on white background', {
      width: 256,
      height: 256,
      steps: 4,
      cfgScale: 1.0,
    });

    assert.ok(Buffer.isBuffer(png), 'Should return a Buffer');
    assert.ok(png.length > 100, 'PNG should have content');
    // Verify PNG magic bytes
    assert.equal(png[0], 0x89, 'Should start with PNG magic');
    assert.equal(png[1], 0x50, 'Should start with PNG magic');
    assert.equal(png[2], 0x4E, 'Should start with PNG magic');
    assert.equal(png[3], 0x47, 'Should start with PNG magic');
  });
});

// --- FLUX model test (multi-file) ---
const FLUX_MODEL_PATH = new URL('./fixtures/flux-dev-Q8_0.gguf', import.meta.url).pathname;
const FLUX_CLIP_L = new URL('./fixtures/clip_l.safetensors', import.meta.url).pathname;
const FLUX_T5XXL = new URL('./fixtures/t5xxl-Q8_0.gguf', import.meta.url).pathname;
const FLUX_VAE = new URL('./fixtures/ae.safetensors', import.meta.url).pathname;

const hasFluxModel = existsSync(FLUX_MODEL_PATH) &&
                     existsSync(FLUX_CLIP_L) &&
                     existsSync(FLUX_T5XXL) &&
                     existsSync(FLUX_VAE);

describe('ImageModel (FLUX)', { skip: !hasFluxModel ? 'No FLUX model files in test/fixtures/' : undefined }, () => {
  let model: ImageModel;

  before(async () => {
    model = createModel(FLUX_MODEL_PATH, 'image');
    await model.load({
      clipLPath: FLUX_CLIP_L,
      t5xxlPath: FLUX_T5XXL,
      vaePath: FLUX_VAE,
      keepVaeOnCpu: true,
    });
  });

  after(async () => {
    await model?.unload();
  });

  it('loads FLUX model with all components', () => {
    assert.equal(model.type, 'image');
    assert.equal(model.loaded, true);
  });

  it('generates a FLUX image', async () => {
    const png = await model.generate('a beautiful sunset over mountains', {
      width: 512,
      height: 512,
      steps: 20,
      cfgScale: 1.0,
      sampleMethod: 'euler',
    });

    assert.ok(Buffer.isBuffer(png), 'Should return a Buffer');
    assert.ok(png.length > 1000, 'PNG should have substantial content');
    assert.equal(png[0], 0x89, 'Should be valid PNG');

    // Save for manual inspection
    const outPath = new URL('./fixtures/flux_output.png', import.meta.url).pathname;
    await writeFile(outPath, png);
  });
});

// --- createModel tests (no model files needed) ---
describe('createModel image', () => {
  it('creates unloaded image model', () => {
    const model = createModel('/fake/model.safetensors', 'image');
    assert.equal(model.type, 'image');
    assert.equal(model.loaded, false);
  });
});
