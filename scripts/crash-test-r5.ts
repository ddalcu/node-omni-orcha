/**
 * Crash test ROUND 5 — Concurrent load race, invalid image dims,
 * native binding type misuse, video generation
 */
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { loadModel, createModel, detectGpu } from '../src/index.ts';
import { loadBinding } from '../src/binding-loader.ts';
import type { LlmModel, ImageModel, SttModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');

const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
const WHISPER_MODEL = path.join('test', 'fixtures', 'whisper-tiny.bin');
const FLUX_DIR = path.join(MODELS_DIR, 'flux2-klein');
const FLUX_MODEL = path.join(FLUX_DIR, 'flux-2-klein-4b-Q4_K_M.gguf');
const FLUX_LLM = path.join(FLUX_DIR, 'Qwen3-4B-Q4_K_M.gguf');
const FLUX_VAE = path.join(FLUX_DIR, 'flux2-vae.safetensors');

let passed = 0;
let failed = 0;

async function test(name: string, fn: () => Promise<void>, timeoutMs = 120000) {
  process.stdout.write(`  [TEST] ${name}... `);
  try {
    await Promise.race([
      fn(),
      new Promise<never>((_, rej) => setTimeout(() => rej(new Error('TIMEOUT')), timeoutMs)),
    ]);
    console.log('PASS');
    passed++;
  } catch (err: any) {
    console.log(`FAIL: ${err.message?.substring(0, 120)}`);
    failed++;
  }
}

// ─── Concurrent load race ───

async function testConcurrentLoadRace() {
  console.log('\n=== Concurrent Load Race (VRAM leak prevention) ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  await test('5 concurrent loads — only one should create context', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    await Promise.all(Array.from({ length: 5 }, () =>
      model.load({ contextSize: 2048, gpuLayers: -1 })
    ));
    if (!model.loaded) throw new Error('Not loaded');
    const r = await model.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content');
    await model.unload();
  });

  if (existsSync(WHISPER_MODEL)) {
    await test('concurrent STT loads', async () => {
      const model = createModel(WHISPER_MODEL, 'stt');
      await Promise.all(Array.from({ length: 3 }, () => model.load()));
      if (!model.loaded) throw new Error('Not loaded');
      await model.unload();
    });
  }

  await test('load after failed load retries correctly', async () => {
    const model = createModel('/nonexistent.gguf', 'llm');
    try { await model.load({ contextSize: 2048 }); } catch (e: any) { /* expected */ }
    // loading flag should be reset, allowing retry
    try { await model.load({ contextSize: 2048 }); } catch (e: any) { /* expected again */ }
    // Should not be stuck in 'loading' state
    if (model.loaded) throw new Error('Should not be loaded');
  });
}

// ─── Image invalid dimensions ───

async function testImageInvalidDimensions() {
  console.log('\n=== Image Invalid Dimensions ===');

  const hasFlux = existsSync(FLUX_MODEL) && existsSync(FLUX_LLM) && existsSync(FLUX_VAE);
  if (!hasFlux) { console.log('  SKIP: No FLUX'); return; }

  let model: ImageModel;

  await test('load FLUX', async () => {
    model = createModel(FLUX_MODEL, 'image');
    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
  });

  await test('generate 0x0 image', async () => {
    try {
      await model.generate('test', { width: 0, height: 0, steps: 1, cfgScale: 1.0 });
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('generate negative dimensions', async () => {
    try {
      await model.generate('test', { width: -1, height: -1, steps: 1, cfgScale: 1.0 });
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('generate 1x1 image', async () => {
    try {
      await model.generate('test', { width: 1, height: 1, steps: 1, cfgScale: 1.0 });
    } catch (e: any) {
      // May fail
    }
  });

  await test('generate 0 steps', async () => {
    try {
      await model.generate('test', { width: 256, height: 256, steps: 0, cfgScale: 1.0 });
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('generate negative steps', async () => {
    try {
      await model.generate('test', { width: 256, height: 256, steps: -1, cfgScale: 1.0 });
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('generate with cfgScale = 0', async () => {
    try {
      await model.generate('test', { width: 256, height: 256, steps: 2, cfgScale: 0 });
    } catch (e: any) {
      // May fail
    }
  });

  await test('generate with NaN cfgScale', async () => {
    try {
      await model.generate('test', { width: 256, height: 256, steps: 2, cfgScale: NaN });
    } catch (e: any) {
      // Error is fine
    }
  });

  // Verify model still works after all the bad inputs
  await test('normal generate after invalid inputs', async () => {
    const png = await model.generate('recovery', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
    if (png[0] !== 0x89) throw new Error('Invalid PNG');
  });

  await model.unload();
}

// ─── Native binding ParseMessages crash vectors ───

async function testNativeParseMessages() {
  console.log('\n=== Native Binding ParseMessages Abuse ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  const binding = loadBinding();

  // First create a valid context to test with
  let ctx: any;
  await test('create valid LLM context', async () => {
    ctx = await (binding['createLlmContext'] as Function)(LLM_MODEL, {
      contextSize: 2048, gpuLayers: 0, flashAttn: false,
      embeddings: false, batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16', chatTemplate: '',
    });
  });

  // Pass array with non-object elements
  await test('complete with array of strings (not objects)', async () => {
    try {
      await (ctx.complete as Function)(['hello', 'world'], {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Pass array with null elements
  await test('complete with array of nulls', async () => {
    try {
      await (ctx.complete as Function)([null, null], {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Pass message with missing role
  await test('complete with message missing role', async () => {
    try {
      await (ctx.complete as Function)([{ content: 'hi' }], {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Pass message with missing content
  await test('complete with message missing content', async () => {
    try {
      await (ctx.complete as Function)([{ role: 'user' }], {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Pass non-array as messages
  await test('complete with object instead of array', async () => {
    try {
      await (ctx.complete as Function)({ role: 'user', content: 'hi' }, {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Pass number as messages
  await test('complete with number as messages', async () => {
    try {
      await (ctx.complete as Function)(42, {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Numeric role/content
  await test('complete with numeric role/content', async () => {
    try {
      await (ctx.complete as Function)([{ role: 123, content: 456 }], {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Valid message after bad ones — verify context isn't corrupted
  await test('valid complete after bad inputs', async () => {
    const r = await (ctx.complete as Function)(
      [{ role: 'user', content: 'Say OK' }],
      { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 }
    );
    if (!r.content && r.content !== '') throw new Error('No content');
  });

  // Stream with bad callback
  await test('stream with non-function callback', async () => {
    try {
      await (ctx.stream as Function)(
        [{ role: 'user', content: 'hi' }],
        { temperature: 0, maxTokens: 5, stopSequences: [] },
        'not a function'
      );
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('stream with no callback', async () => {
    try {
      await (ctx.stream as Function)(
        [{ role: 'user', content: 'hi' }],
        { temperature: 0, maxTokens: 5, stopSequences: [] }
      );
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // embed with non-string
  await test('embed with number', async () => {
    try {
      await (ctx.embed as Function)(42);
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // embedBatch with non-array
  await test('embedBatch with string', async () => {
    try {
      await (ctx.embedBatch as Function)('hello');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await (ctx.unload as Function)();
}

// ─── Run ───

async function main() {
  const gpu = detectGpu();
  console.log('========================================');
  console.log('  CRASH TEST ROUND 5');
  console.log(`  GPU: ${gpu.name} (${gpu.backend})`);
  console.log('========================================');

  await testConcurrentLoadRace();
  await testNativeParseMessages();
  await testImageInvalidDimensions();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed`);
  console.log('========================================');

  if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
