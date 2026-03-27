/**
 * Crash test ROUND 6 — Video generation, malformed tool_calls,
 * load/unload interleaving, abort state
 */
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { loadModel, createModel, detectGpu } from '../src/index.ts';
import { loadBinding } from '../src/binding-loader.ts';
import type { LlmModel, ImageModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');

const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
const FLUX_DIR = path.join(MODELS_DIR, 'flux2-klein');
const FLUX_MODEL = path.join(FLUX_DIR, 'flux-2-klein-4b-Q4_K_M.gguf');
const FLUX_LLM = path.join(FLUX_DIR, 'Qwen3-4B-Q4_K_M.gguf');
const FLUX_VAE = path.join(FLUX_DIR, 'flux2-vae.safetensors');
const WAN_DIR = path.join(MODELS_DIR, 'wan22-5b');
const WAN_MODEL = path.join(WAN_DIR, 'Wan2.2-TI2V-5B-Q4_K_M.gguf');
const WAN_VAE = path.join(WAN_DIR, 'Wan2.2_VAE.safetensors');
const UMT5_PATH = path.join(WAN_DIR, 'umt5-xxl-encoder-Q8_0.gguf');

let passed = 0;
let failed = 0;

async function test(name: string, fn: () => Promise<void>, timeoutMs = 300000) {
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

// ─── Video generation with WAN 2.2 ───

async function testVideoGeneration() {
  console.log('\n=== Video Generation (WAN 2.2) ===');

  const hasWan = existsSync(WAN_MODEL) && existsSync(WAN_VAE) && existsSync(UMT5_PATH);
  if (!hasWan) {
    console.log('  SKIP: No WAN 2.2 model files');
    return;
  }

  let model: ImageModel;

  await test('load WAN 2.2 video model', async () => {
    model = createModel(WAN_MODEL, 'image');
    await model.load({
      t5xxlPath: UMT5_PATH,
      vaePath: WAN_VAE,
      flashAttn: true,
      vaeDecodeOnly: true,
      keepVaeOnCpu: true,
    });
    if (!model.loaded) throw new Error('Should be loaded');
  });

  await test('generate 5 video frames (minimal)', async () => {
    const frames = await model.generateVideo('a woman walking on a beach', {
      width: 480, height: 272, videoFrames: 5, steps: 5,
      cfgScale: 6.0, flowShift: 3.0, seed: 42,
    });
    if (!Array.isArray(frames)) throw new Error('Should return array');
    if (frames.length === 0) throw new Error('Should have frames');
    for (let i = 0; i < frames.length; i++) {
      if (!Buffer.isBuffer(frames[i])) throw new Error(`Frame ${i} not a Buffer`);
      if (frames[i][0] !== 0x89) throw new Error(`Frame ${i} not valid PNG`);
    }
    console.log(`[${frames.length} frames] `);
  });

  // generateVideo with invalid params
  await test('generateVideo with 0 frames', async () => {
    try {
      await model.generateVideo('test', { videoFrames: 0, steps: 1 });
    } catch (e: any) {
      // Should error, not crash
    }
  });

  await test('generateVideo with 0 steps', async () => {
    try {
      await model.generateVideo('test', { steps: 0 });
    } catch (e: any) {
      // Should error, not crash
    }
  });

  await test('generateVideo with 0x0 dimensions', async () => {
    try {
      await model.generateVideo('test', { width: 0, height: 0, steps: 1 });
    } catch (e: any) {
      // Should error, not crash
    }
  });

  await test('generateVideo with empty prompt', async () => {
    try {
      await model.generateVideo('', { videoFrames: 5, steps: 2, width: 480, height: 272 });
    } catch (e: any) {
      // May work or error
    }
  });

  // Concurrent video generation (serialized by mutex)
  await test('concurrent video generate (2x, serialized)', async () => {
    const [frames1, frames2] = await Promise.all([
      model.generateVideo('a sunset', { videoFrames: 5, steps: 2, seed: 1, width: 480, height: 272 }),
      model.generateVideo('a sunrise', { videoFrames: 5, steps: 2, seed: 2, width: 480, height: 272 }),
    ]);
    if (frames1.length === 0 || frames2.length === 0) throw new Error('Empty frames');
  });

  await model.unload();
}

// ─── Malformed tool_calls in messages ───

async function testMalformedToolCalls() {
  console.log('\n=== Malformed Tool Calls in Messages ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  const binding = loadBinding();
  let ctx: any;

  await test('create LLM context', async () => {
    ctx = await (binding['createLlmContext'] as Function)(LLM_MODEL, {
      contextSize: 2048, gpuLayers: -1, flashAttn: true,
      embeddings: false, batchSize: 4096, cacheTypeK: 'f16', cacheTypeV: 'f16', chatTemplate: '',
    });
  });

  // tool_calls with missing name field
  await test('message with tool_calls missing name', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: '', tool_calls: [{ args: '{}' }] },
      ], { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // tool_calls with missing args field
  await test('message with tool_calls missing args', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: '', tool_calls: [{ name: 'test' }] },
      ], { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // tool_calls that's not an array of objects
  await test('message with tool_calls as array of strings', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: '', tool_calls: ['not', 'objects'] },
      ], { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // tool_calls with null elements
  await test('message with null tool_calls elements', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: '', tool_calls: [null] },
      ], { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // tool message with numeric content
  await test('tool message with numeric content', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
        { role: 'tool', content: 42, tool_call_id: 'id1', name: 'test' },
      ], { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // tools array with missing fields
  await test('tools with missing description', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
      ], {
        temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0,
        tools: [{ name: 'test', parameters: '{}' }],
        toolChoice: 'auto',
      });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // tools with null element
  await test('tools array with null element', async () => {
    try {
      await (ctx.complete as Function)([
        { role: 'user', content: 'hi' },
      ], {
        temperature: 0, maxTokens: 5, stopSequences: [],
        tools: [null],
        toolChoice: 'auto',
      });
    } catch (e: any) {
      // JS error, not crash
    }
  });

  // Verify context still works after all the bad inputs
  await test('valid complete after malformed tool_calls', async () => {
    const r = await (ctx.complete as Function)(
      [{ role: 'user', content: 'Say OK' }],
      { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 }
    );
    if (!r.content && r.content !== '') throw new Error('No content');
  });

  await (ctx.unload as Function)();
}

// ─── Abort flag state management ───

async function testAbortState() {
  console.log('\n=== Abort Flag State ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  let llm: LlmModel;

  await test('load LLM', async () => {
    llm = await loadModel(LLM_MODEL, {
      type: 'llm', contextSize: 2048, gpuLayers: -1,
    }) as LlmModel;
  });

  // Abort during stream, then verify complete works (abort flag reset)
  await test('abort during stream then complete', async () => {
    const controller = new AbortController();
    let chunks = 0;
    for await (const chunk of llm.stream(
      [{ role: 'user', content: 'Write forever' }],
      { maxTokens: 200, temperature: 0, thinkingBudget: 0, signal: controller.signal }
    )) {
      chunks++;
      if (chunks >= 3) controller.abort();
      if (chunk.done) break;
    }
    // Now complete should work (abort flag should be reset)
    const r = await llm.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content after abort');
  });

  // Multiple aborts in sequence
  await test('3 aborted streams then complete', async () => {
    for (let i = 0; i < 3; i++) {
      const ctrl = new AbortController();
      for await (const chunk of llm.stream(
        [{ role: 'user', content: 'Count forever' }],
        { maxTokens: 100, temperature: 0, thinkingBudget: 0, signal: ctrl.signal }
      )) {
        ctrl.abort();
        break;
      }
    }
    const r = await llm.complete(
      [{ role: 'user', content: 'Alive?' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('Dead after aborts');
  });

  await llm.unload();
}

// ─── Image generate + generateVideo mixed on same context ───

async function testImageVideoMixed() {
  console.log('\n=== Image + Video Mixed Operations ===');

  const hasFlux = existsSync(FLUX_MODEL) && existsSync(FLUX_LLM) && existsSync(FLUX_VAE);
  if (!hasFlux) { console.log('  SKIP'); return; }

  let model: ImageModel;

  await test('load FLUX (image-only model)', async () => {
    model = createModel(FLUX_MODEL, 'image');
    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
  });

  // FLUX is an image model, not a video model. Calling generateVideo should error, not crash.
  await test('generateVideo on image-only model (FLUX)', async () => {
    try {
      await model.generateVideo('a cat walking', {
        videoFrames: 5, steps: 2, width: 256, height: 256,
      });
      // If it somehow works, that's fine too
    } catch (e: any) {
      // Error is expected, crash is not
    }
  });

  // Image generate should still work after bad generateVideo call
  await test('generate after failed generateVideo', async () => {
    const png = await model.generate('a cat', {
      width: 256, height: 256, steps: 2, cfgScale: 1.0,
    });
    if (png[0] !== 0x89) throw new Error('Invalid PNG');
  });

  await model.unload();
}

// ─── Load during unload ───

async function testLoadDuringUnload() {
  console.log('\n=== Load During Unload ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  await test('unload then immediately load (sequential)', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    await model.load({ contextSize: 2048, gpuLayers: -1 });
    await model.unload();
    // loading flag should be false after unload, allowing reload
    await model.load({ contextSize: 2048, gpuLayers: -1 });
    const r = await model.complete(
      [{ role: 'user', content: 'OK?' }],
      { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content after reload');
    await model.unload();
  });

  await test('rapid load/unload cycles (5x)', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    for (let i = 0; i < 5; i++) {
      await model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false });
      await model.unload();
    }
    // Final load + use
    await model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false });
    const r = await model.complete(
      [{ role: 'user', content: 'Survived?' }],
      { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content after cycles');
    await model.unload();
  });
}

// ─── Run ───

async function main() {
  const gpu = detectGpu();
  console.log('========================================');
  console.log('  CRASH TEST ROUND 6');
  console.log(`  GPU: ${gpu.name} (${gpu.backend})`);
  console.log('========================================');

  await testMalformedToolCalls();
  await testAbortState();
  await testImageVideoMixed();
  await testLoadDuringUnload();
  await testVideoGeneration();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed`);
  console.log('========================================');

  if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
