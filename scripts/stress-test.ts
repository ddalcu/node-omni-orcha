/**
 * Stress test for node-omni-orcha
 * Tests crash vectors: invalid inputs, edge cases, concurrency, double-free, use-after-unload
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { loadModel, createModel, readGGUFMetadata, detectGpu } from '../src/index.ts';
import { loadBinding } from '../src/binding-loader.ts';
import type { LlmModel, TtsModel, ImageModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || '', '.orcha', 'workspace', '.models');
const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
const EMBED_MODEL = path.join(MODELS_DIR, 'nomic-embed-text-v1-5-q4_k_m', 'nomic-embed-text-v1.5.Q4_K_M.gguf');
const TTS_DIR = path.join(MODELS_DIR, 'qwen3-tts');
const FLUX_MODEL = path.join(MODELS_DIR, 'flux2-klein', 'flux-2-klein-4b-Q4_K_M.gguf');
const FLUX_LLM = path.join(MODELS_DIR, 'flux2-klein', 'Qwen3-4B-Q4_K_M.gguf');
const FLUX_VAE = path.join(MODELS_DIR, 'flux2-klein', 'flux2-vae.safetensors');

const hasLlm = existsSync(LLM_MODEL);
const hasEmbed = existsSync(EMBED_MODEL);
const hasTts = existsSync(path.join(TTS_DIR, 'qwen3-tts-0.6b-f16.gguf'));
const hasFlux = existsSync(FLUX_MODEL) && existsSync(FLUX_LLM) && existsSync(FLUX_VAE);

let passed = 0;
let failed = 0;
let skipped = 0;

function log(emoji: string, msg: string) {
  console.log(`  ${emoji} ${msg}`);
}

// ═══════════════════════════════════════════════════════════════
// 1. Pure TS-layer validation tests (no model needed)
// ═══════════════════════════════════════════════════════════════

describe('STRESS: TypeScript layer validation', () => {
  it('loadModel rejects with invalid type', async () => {
    await assert.rejects(
      () => loadModel('/fake/model.gguf', { type: 'invalid' as any }),
      /Unknown model type/,
    );
    log('OK', 'loadModel with invalid type throws');
  });

  it('createModel rejects with invalid type', () => {
    assert.throws(
      () => createModel('/fake/model.gguf', 'invalid' as any),
      /Unknown model type/,
    );
    log('OK', 'createModel with invalid type throws');
  });

  it('createModel with empty string path does not crash', () => {
    // Should create the model object (path validation happens at load time)
    const model = createModel('', 'llm');
    assert.equal(model.type, 'llm');
    assert.equal(model.loaded, false);
    log('OK', 'createModel with empty path creates unloaded model');
  });

  it('loadModel with non-existent file rejects gracefully', async () => {
    await assert.rejects(
      () => loadModel('/nonexistent/path/model.gguf', { type: 'llm', contextSize: 2048 }),
    );
    log('OK', 'loadModel with non-existent file rejects');
  });

  it('readGGUFMetadata with non-existent file returns null', async () => {
    const result = await readGGUFMetadata('/nonexistent/model.gguf');
    assert.equal(result, null);
    log('OK', 'readGGUFMetadata with non-existent file returns null');
  });

  it('readGGUFMetadata with non-GGUF file returns null', async () => {
    const result = await readGGUFMetadata(path.join(import.meta.dirname, '..', 'package.json'));
    assert.equal(result, null);
    log('OK', 'readGGUFMetadata with non-GGUF file returns null');
  });

  it('readGGUFMetadata with empty string returns null', async () => {
    const result = await readGGUFMetadata('');
    assert.equal(result, null);
    log('OK', 'readGGUFMetadata with empty string returns null');
  });

  it('detectGpu returns valid info', () => {
    const gpu = detectGpu();
    assert.ok(['metal', 'cuda', 'cpu'].includes(gpu.backend));
    log('OK', `detectGpu returns backend: ${gpu.backend}`);
  });

  it('loadBinding returns a valid object', () => {
    const binding = loadBinding();
    assert.ok(binding);
    assert.equal(typeof binding['createLlmContext'], 'function');
    assert.equal(typeof binding['createTtsContext'], 'function');
    assert.equal(typeof binding['createImageContext'], 'function');
    log('OK', 'loadBinding returns valid binding with all context creators');
  });

  it('loadBinding is cached (returns same instance)', () => {
    const a = loadBinding();
    const b = loadBinding();
    assert.strictEqual(a, b);
    log('OK', 'loadBinding returns cached instance');
  });
});

// ═══════════════════════════════════════════════════════════════
// 2. Native binding edge cases (call C++ directly)
// ═══════════════════════════════════════════════════════════════

describe('STRESS: Native binding direct calls', () => {
  const binding = loadBinding();

  it('createLlmContext with non-existent model rejects', async () => {
    await assert.rejects(
      () => (binding['createLlmContext'] as Function)('/nonexistent/model.gguf', {
        contextSize: 2048, gpuLayers: 0, flashAttn: false, embeddings: false,
        batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16', chatTemplate: '',
      }),
    );
    log('OK', 'createLlmContext with non-existent model rejects');
  });

  it('createTtsContext with non-existent path rejects', async () => {
    await assert.rejects(
      () => (binding['createTtsContext'] as Function)('/nonexistent/tts/', {}),
    );
    log('OK', 'createTtsContext with non-existent path rejects');
  });

  it('createImageContext with non-existent model rejects', async () => {
    await assert.rejects(
      () => (binding['createImageContext'] as Function)('/nonexistent/model.safetensors', {
        clipLPath: '', t5xxlPath: '', llmPath: '', vaePath: '',
        highNoiseDiffusionModelPath: '', threads: 1, keepVaeOnCpu: false,
        offloadToCpu: false, flashAttn: false, vaeDecodeOnly: true,
      }),
    );
    log('OK', 'createImageContext with non-existent model rejects');
  });
});

// ═══════════════════════════════════════════════════════════════
// 3. LLM model crash scenarios
// ═══════════════════════════════════════════════════════════════

describe('STRESS: LLM model edge cases', { skip: !hasLlm ? 'No LLM model' : undefined }, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048 });
  });

  after(async () => {
    await model?.unload();
  });

  it('complete with empty messages array', async () => {
    // This should either return an error or an empty result, NOT crash
    try {
      const result = await model.complete([], { temperature: 0, maxTokens: 10 });
      // If it succeeds, it should have some kind of result
      log('OK', `complete([]) returned content: "${result.content.slice(0, 50)}"`);
    } catch (err: any) {
      log('OK', `complete([]) threw (expected): ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with very short maxTokens (1)', async () => {
    const result = await model.complete(
      [{ role: 'user', content: 'Hello' }],
      { temperature: 0, maxTokens: 1 },
    );
    assert.ok(result.usage.outputTokens <= 2);
    log('OK', `complete with maxTokens=1 returned ${result.usage.outputTokens} tokens`);
  });

  it('complete with maxTokens=0', async () => {
    try {
      const result = await model.complete(
        [{ role: 'user', content: 'Hello' }],
        { temperature: 0, maxTokens: 0 },
      );
      log('OK', `complete with maxTokens=0 returned ${result.usage.outputTokens} tokens`);
    } catch (err: any) {
      log('OK', `complete with maxTokens=0 threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with negative maxTokens', async () => {
    try {
      const result = await model.complete(
        [{ role: 'user', content: 'Hello' }],
        { temperature: 0, maxTokens: -1 },
      );
      log('OK', `complete with maxTokens=-1 returned ${result.usage.outputTokens} tokens`);
    } catch (err: any) {
      log('OK', `complete with maxTokens=-1 threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with temperature=0', async () => {
    const result = await model.complete(
      [{ role: 'user', content: 'Say hello' }],
      { temperature: 0, maxTokens: 10 },
    );
    assert.ok(result.content.length >= 0);
    log('OK', 'complete with temperature=0 works');
  });

  it('complete with negative temperature', async () => {
    try {
      const result = await model.complete(
        [{ role: 'user', content: 'Hello' }],
        { temperature: -1, maxTokens: 10 },
      );
      log('OK', `complete with temperature=-1 returned: "${result.content.slice(0, 50)}"`);
    } catch (err: any) {
      log('OK', `complete with temperature=-1 threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with very high temperature', async () => {
    const result = await model.complete(
      [{ role: 'user', content: 'Hello' }],
      { temperature: 100, maxTokens: 10 },
    );
    assert.ok(result.content.length >= 0);
    log('OK', 'complete with temperature=100 works');
  });

  it('complete with empty string content', async () => {
    try {
      const result = await model.complete(
        [{ role: 'user', content: '' }],
        { temperature: 0, maxTokens: 10 },
      );
      log('OK', `complete with empty content returned: "${result.content.slice(0, 50)}"`);
    } catch (err: any) {
      log('OK', `complete with empty content threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with very long input (prompt overflow)', async () => {
    const longMsg = 'a '.repeat(5000); // Should exceed context
    try {
      const result = await model.complete(
        [{ role: 'user', content: longMsg }],
        { temperature: 0, maxTokens: 10 },
      );
      log('OK', `complete with long input returned ${result.usage.inputTokens} input tokens`);
    } catch (err: any) {
      log('OK', `complete with long input threw (expected): ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with thinkingBudget=0 (disable reasoning)', async () => {
    const result = await model.complete(
      [{ role: 'user', content: 'Say hello' }],
      { temperature: 0, maxTokens: 32, thinkingBudget: 0 },
    );
    assert.ok(result.content.length > 0 || result.reasoning === undefined);
    log('OK', 'complete with thinkingBudget=0 works');
  });

  it('complete with thinkingBudget=1 (very small budget)', async () => {
    try {
      const result = await model.complete(
        [{ role: 'user', content: 'What is 2+2?' }],
        { temperature: 0, maxTokens: 64, thinkingBudget: 1 },
      );
      log('OK', `complete with thinkingBudget=1 returned: content="${result.content.slice(0, 30)}" reasoning="${(result.reasoning ?? '').slice(0, 30)}"`);
    } catch (err: any) {
      log('OK', `complete with thinkingBudget=1 threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('complete with empty tools array', async () => {
    const result = await model.complete(
      [{ role: 'user', content: 'Hello' }],
      { temperature: 0, maxTokens: 16, tools: [] },
    );
    assert.ok(result.content.length >= 0);
    log('OK', 'complete with empty tools array works');
  });

  it('complete with tool but no tool choice', async () => {
    const result = await model.complete(
      [{ role: 'user', content: 'Hello' }],
      {
        temperature: 0, maxTokens: 32,
        tools: [{ name: 'test', description: 'A test tool', parameters: { type: 'object', properties: {} } }],
      },
    );
    assert.ok(result.content.length >= 0 || result.toolCalls);
    log('OK', 'complete with tool (no toolChoice) works');
  });

  it('stream with empty messages array', async () => {
    try {
      const chunks: any[] = [];
      for await (const chunk of model.stream([], { temperature: 0, maxTokens: 10 })) {
        chunks.push(chunk);
        if (chunk.done) break;
      }
      log('OK', `stream([]) returned ${chunks.length} chunks`);
    } catch (err: any) {
      log('OK', `stream([]) threw (expected): ${err.message?.slice(0, 80)}`);
    }
  });

  it('stream with abort signal (immediate abort)', async () => {
    const controller = new AbortController();
    controller.abort(); // abort immediately
    try {
      const chunks: any[] = [];
      for await (const chunk of model.stream(
        [{ role: 'user', content: 'Count from 1 to 100' }],
        { temperature: 0, maxTokens: 100, signal: controller.signal },
      )) {
        chunks.push(chunk);
        if (chunk.done) break;
      }
      log('OK', `stream with immediate abort returned ${chunks.length} chunks`);
    } catch (err: any) {
      log('OK', `stream with immediate abort threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('stream with abort signal (abort after first chunk)', async () => {
    const controller = new AbortController();
    try {
      let count = 0;
      for await (const chunk of model.stream(
        [{ role: 'user', content: 'Count from 1 to 100' }],
        { temperature: 0, maxTokens: 100, signal: controller.signal },
      )) {
        count++;
        if (count === 1) controller.abort();
        if (chunk.done) break;
      }
      log('OK', `stream with delayed abort returned ${count} chunks`);
    } catch (err: any) {
      log('OK', `stream with delayed abort threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('multiple sequential completions (KV cache reuse)', async () => {
    for (let i = 0; i < 5; i++) {
      const result = await model.complete(
        [{ role: 'user', content: `Test ${i}` }],
        { temperature: 0, maxTokens: 5 },
      );
      assert.ok(result.content.length >= 0);
    }
    log('OK', 'multiple sequential completions work');
  });

  it('load() when already loaded is idempotent', async () => {
    await model.load({ contextSize: 2048 });
    assert.equal(model.loaded, true);
    log('OK', 'load() on already loaded model is idempotent');
  });
});

// ═══════════════════════════════════════════════════════════════
// 4. Concurrency stress tests
// ═══════════════════════════════════════════════════════════════

describe('STRESS: LLM concurrency', { skip: !hasLlm ? 'No LLM model' : undefined }, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048 });
  });

  after(async () => {
    await model?.unload();
  });

  it('two concurrent completions on same context are serialized (no crash)', async () => {
    // With the TS-level serialize mutex, both calls should succeed sequentially
    const [r1, r2] = await Promise.allSettled([
      model.complete([{ role: 'user', content: 'Say A' }], { temperature: 0, maxTokens: 5 }),
      model.complete([{ role: 'user', content: 'Say B' }], { temperature: 0, maxTokens: 5 }),
    ]);
    assert.equal(r1.status, 'fulfilled', `r1 should succeed, got: ${r1.status}`);
    assert.equal(r2.status, 'fulfilled', `r2 should succeed, got: ${r2.status}`);
    log('OK', 'concurrent completions serialized safely (both succeeded)');
  });

  it('completion + stream concurrently are serialized (no crash)', async () => {
    const streamPromise = (async () => {
      const chunks: any[] = [];
      for await (const chunk of model.stream(
        [{ role: 'user', content: 'Count to 3' }],
        { temperature: 0, maxTokens: 20 },
      )) {
        chunks.push(chunk);
        if (chunk.done) break;
      }
      return chunks;
    })();

    const [complete, stream] = await Promise.allSettled([
      model.complete([{ role: 'user', content: 'Say hi' }], { temperature: 0, maxTokens: 5 }),
      streamPromise,
    ]);
    assert.equal(complete.status, 'fulfilled', `complete should succeed`);
    assert.equal(stream.status, 'fulfilled', `stream should succeed`);
    log('OK', 'complete+stream concurrent serialized safely (both succeeded)');
  });
});

// ═══════════════════════════════════════════════════════════════
// 5. Unload lifecycle tests
// ═══════════════════════════════════════════════════════════════

describe('STRESS: Unload lifecycle', { skip: !hasLlm ? 'No LLM model' : undefined }, () => {
  it('unload then complete throws (use-after-unload)', async () => {
    const model = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048 });
    await model.unload();

    await assert.rejects(
      () => model.complete([{ role: 'user', content: 'Hello' }], { maxTokens: 5 }),
      /not loaded/,
    );
    log('OK', 'use-after-unload throws');
  });

  it('double unload does not crash', async () => {
    const model = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048 });
    await model.unload();
    await model.unload(); // should be safe
    log('OK', 'double unload is safe');
  });

  it('unload then embed throws', async () => {
    const model = await loadModel(EMBED_MODEL, { type: 'llm', contextSize: 512, embeddings: true, gpuLayers: 0 });
    await model.unload();

    await assert.rejects(
      () => model.embed('test'),
      /not loaded/,
    );
    log('OK', 'embed after unload throws');
  });

  it('load → unload → reload → complete works', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    await model.load({ contextSize: 2048 });
    assert.equal(model.loaded, true);

    await model.unload();
    assert.equal(model.loaded, false);

    await model.load({ contextSize: 2048 });
    assert.equal(model.loaded, true);

    const result = await model.complete(
      [{ role: 'user', content: 'Say hi' }],
      { temperature: 0, maxTokens: 5 },
    );
    assert.ok(result.content.length >= 0);
    log('OK', 'load → unload → reload → complete cycle works');

    await model.unload();
  });
});

// ═══════════════════════════════════════════════════════════════
// 6. Embedding edge cases
// ═══════════════════════════════════════════════════════════════

describe('STRESS: Embedding edge cases', { skip: !hasEmbed ? 'No embedding model' : undefined }, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(EMBED_MODEL, { type: 'llm', contextSize: 512, embeddings: true, gpuLayers: 0 });
  });

  after(async () => {
    await model?.unload();
  });

  it('embed with empty string', async () => {
    try {
      const emb = await model.embed('');
      log('OK', `embed('') returned array of length ${emb.length}`);
    } catch (err: any) {
      log('OK', `embed('') threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('embed with single character', async () => {
    const emb = await model.embed('a');
    assert.ok(emb.length > 0);
    log('OK', `embed('a') returned ${emb.length}-dim vector`);
  });

  it('embed with very long text', async () => {
    const longText = 'word '.repeat(2000);
    try {
      const emb = await model.embed(longText);
      log('OK', `embed(long) returned ${emb.length}-dim vector`);
    } catch (err: any) {
      log('OK', `embed(long) threw (expected): ${err.message?.slice(0, 80)}`);
    }
  });

  it('embedBatch with empty array', async () => {
    try {
      const results = await model.embedBatch([]);
      log('OK', `embedBatch([]) returned ${results.length} results`);
    } catch (err: any) {
      log('OK', `embedBatch([]) threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('embedBatch with single item', async () => {
    const results = await model.embedBatch(['hello']);
    assert.equal(results.length, 1);
    assert.ok(results[0]!.length > 0);
    log('OK', `embedBatch(['hello']) returned 1 result with ${results[0]!.length} dims`);
  });

  it('embedBatch with multiple items', async () => {
    const results = await model.embedBatch(['hello', 'world', 'test']);
    assert.equal(results.length, 3);
    for (const r of results) {
      assert.ok(r.length > 0);
    }
    log('OK', `embedBatch(3 items) returned 3 results`);
  });

  it('embed multiple sequential calls', async () => {
    for (let i = 0; i < 10; i++) {
      const emb = await model.embed(`test ${i}`);
      assert.ok(emb.length > 0);
    }
    log('OK', 'embed 10 sequential calls work');
  });
});

// ═══════════════════════════════════════════════════════════════
// 7. TTS edge cases
// ═══════════════════════════════════════════════════════════════

describe('STRESS: TTS edge cases', { skip: !hasTts ? 'No TTS model' : undefined }, () => {
  let model: TtsModel;

  before(async () => {
    model = createModel(TTS_DIR, 'tts');
    await model.load();
  });

  after(async () => {
    await model?.unload();
  });

  it('speak with empty string rejects', async () => {
    await assert.rejects(
      () => model.speak(''),
      /must not be empty/,
    );
    log('OK', 'speak("") rejects with validation error');
  });

  it('speak with whitespace-only string rejects', async () => {
    await assert.rejects(
      () => model.speak('   \n\t  '),
      /must not be empty/,
    );
    log('OK', 'speak("   ") rejects with validation error');
  });

  it('speak with normal text', { timeout: 60_000 }, async () => {
    const wav = await model.speak('Hello world, this is a test of the text to speech system.');
    assert.ok(Buffer.isBuffer(wav));
    assert.ok(wav.length > 1000);
    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');
    log('OK', `speak(normal text) returned ${wav.length} bytes`);
  });

  it('speak with short maxDurationSeconds', { timeout: 30_000 }, async () => {
    const wav = await model.speak(
      'This is a test to ensure the maximum duration cap works correctly.',
      { maxDurationSeconds: 5 },
    );
    assert.ok(Buffer.isBuffer(wav));
    // 5 seconds at 24kHz mono 16-bit = 240000 bytes of PCM + 44 byte header
    // Should be well under 300KB
    assert.ok(wav.length < 500_000, `WAV should be under 500KB for 5s cap, got ${wav.length}`);
    log('OK', `speak(maxDuration=5s) returned ${(wav.length / 1024).toFixed(0)}KB`);
  });

  it('speak with non-existent reference audio path', { timeout: 30_000 }, async () => {
    try {
      const wav = await model.speak('This is a test sentence for voice cloning.', { referenceAudioPath: '/nonexistent/voice.wav' });
      log('WARN', `speak with bad ref path returned ${wav.length} bytes (unexpected success)`);
    } catch (err: any) {
      log('OK', `speak with bad ref path threw (expected): ${err.message?.slice(0, 80)}`);
    }
  });

  it('double unload TTS is safe', async () => {
    const tts = createModel(TTS_DIR, 'tts');
    await tts.load();
    await tts.unload();
    await tts.unload();
    log('OK', 'TTS double unload is safe');
  });

  it('speak after unload throws', async () => {
    const tts = createModel(TTS_DIR, 'tts');
    await tts.load();
    await tts.unload();
    await assert.rejects(
      () => tts.speak('Test'),
      /not loaded/,
    );
    log('OK', 'speak after unload throws');
  });
});

// ═══════════════════════════════════════════════════════════════
// 8. Image (FLUX) edge cases
// ═══════════════════════════════════════════════════════════════

describe('STRESS: Image (FLUX) edge cases', { skip: !hasFlux ? 'No FLUX model' : undefined }, () => {
  let model: ImageModel;

  before(async () => {
    model = createModel(FLUX_MODEL, 'image');
    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
  });

  after(async () => {
    await model?.unload();
  });

  it('loads and reports as loaded', () => {
    assert.equal(model.type, 'image');
    assert.equal(model.loaded, true);
    log('OK', 'FLUX image model loaded');
  });

  it('generate with empty prompt', async () => {
    try {
      const png = await model.generate('', { width: 256, height: 256, steps: 1, cfgScale: 1.0 });
      assert.ok(Buffer.isBuffer(png));
      log('OK', `generate('') returned ${(png.length / 1024).toFixed(0)}KB PNG`);
    } catch (err: any) {
      log('OK', `generate('') threw: ${err.message?.slice(0, 80)}`);
    }
  });

  it('generate a small image', { timeout: 120_000 }, async () => {
    const png = await model.generate('a red circle', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
    assert.ok(Buffer.isBuffer(png));
    assert.ok(png.length > 100);
    assert.equal(png[0], 0x89, 'PNG magic byte');
    log('OK', `generate(small) returned ${(png.length / 1024).toFixed(0)}KB PNG`);
  });

  it('generate after unload throws', async () => {
    const img = createModel(FLUX_MODEL, 'image');
    await img.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
    await img.unload();
    await assert.rejects(
      () => img.generate('test', { width: 256, height: 256, steps: 1 }),
      /not loaded/,
    );
    log('OK', 'generate after unload throws');
  });

  it('double unload Image is safe', async () => {
    const img = createModel(FLUX_MODEL, 'image');
    await img.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
    await img.unload();
    await img.unload();
    log('OK', 'Image double unload is safe');
  });
});

// ═══════════════════════════════════════════════════════════════
// 9. GGUF reader edge cases
// ═══════════════════════════════════════════════════════════════

describe('STRESS: GGUF reader edge cases', () => {
  it('readGGUFMetadata with directory path', async () => {
    const result = await readGGUFMetadata('/tmp');
    assert.equal(result, null);
    log('OK', 'readGGUFMetadata with directory returns null');
  });

  it('readGGUFMetadata with binary file', async () => {
    const result = await readGGUFMetadata(path.join(import.meta.dirname, '..', 'build', 'Release', 'omni.node'));
    assert.equal(result, null);
    log('OK', 'readGGUFMetadata with binary file returns null');
  });

  it('readGGUFMetadata with actual GGUF', { skip: !hasLlm ? 'No LLM model' : undefined }, async () => {
    const result = await readGGUFMetadata(LLM_MODEL);
    assert.ok(result);
    assert.ok(result!.contextLength > 0);
    assert.ok(result!.blockCount > 0);
    assert.ok(result!.embeddingLength > 0);
    log('OK', `readGGUFMetadata: arch=${result!.architecture}, ctx=${result!.contextLength}, layers=${result!.blockCount}`);
  });
});

// ═══════════════════════════════════════════════════════════════
// 10. Rapid load/unload cycling
// ═══════════════════════════════════════════════════════════════

describe('STRESS: Rapid load/unload cycling', { skip: !hasEmbed ? 'No embedding model' : undefined }, () => {
  it('load and unload embedding model 5 times rapidly', async () => {
    for (let i = 0; i < 5; i++) {
      const model = createModel(EMBED_MODEL, 'llm');
      await model.load({ contextSize: 256, embeddings: true, gpuLayers: 0 });
      const emb = await model.embed('test');
      assert.ok(emb.length > 0);
      await model.unload();
    }
    log('OK', 'rapid load/unload cycling (5x) works');
  });
});

// Print summary at the end
process.on('exit', () => {
  console.log('\n═══════════════════════════════════');
  console.log('  STRESS TEST COMPLETE');
  console.log('═══════════════════════════════════');
});
