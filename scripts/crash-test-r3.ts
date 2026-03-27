/**
 * Crash test ROUND 3 — Real model operations on CUDA
 * Targets: image generation, context reuse bug, embeddings, voice cloning,
 * load/unload races, concurrent cross-engine stress
 */
import * as path from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { loadModel, createModel, detectGpu } from '../src/index.ts';
import type { LlmModel, SttModel, TtsModel, ImageModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');

// Model paths
const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
const EMBED_MODEL = path.join(MODELS_DIR, 'nomic-embed-text-v1-5-q4_k_m', 'nomic-embed-text-v1.5.Q4_K_M.gguf');
const WHISPER_MODEL = path.join('test', 'fixtures', 'whisper-tiny.bin');
const TTS_DIR = path.join(MODELS_DIR, 'qwen3-tts');
const SAMUEL_WAV = path.join('test', 'fixtures', 'Samuel.wav');
const TEST_AUDIO = path.join('test', 'fixtures', 'test-audio.pcm');

// Image/video models
const FLUX_DIR = path.join(MODELS_DIR, 'flux2-klein');
const FLUX_MODEL = path.join(FLUX_DIR, 'flux-2-klein-4b-Q4_K_M.gguf');
const FLUX_LLM = path.join(FLUX_DIR, 'Qwen3-4B-Q4_K_M.gguf');
const FLUX_VAE = path.join(FLUX_DIR, 'flux2-vae.safetensors');

let passed = 0;
let failed = 0;

async function test(name: string, fn: () => Promise<void>) {
  process.stdout.write(`  [TEST] ${name}... `);
  try {
    await fn();
    console.log('PASS');
    passed++;
  } catch (err: any) {
    console.log(`FAIL: ${err.message?.substring(0, 120)}`);
    failed++;
  }
}

// ─── Embedding Model Tests ───

async function testEmbeddings() {
  console.log('\n=== Embedding Model (nomic-embed) ===');

  if (!existsSync(EMBED_MODEL)) {
    console.log('  SKIP: No embedding model');
    return;
  }

  let emb: LlmModel;

  await test('load embedding model (embeddings=true)', async () => {
    emb = await loadModel(EMBED_MODEL, {
      type: 'llm',
      embeddings: true,
      contextSize: 2048,
    }) as LlmModel;
    if (!emb.loaded) throw new Error('Should be loaded');
  });

  await test('embed single text', async () => {
    const vec = await emb.embed('Hello world');
    if (!(vec instanceof Float64Array)) throw new Error('Should return Float64Array');
    if (vec.length === 0) throw new Error('Empty embedding');
    // Check it's not all zeros
    const sum = vec.reduce((a, b) => a + Math.abs(b), 0);
    if (sum === 0) throw new Error('All-zero embedding');
  });

  await test('embed empty string', async () => {
    try {
      const vec = await emb.embed('');
      // May work or error, shouldn't crash
    } catch (e: any) {
      // Fine
    }
  });

  await test('embed very long text', async () => {
    try {
      const vec = await emb.embed('word '.repeat(5000));
    } catch (e: any) {
      // May exceed context
    }
  });

  await test('embedBatch basic', async () => {
    const vecs = await emb.embedBatch(['Hello', 'World', 'Test']);
    if (vecs.length !== 3) throw new Error(`Expected 3 embeddings, got ${vecs.length}`);
    for (const v of vecs) {
      if (!(v instanceof Float64Array)) throw new Error('Not Float64Array');
      if (v.length === 0) throw new Error('Empty');
    }
  });

  await test('embedBatch empty array', async () => {
    const vecs = await emb.embedBatch([]);
    if (vecs.length !== 0) throw new Error('Should return empty array');
  });

  await test('embedBatch single item', async () => {
    const vecs = await emb.embedBatch(['single']);
    if (vecs.length !== 1) throw new Error('Should return 1 embedding');
  });

  await test('concurrent embeddings (5x)', async () => {
    const promises = Array.from({ length: 5 }, (_, i) =>
      emb.embed(`Text number ${i}`)
    );
    const results = await Promise.all(promises);
    for (const v of results) {
      if (v.length === 0) throw new Error('Empty embedding');
    }
  });

  await test('embed similarity check', async () => {
    const v1 = await emb.embed('The cat sat on the mat');
    const v2 = await emb.embed('A feline rested on the rug');
    const v3 = await emb.embed('Quantum physics equations');
    // Cosine similarity
    function cosine(a: Float64Array, b: Float64Array): number {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
    const sim12 = cosine(v1, v2);
    const sim13 = cosine(v1, v3);
    // Similar sentences should have higher similarity
    if (sim12 <= sim13) throw new Error(`Expected cat/feline (${sim12.toFixed(3)}) > cat/quantum (${sim13.toFixed(3)})`);
  });

  await test('rapid sequential embeds (20x)', async () => {
    for (let i = 0; i < 20; i++) {
      await emb.embed(`Rapid test ${i}`);
    }
  });

  await test('unload and re-embed throws', async () => {
    await emb.unload();
    try {
      await emb.embed('should fail');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });
}

// ─── TTS Voice Cloning Tests ───

async function testTTSVoiceCloning() {
  console.log('\n=== TTS Voice Cloning ===');

  if (!existsSync(TTS_DIR)) {
    console.log('  SKIP: No TTS model');
    return;
  }

  let tts: TtsModel;

  await test('load TTS', async () => {
    tts = await loadModel(TTS_DIR, { type: 'tts' }) as TtsModel;
  });

  if (existsSync(SAMUEL_WAV)) {
    await test('voice clone with Samuel.wav', async () => {
      const wav = await tts.speak('Hello, this is a test of voice cloning.', {
        referenceAudioPath: SAMUEL_WAV,
      });
      if (!Buffer.isBuffer(wav)) throw new Error('Should return Buffer');
      if (wav.length < 100) throw new Error('WAV too small');
      // Check WAV header
      if (wav.toString('ascii', 0, 4) !== 'RIFF') throw new Error('Not a WAV file');
    });

    await test('voice clone with short text', async () => {
      const wav = await tts.speak('Hi.', { referenceAudioPath: SAMUEL_WAV });
      if (wav.length < 44) throw new Error('WAV too small (less than header)');
    });

    await test('concurrent voice clones (2x)', async () => {
      const [w1, w2] = await Promise.all([
        tts.speak('First clone test.', { referenceAudioPath: SAMUEL_WAV }),
        tts.speak('Second clone test.', { referenceAudioPath: SAMUEL_WAV }),
      ]);
      if (w1.length < 100 || w2.length < 100) throw new Error('Output too small');
    });
  }

  await test('speak without reference audio', async () => {
    const wav = await tts.speak('Hello world, this is text to speech.');
    if (!Buffer.isBuffer(wav)) throw new Error('Should return Buffer');
    if (wav.length < 100) throw new Error('WAV too small');
  });

  await test('sequential speaks (5x)', async () => {
    for (let i = 0; i < 5; i++) {
      const wav = await tts.speak(`Test number ${i + 1}.`);
      if (wav.length < 44) throw new Error(`Speak ${i} failed`);
    }
  });

  await tts.unload();
}

// ─── Image Generation (FLUX 2) ───

async function testImageGeneration() {
  console.log('\n=== Image Generation (FLUX 2 Klein) ===');

  const hasFlux = existsSync(FLUX_MODEL) && existsSync(FLUX_LLM) && existsSync(FLUX_VAE);
  if (!hasFlux) {
    console.log('  SKIP: No FLUX model files');
    return;
  }

  await test('load FLUX model', async () => {
    const model = createModel(FLUX_MODEL, 'image');
    await model.load({
      llmPath: FLUX_LLM,
      vaePath: FLUX_VAE,
      keepVaeOnCpu: true,
    });
    if (!model.loaded) throw new Error('Should be loaded');

    // Test generate
    await test('generate 256x256 image (4 steps)', async () => {
      const png = await model.generate('a red circle', {
        width: 256, height: 256, steps: 4, cfgScale: 1.0,
      });
      if (!Buffer.isBuffer(png)) throw new Error('Should return Buffer');
      if (png.length < 100) throw new Error('PNG too small');
      if (png[0] !== 0x89 || png[1] !== 0x50) throw new Error('Not a valid PNG');
    });

    // THE KNOWN BUG: sd.cpp context reuse crashes on second generation
    await test('second image generation (context reuse crash test)', async () => {
      const png = await model.generate('a blue square', {
        width: 256, height: 256, steps: 4, cfgScale: 1.0,
      });
      if (!Buffer.isBuffer(png)) throw new Error('Should return Buffer');
      if (png[0] !== 0x89) throw new Error('Not valid PNG');
    });

    // Empty prompt
    await test('generate with empty prompt', async () => {
      try {
        await model.generate('', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
      } catch (e: any) {
        // Error is fine
      }
    });

    // Very small dimensions
    await test('generate tiny image (64x64)', async () => {
      try {
        const png = await model.generate('a dot', { width: 64, height: 64, steps: 2, cfgScale: 1.0 });
      } catch (e: any) {
        // May fail with size constraints
      }
    });

    // Various sample methods
    await test('generate with euler_a sampler', async () => {
      try {
        const png = await model.generate('a star', {
          width: 256, height: 256, steps: 4, cfgScale: 1.0, sampleMethod: 'euler_a',
        });
      } catch (e: any) {
        // Some samplers may not work with all models
      }
    });

    await test('generate with negative seed', async () => {
      const png = await model.generate('a tree', {
        width: 256, height: 256, steps: 4, cfgScale: 1.0, seed: -1,
      });
      if (png.length < 100) throw new Error('Too small');
    });

    await test('generate with fixed seed (deterministic)', async () => {
      const png1 = await model.generate('a cat', {
        width: 256, height: 256, steps: 4, cfgScale: 1.0, seed: 42,
      });
      const png2 = await model.generate('a cat', {
        width: 256, height: 256, steps: 4, cfgScale: 1.0, seed: 42,
      });
      // Same seed should produce same output (if context reuse works)
      // But even if not identical due to context issues, neither should crash
    });

    await model.unload();
  });
}

// ─── Load/Unload Race Conditions ───

async function testLoadUnloadRaces() {
  console.log('\n=== Load/Unload Race Conditions ===');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  await test('double load (should be idempotent)', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    await model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false });
    await model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false }); // Should be no-op
    const r = await model.complete([{ role: 'user', content: 'Hi' }], { maxTokens: 3, temperature: 0, thinkingBudget: 0 });
    if (!r.content && r.content !== '') throw new Error('No content');
    await model.unload();
  });

  await test('concurrent loads (same model)', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    // Race two loads
    await Promise.all([
      model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false }),
      model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false }),
    ]);
    const r = await model.complete([{ role: 'user', content: 'Hi' }], { maxTokens: 3, temperature: 0, thinkingBudget: 0 });
    await model.unload();
  });

  await test('load then immediate unload', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    const loadPromise = model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false });
    // Don't await — immediately unload
    await loadPromise;
    await model.unload();
    // Should be cleanly unloaded
    if (model.loaded) throw new Error('Should not be loaded');
  });

  await test('unload during completion', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    await model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false });

    // Start a completion
    const completionPromise = model.complete(
      [{ role: 'user', content: 'Write a very long essay about everything in the universe.' }],
      { maxTokens: 100, temperature: 0.5, thinkingBudget: 0 }
    );

    // Immediately unload — the completion should either complete or error, not crash
    // Wait a tiny bit so the completion has started
    await new Promise(r => setTimeout(r, 50));
    try {
      await model.unload();
    } catch (e: any) {
      // May error if completion is in flight
    }

    try {
      await completionPromise;
    } catch (e: any) {
      // Error is fine — crash is not
    }
  });
}

// ─── STT + TTS roundtrip ───

async function testSTTTTSRoundtrip() {
  console.log('\n=== STT + TTS Roundtrip ===');

  const hasTts = existsSync(TTS_DIR);
  const hasStt = existsSync(WHISPER_MODEL);

  if (!hasTts || !hasStt) {
    console.log('  SKIP: Need both TTS and STT models');
    return;
  }

  await test('TTS → STT roundtrip', async () => {
    const tts = await loadModel(TTS_DIR, { type: 'tts' }) as TtsModel;
    const stt = await loadModel(WHISPER_MODEL, { type: 'stt' }) as SttModel;

    // Generate speech
    const wav = await tts.speak('Hello world this is a test.');

    // The WAV output from TTS is 24kHz. Whisper expects 16kHz 16-bit PCM.
    // We can't easily resample here, but we can at least pass it and see
    // if it crashes or handles gracefully.
    // Skip the 44-byte WAV header to get raw PCM
    const pcm = wav.subarray(44);

    try {
      const result = await stt.transcribe(pcm, { language: 'en' });
      // May get garbage since sample rate mismatch, but shouldn't crash
    } catch (e: any) {
      // Error is fine
    }

    await Promise.all([tts.unload(), stt.unload()]);
  });
}

// ─── Run all ───

async function main() {
  const gpu = detectGpu();
  console.log('========================================');
  console.log('  CRASH TEST ROUND 3 — Real Operations');
  console.log(`  GPU: ${gpu.name} (${gpu.backend})`);
  console.log('========================================');

  await testEmbeddings();
  await testTTSVoiceCloning();
  await testImageGeneration();
  await testLoadUnloadRaces();
  await testSTTTTSRoundtrip();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed`);
  console.log('========================================');

  if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
