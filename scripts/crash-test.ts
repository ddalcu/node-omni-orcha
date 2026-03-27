/**
 * Comprehensive crash test for node-omni-orcha.
 * Tries to crash every engine through edge cases, bad inputs, concurrent access, etc.
 */
import * as path from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { loadModel, createModel, readGGUFMetadata, detectGpu } from '../src/index.ts';
import type { LlmModel, SttModel, TtsModel, ImageModel, ChatMessage } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');

const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
const WHISPER_MODEL = path.join('test', 'fixtures', 'whisper-tiny.bin');
const TTS_DIR = path.join(MODELS_DIR, 'qwen3-tts');
const TEST_AUDIO = path.join('test', 'fixtures', 'test-audio.pcm');

let passed = 0;
let failed = 0;
let crashed = 0;

async function test(name: string, fn: () => Promise<void>) {
  process.stdout.write(`  [TEST] ${name}... `);
  try {
    await fn();
    console.log('PASS');
    passed++;
  } catch (err: any) {
    // Check if this was a controlled error (thrown JS error) vs a crash
    if (err?.message?.includes('SEGFAULT') || err?.message?.includes('SIGSEGV') || err?.code === 'ERR_WORKER_OUT_OF_MEMORY') {
      console.log(`CRASH: ${err.message}`);
      crashed++;
    } else {
      console.log(`FAIL: ${err.message}`);
      failed++;
    }
  }
}

async function section(name: string) {
  console.log(`\n=== ${name} ===`);
}

// ─── GGUF Reader edge cases ───

async function testGGUFReader() {
  await section('GGUF Reader');

  await test('readGGUFMetadata with non-existent file', async () => {
    const result = await readGGUFMetadata('/nonexistent/path.gguf');
    if (result !== null) throw new Error('Expected null for non-existent file');
  });

  await test('readGGUFMetadata with empty string path', async () => {
    const result = await readGGUFMetadata('');
    if (result !== null) throw new Error('Expected null for empty path');
  });

  await test('readGGUFMetadata with directory path', async () => {
    const result = await readGGUFMetadata(MODELS_DIR);
    if (result !== null) throw new Error('Expected null for directory');
  });

  await test('readGGUFMetadata with non-GGUF file', async () => {
    const result = await readGGUFMetadata('package.json');
    if (result !== null) throw new Error('Expected null for non-GGUF file');
  });

  if (existsSync(LLM_MODEL)) {
    await test('readGGUFMetadata with valid model', async () => {
      const result = await readGGUFMetadata(LLM_MODEL);
      if (!result) throw new Error('Expected metadata for valid model');
      if (result.contextLength <= 0) throw new Error('Expected positive context length');
    });
  }
}

// ─── GPU Detection ───

async function testGpuDetection() {
  await section('GPU Detection');

  await test('detectGpu returns valid backend', async () => {
    const gpu = detectGpu();
    if (!['metal', 'cuda', 'cpu'].includes(gpu.backend)) {
      throw new Error(`Invalid backend: ${gpu.backend}`);
    }
  });

  await test('detectGpu is idempotent', async () => {
    const g1 = detectGpu();
    const g2 = detectGpu();
    if (g1.backend !== g2.backend) throw new Error('Inconsistent GPU detection');
  });
}

// ─── LLM Crash Tests ───

async function testLLM() {
  await section('LLM Engine');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model available');
    return;
  }

  // Invalid model paths
  await test('loadModel with non-existent file throws (not crash)', async () => {
    try {
      await loadModel('/nonexistent.gguf', { type: 'llm', contextSize: 2048 });
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
      // Good - it threw an error instead of crashing
    }
  });

  await test('createModel with empty path then load throws', async () => {
    const model = createModel('', 'llm');
    try {
      await model.load({ contextSize: 2048 });
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Use before load
  await test('complete before load throws', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    try {
      await model.complete([{ role: 'user', content: 'Hi' }]);
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('embed before load throws', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    try {
      await model.embed('test');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Load, use, unload, test various crash scenarios
  let llm: LlmModel;

  await test('load LLM model', async () => {
    llm = await loadModel(LLM_MODEL, {
      type: 'llm',
      contextSize: 2048,
      gpuLayers: 0,  // CPU only for speed
    }) as LlmModel;
    if (!llm.loaded) throw new Error('Should be loaded');
  });

  await test('empty messages array', async () => {
    try {
      await llm.complete([]);
      // Some models may handle this, some may throw
    } catch (e: any) {
      // Error is fine, crash is not
    }
  });

  await test('message with empty content', async () => {
    const result = await llm.complete([
      { role: 'user', content: '' }
    ], { maxTokens: 5, temperature: 0 });
    // Should complete without crashing
  });

  await test('very long prompt', async () => {
    const longText = 'a'.repeat(10000);
    try {
      await llm.complete([
        { role: 'user', content: longText }
      ], { maxTokens: 5, temperature: 0 });
    } catch (e: any) {
      // May exceed context, but shouldn't crash
    }
  });

  await test('unicode and special characters', async () => {
    await llm.complete([
      { role: 'user', content: '你好世界 🌍 \n\t\0 \x00 null bytes?' }
    ], { maxTokens: 5, temperature: 0 });
  });

  await test('maxTokens = 0', async () => {
    try {
      const result = await llm.complete([
        { role: 'user', content: 'Hi' }
      ], { maxTokens: 0, temperature: 0 });
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('maxTokens = 1', async () => {
    const result = await llm.complete([
      { role: 'user', content: 'Hi' }
    ], { maxTokens: 1, temperature: 0 });
  });

  await test('negative temperature', async () => {
    try {
      await llm.complete([
        { role: 'user', content: 'Hi' }
      ], { maxTokens: 5, temperature: -1 });
    } catch (e: any) {
      // Error or fallback is fine
    }
  });

  await test('extremely high temperature', async () => {
    await llm.complete([
      { role: 'user', content: 'Hi' }
    ], { maxTokens: 5, temperature: 100 });
  });

  await test('thinkingBudget = 0 (disabled)', async () => {
    const result = await llm.complete([
      { role: 'user', content: 'Hello' }
    ], { maxTokens: 32, temperature: 0, thinkingBudget: 0 });
    if (!result.content && result.content !== '') throw new Error('Should have content or empty string');
  });

  await test('thinkingBudget = 1 (minimal)', async () => {
    try {
      await llm.complete([
        { role: 'user', content: 'Hello' }
      ], { maxTokens: 32, temperature: 0, thinkingBudget: 1 });
    } catch (e: any) {
      // Budget too small may error
    }
  });

  await test('multiple system messages (may fail with template error)', async () => {
    try {
      await llm.complete([
        { role: 'system', content: 'You are a pirate.' },
        { role: 'system', content: 'Actually you are a robot.' },
        { role: 'user', content: 'Who are you?' }
      ], { maxTokens: 16, temperature: 0 });
    } catch (e: any) {
      // Template error is expected for some models — not a crash
    }
  });

  await test('alternating user/assistant messages', async () => {
    await llm.complete([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'How are you?' },
      { role: 'assistant', content: 'Good!' },
      { role: 'user', content: 'Great!' },
    ], { maxTokens: 16, temperature: 0 });
  });

  await test('tool message without prior tool call', async () => {
    try {
      await llm.complete([
        { role: 'user', content: 'Hi' },
        { role: 'tool', content: '{"result": "42"}', tool_call_id: 'fake_id', name: 'get_answer' },
      ], { maxTokens: 16, temperature: 0 });
    } catch (e: any) {
      // May fail with template error, shouldn't crash
    }
  });

  await test('concurrent completions', async () => {
    const promises = Array.from({ length: 3 }, (_, i) =>
      llm.complete([
        { role: 'user', content: `Count to ${i + 1}` }
      ], { maxTokens: 10, temperature: 0 })
    );
    try {
      await Promise.all(promises);
    } catch (e: any) {
      // May fail if context doesn't support concurrent access, but shouldn't crash
    }
  });

  await test('stream basic', async () => {
    const chunks: string[] = [];
    for await (const chunk of llm.stream([
      { role: 'user', content: 'Say hi' }
    ], { maxTokens: 10, temperature: 0 })) {
      if (chunk.content) chunks.push(chunk.content);
      if (chunk.done) break;
    }
  });

  await test('stream with immediate abort signal', async () => {
    const controller = new AbortController();
    controller.abort(); // Abort immediately
    try {
      for await (const chunk of llm.stream([
        { role: 'user', content: 'Count to a million' }
      ], { maxTokens: 1000, temperature: 0, signal: controller.signal })) {
        // Should stop quickly
      }
    } catch (e: any) {
      // AbortError is fine
    }
  });

  await test('double unload', async () => {
    await llm.unload();
    await llm.unload(); // Should not crash on second unload
  });

  await test('use after unload throws', async () => {
    try {
      await llm.complete([{ role: 'user', content: 'Hi' }]);
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('load after unload (re-use model object)', async () => {
    await llm.load({ contextSize: 2048, gpuLayers: 0 });
    const result = await llm.complete([
      { role: 'user', content: 'Say OK' }
    ], { maxTokens: 5, temperature: 0, thinkingBudget: 0 });
    if (!result.content) throw new Error('Should have content after reload');
    await llm.unload();
  });
}

// ─── STT Crash Tests ───

async function testSTT() {
  await section('STT Engine');

  if (!existsSync(WHISPER_MODEL)) {
    console.log('  SKIP: No whisper model at ' + WHISPER_MODEL);
    return;
  }

  await test('load STT with invalid path throws', async () => {
    try {
      await loadModel('/nonexistent.bin', { type: 'stt' });
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  let stt: SttModel;

  await test('load STT model', async () => {
    stt = await loadModel(WHISPER_MODEL, { type: 'stt' }) as SttModel;
    if (!stt.loaded) throw new Error('Should be loaded');
  });

  await test('transcribe before load throws', async () => {
    const model = createModel(WHISPER_MODEL, 'stt');
    try {
      await model.transcribe(Buffer.alloc(0));
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('transcribe empty buffer', async () => {
    try {
      await stt.transcribe(Buffer.alloc(0));
    } catch (e: any) {
      // Error is fine, crash is not
    }
  });

  await test('transcribe tiny buffer (1 byte)', async () => {
    try {
      await stt.transcribe(Buffer.alloc(1));
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('transcribe random noise buffer', async () => {
    const noise = Buffer.alloc(32000 * 2); // 1 second of 16kHz 16-bit
    for (let i = 0; i < noise.length; i++) noise[i] = Math.floor(Math.random() * 256);
    await stt.transcribe(noise, { language: 'en' });
  });

  await test('transcribe silence', async () => {
    const silence = Buffer.alloc(32000 * 2); // 1 second of silence
    await stt.transcribe(silence, { language: 'en' });
  });

  if (existsSync(TEST_AUDIO)) {
    await test('transcribe valid audio', async () => {
      const audio = readFileSync(TEST_AUDIO);
      const result = await stt.transcribe(audio, { language: 'en' });
      if (!result.text && result.text !== '') throw new Error('Should have text');
    });
  }

  await test('detectLanguage with empty buffer', async () => {
    try {
      await stt.detectLanguage(Buffer.alloc(0));
    } catch (e: any) {
      // Error fine
    }
  });

  await test('concurrent transcriptions', async () => {
    const noise = Buffer.alloc(32000 * 2);
    try {
      await Promise.all([
        stt.transcribe(noise, { language: 'en' }),
        stt.transcribe(noise, { language: 'en' }),
        stt.transcribe(noise, { language: 'en' }),
      ]);
    } catch (e: any) {
      // May fail but shouldn't crash
    }
  });

  await test('double unload STT', async () => {
    await stt.unload();
    await stt.unload();
  });

  await test('transcribe after unload throws', async () => {
    try {
      await stt.transcribe(Buffer.alloc(100));
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });
}

// ─── TTS Crash Tests ───

async function testTTS() {
  await section('TTS Engine');

  if (!existsSync(TTS_DIR)) {
    console.log('  SKIP: No TTS model at ' + TTS_DIR);
    return;
  }

  await test('load TTS with invalid path throws', async () => {
    try {
      await loadModel('/nonexistent/', { type: 'tts' });
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  let tts: TtsModel;

  await test('load TTS model', async () => {
    tts = await loadModel(TTS_DIR, { type: 'tts' }) as TtsModel;
    if (!tts.loaded) throw new Error('Should be loaded');
  });

  await test('speak empty string', async () => {
    try {
      await tts.speak('');
    } catch (e: any) {
      // Error is fine, crash is not
    }
  });

  await test('speak single character', async () => {
    try {
      const wav = await tts.speak('A');
      if (!wav || wav.length === 0) throw new Error('Expected WAV output');
    } catch (e: any) {
      // Error is fine
    }
  });

  await test('speak unicode/emoji', async () => {
    try {
      await tts.speak('你好世界 🌍');
    } catch (e: any) {
      // May not support, but shouldn't crash
    }
  });

  await test('speak with non-existent reference audio', async () => {
    try {
      await tts.speak('Hello', { referenceAudioPath: '/nonexistent.wav' });
    } catch (e: any) {
      // Should error, not crash
    }
  });

  await test('speak with temperature 0', async () => {
    try {
      await tts.speak('Hello world', { temperature: 0 });
    } catch (e: any) {
      // May fail but shouldn't crash
    }
  });

  await test('double unload TTS', async () => {
    await tts.unload();
    await tts.unload();
  });

  await test('speak after unload throws', async () => {
    try {
      await tts.speak('Hello');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });
}

// ─── Image Crash Tests ───

async function testImage() {
  await section('Image Engine');

  await test('createImageModel with invalid path', async () => {
    const model = createModel('/nonexistent.gguf', 'image');
    try {
      await model.load();
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('generate before load throws', async () => {
    const model = createModel('/nonexistent.gguf', 'image');
    try {
      await model.generate('a cat');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });
}

// ─── Binding loader edge cases ───

async function testBindingLoader() {
  await section('Binding Loader');

  await test('loadBinding returns same instance', async () => {
    const { loadBinding } = await import('../src/binding-loader.ts');
    const b1 = loadBinding();
    const b2 = loadBinding();
    if (b1 !== b2) throw new Error('Should return cached instance');
  });
}

// ─── Run all tests ───

async function main() {
  console.log('========================================');
  console.log('  CRASH TEST SUITE - node-omni-orcha');
  console.log('========================================');
  console.log(`GPU: ${JSON.stringify(detectGpu())}`);
  console.log(`LLM model: ${existsSync(LLM_MODEL) ? 'FOUND' : 'MISSING'}`);
  console.log(`Whisper model: ${existsSync(WHISPER_MODEL) ? 'FOUND' : 'MISSING'}`);
  console.log(`TTS dir: ${existsSync(TTS_DIR) ? 'FOUND' : 'MISSING'}`);

  await testGGUFReader();
  await testGpuDetection();
  await testBindingLoader();
  await testLLM();
  await testSTT();
  await testTTS();
  await testImage();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed, ${crashed} crashed`);
  console.log('========================================');

  if (crashed > 0) {
    process.exit(2);
  } else if (failed > 0) {
    process.exit(1);
  }
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
