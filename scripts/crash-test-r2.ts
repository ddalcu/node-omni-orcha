/**
 * Crash test ROUND 2 — deeper edge cases
 * Focus: embedding on non-embedding model, stream error recovery,
 * load/unload races, ParseMessages crashes, type mismatches, image reuse
 */
import * as path from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { loadModel, createModel } from '../src/index.ts';
import { loadBinding } from '../src/binding-loader.ts';
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

// ─── LLM Deep Tests ───

async function testLLMDeep() {
  await section('LLM Deep Edge Cases');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  let llm: LlmModel;
  await test('load LLM', async () => {
    llm = await loadModel(LLM_MODEL, {
      type: 'llm', contextSize: 2048, gpuLayers: 0, flashAttn: false,
    }) as LlmModel;
  });

  // Embed on a non-embedding model
  await test('embed on non-embedding model (should error, not crash)', async () => {
    try {
      await llm.embed('test text');
    } catch (e: any) {
      // Error is expected — model wasn't loaded with embeddings: true
    }
  });

  await test('embedBatch on non-embedding model', async () => {
    try {
      await llm.embedBatch(['hello', 'world']);
    } catch (e: any) {
      // Error is expected
    }
  });

  await test('embed empty string', async () => {
    try {
      await llm.embed('');
    } catch (e: any) {
      // May error
    }
  });

  // Stream then immediately abort via controller
  await test('stream mid-abort via controller', async () => {
    const controller = new AbortController();
    let count = 0;
    for await (const chunk of llm.stream(
      [{ role: 'user', content: 'Write a long essay about philosophy' }],
      { maxTokens: 200, temperature: 0.5, thinkingBudget: 0, signal: controller.signal }
    )) {
      count++;
      if (count >= 3) controller.abort();
      if (chunk.done) break;
    }
  });

  // Complete after stream abort — make sure context is still usable
  await test('complete after aborted stream (context recovery)', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('Should have content');
  });

  // Rapid fire: many sequential completions
  await test('rapid sequential completions (10x)', async () => {
    for (let i = 0; i < 10; i++) {
      await llm.complete(
        [{ role: 'user', content: `Number ${i}` }],
        { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
      );
    }
  });

  // Heavy concurrent (5 at once)
  await test('heavy concurrent completions (5x)', async () => {
    const promises = Array.from({ length: 5 }, (_, i) =>
      llm.complete(
        [{ role: 'user', content: `Task ${i}` }],
        { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
      )
    );
    await Promise.all(promises);
  });

  // Interleave stream and complete
  await test('interleaved stream + complete', async () => {
    const streamPromise = (async () => {
      const chunks: string[] = [];
      for await (const chunk of llm.stream(
        [{ role: 'user', content: 'Count 1-5' }],
        { maxTokens: 20, temperature: 0, thinkingBudget: 0 }
      )) {
        if (chunk.content) chunks.push(chunk.content);
        if (chunk.done) break;
      }
      return chunks.join('');
    })();

    const completePromise = llm.complete(
      [{ role: 'user', content: 'Say hi' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );

    await Promise.all([streamPromise, completePromise]);
  });

  // Tool call with malformed parameters
  await test('tool with empty parameters', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'What is the weather?' }],
        {
          maxTokens: 64, temperature: 0, thinkingBudget: 0,
          tools: [{ name: 'weather', description: 'Get weather', parameters: {} }],
          toolChoice: 'required',
        }
      );
    } catch (e: any) {
      // Template error is acceptable
    }
  });

  // Very deep message history
  await test('deep conversation history (50 messages)', async () => {
    const messages: ChatMessage[] = [];
    for (let i = 0; i < 25; i++) {
      messages.push({ role: 'user', content: `Message ${i}` });
      messages.push({ role: 'assistant', content: `Reply ${i}` });
    }
    try {
      await llm.complete(messages, { maxTokens: 5, temperature: 0, thinkingBudget: 0 });
    } catch (e: any) {
      // May exceed context, fine as long as no crash
    }
  });

  // Messages with only whitespace
  await test('whitespace-only content', async () => {
    await llm.complete(
      [{ role: 'user', content: '   \n\t   ' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
  });

  // Stop sequences
  await test('stop sequences with exact match', async () => {
    await llm.complete(
      [{ role: 'user', content: 'Say the word STOP' }],
      { maxTokens: 50, temperature: 0, thinkingBudget: 0, stopSequences: ['STOP'] }
    );
  });

  await test('empty stop sequence', async () => {
    await llm.complete(
      [{ role: 'user', content: 'Hi' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0, stopSequences: [''] }
    );
  });

  // NaN / Infinity temperature
  await test('NaN temperature', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'Hi' }],
        { maxTokens: 5, temperature: NaN }
      );
    } catch (e: any) {
      // Fine
    }
  });

  await test('Infinity temperature', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'Hi' }],
        { maxTokens: 5, temperature: Infinity }
      );
    } catch (e: any) {
      // Fine
    }
  });

  await test('negative maxTokens', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'Hi' }],
        { maxTokens: -1, temperature: 0 }
      );
    } catch (e: any) {
      // Fine
    }
  });

  await test('huge maxTokens', async () => {
    await llm.complete(
      [{ role: 'user', content: 'Hi' }],
      { maxTokens: 999999, temperature: 0, thinkingBudget: 0 }
    );
  });

  await llm.unload();
}

// ─── Native Binding Direct Access ───

async function testNativeEdgeCases() {
  await section('Native Binding Edge Cases');

  const binding = loadBinding();

  // Pass wrong argument types directly to binding
  await test('createLlmContext with number instead of string', async () => {
    try {
      await (binding['createLlmContext'] as Function)(42, {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('createLlmContext with null path', async () => {
    try {
      await (binding['createLlmContext'] as Function)(null, {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('createLlmContext with undefined', async () => {
    try {
      await (binding['createLlmContext'] as Function)(undefined, {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('createLlmContext with no args', async () => {
    try {
      await (binding['createLlmContext'] as Function)();
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('createSttContext with number', async () => {
    try {
      await (binding['createSttContext'] as Function)(123);
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('createTtsContext with null', async () => {
    try {
      await (binding['createTtsContext'] as Function)(null, {});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  await test('createImageContext with no args', async () => {
    try {
      await (binding['createImageContext'] as Function)();
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });
}

// ─── STT Deep Tests ───

async function testSTTDeep() {
  await section('STT Deep Edge Cases');

  if (!existsSync(WHISPER_MODEL)) {
    console.log('  SKIP: No whisper model');
    return;
  }

  let stt: SttModel;
  await test('load STT', async () => {
    stt = await loadModel(WHISPER_MODEL, { type: 'stt' }) as SttModel;
  });

  // Very large buffer (10 seconds of noise)
  await test('large audio buffer (10s noise)', async () => {
    const big = Buffer.alloc(16000 * 2 * 10); // 10 seconds
    for (let i = 0; i < big.length; i += 2) {
      big.writeInt16LE(Math.floor(Math.random() * 65536 - 32768), i);
    }
    await stt.transcribe(big, { language: 'en' });
  });

  // Invalid language code
  await test('transcribe with invalid language code', async () => {
    const audio = Buffer.alloc(16000 * 2); // 1 second silence
    try {
      await stt.transcribe(audio, { language: 'zzzz_invalid' });
    } catch (e: any) {
      // Error is fine
    }
  });

  // Rapid sequential transcriptions
  await test('rapid sequential transcriptions (5x)', async () => {
    const audio = Buffer.alloc(16000 * 2);
    for (let i = 0; i < 5; i++) {
      await stt.transcribe(audio, { language: 'en' });
    }
  });

  // Heavy concurrent transcriptions (now with mutex)
  await test('heavy concurrent transcriptions (5x)', async () => {
    const audio = Buffer.alloc(16000 * 2);
    const promises = Array.from({ length: 5 }, () =>
      stt.transcribe(audio, { language: 'en' })
    );
    await Promise.all(promises);
  });

  // Concurrent transcribe + detectLanguage
  await test('concurrent transcribe + detectLanguage', async () => {
    const audio = Buffer.alloc(16000 * 2 * 2); // 2 seconds
    await Promise.all([
      stt.transcribe(audio, { language: 'en' }),
      stt.detectLanguage(audio),
    ]);
  });

  // Odd-sized buffer (not a multiple of 2)
  await test('odd-sized audio buffer (1001 bytes)', async () => {
    try {
      await stt.transcribe(Buffer.alloc(1001), { language: 'en' });
    } catch (e: any) {
      // May produce garbage results but shouldn't crash
    }
  });

  // Buffer with extreme values
  await test('audio buffer with max/min int16 values', async () => {
    const extreme = Buffer.alloc(16000 * 2);
    for (let i = 0; i < extreme.length; i += 2) {
      extreme.writeInt16LE(i % 4 === 0 ? 32767 : -32768, i);
    }
    await stt.transcribe(extreme, { language: 'en' });
  });

  await stt.unload();
}

// ─── TTS Deep Tests ───

async function testTTSDeep() {
  await section('TTS Deep Edge Cases');

  if (!existsSync(TTS_DIR)) {
    console.log('  SKIP: No TTS model');
    return;
  }

  let tts: TtsModel;
  await test('load TTS', async () => {
    tts = await loadModel(TTS_DIR, { type: 'tts' }) as TtsModel;
  });

  // Very long text
  await test('speak very long text (500 chars)', async () => {
    try {
      await tts.speak('Hello world. '.repeat(40));
    } catch (e: any) {
      // May hit token limit but shouldn't crash
    }
  });

  // Special characters
  await test('speak with special characters', async () => {
    try {
      await tts.speak('Hello! @#$%^&*()_+-={}[]|\\:";\'<>?,./~`');
    } catch (e: any) {
      // Fine
    }
  });

  // Numbers and mixed content
  await test('speak numbers and URLs', async () => {
    try {
      await tts.speak('Call 1-800-555-0123 or visit https://example.com');
    } catch (e: any) {
      // Fine
    }
  });

  // Concurrent speak (now with mutex)
  await test('concurrent speak (3x)', async () => {
    try {
      const promises = [
        tts.speak('Hello one'),
        tts.speak('Hello two'),
        tts.speak('Hello three'),
      ];
      await Promise.all(promises);
    } catch (e: any) {
      // May error but shouldn't crash
    }
  });

  // Temperature extremes
  await test('speak with temperature 100', async () => {
    try {
      await tts.speak('Hello', { temperature: 100 });
    } catch (e: any) {
      // Fine
    }
  });

  await tts.unload();
}

// ─── Image Edge Cases ───

async function testImageDeep() {
  await section('Image Deep Edge Cases');

  // Test with empty options
  await test('createImageModel with empty load options', async () => {
    const model = createModel('nonexistent.gguf', 'image');
    try {
      await model.load({});
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Generate on unloaded model
  await test('generate on unloaded image model', async () => {
    const model = createModel('nonexistent.gguf', 'image');
    try {
      await model.generate('a cat');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // generateVideo on unloaded model
  await test('generateVideo on unloaded image model', async () => {
    const model = createModel('nonexistent.gguf', 'image');
    try {
      await model.generateVideo('a cat walking');
      throw new Error('Should have thrown');
    } catch (e: any) {
      if (e.message === 'Should have thrown') throw e;
    }
  });

  // Double unload on unloaded model
  await test('unload on never-loaded image model', async () => {
    const model = createModel('nonexistent.gguf', 'image');
    await model.unload();
    await model.unload();
  });
}

// ─── Cross-engine tests ───

async function testCrossEngine() {
  await section('Cross-Engine Stress');

  // Load multiple models simultaneously
  const hasLlm = existsSync(LLM_MODEL);
  const hasStt = existsSync(WHISPER_MODEL);

  if (hasLlm && hasStt) {
    await test('concurrent model loading (LLM + STT)', async () => {
      const [llm, stt] = await Promise.all([
        loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048, gpuLayers: 0, flashAttn: false }),
        loadModel(WHISPER_MODEL, { type: 'stt' }),
      ]);
      // Use both
      const [completionResult, transcribeResult] = await Promise.all([
        (llm as LlmModel).complete(
          [{ role: 'user', content: 'Hi' }],
          { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
        ),
        (stt as SttModel).transcribe(Buffer.alloc(16000 * 2), { language: 'en' }),
      ]);
      await Promise.all([llm.unload(), stt.unload()]);
    });
  }

  // Load same model path twice (different instances)
  if (hasLlm) {
    await test('two LLM instances from same file', async () => {
      const llm1 = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048, gpuLayers: 0, flashAttn: false }) as LlmModel;
      const llm2 = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048, gpuLayers: 0, flashAttn: false }) as LlmModel;

      // Use both concurrently (different instances, different contexts — should be safe)
      const [r1, r2] = await Promise.all([
        llm1.complete([{ role: 'user', content: 'Say A' }], { maxTokens: 3, temperature: 0, thinkingBudget: 0 }),
        llm2.complete([{ role: 'user', content: 'Say B' }], { maxTokens: 3, temperature: 0, thinkingBudget: 0 }),
      ]);

      await Promise.all([llm1.unload(), llm2.unload()]);
    });
  }
}

// ─── loadBinding edge cases ───

async function testLoadBinding() {
  await section('loadBinding edge cases');

  const { loadBinding: lb } = await import('../src/binding-loader.ts');

  await test('loadBinding called 100 times', async () => {
    for (let i = 0; i < 100; i++) lb();
  });

  await test('binding has expected factory functions', async () => {
    const b = lb();
    if (typeof b['createLlmContext'] !== 'function') throw new Error('Missing createLlmContext');
    if (typeof b['createSttContext'] !== 'function') throw new Error('Missing createSttContext');
    if (typeof b['createTtsContext'] !== 'function') throw new Error('Missing createTtsContext');
    if (typeof b['createImageContext'] !== 'function') throw new Error('Missing createImageContext');
  });
}

// ─── Run ───

async function main() {
  console.log('========================================');
  console.log('  CRASH TEST ROUND 2 - Deep Edge Cases');
  console.log('========================================');

  await testLoadBinding();
  await testNativeEdgeCases();
  await testLLMDeep();
  await testSTTDeep();
  await testTTSDeep();
  await testImageDeep();
  await testCrossEngine();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed, ${crashed} crashed`);
  console.log('========================================');

  if (crashed > 0) process.exit(2);
  else if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
