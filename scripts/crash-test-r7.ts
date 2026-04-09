/**
 * Crash test ROUND 7 — Final sweep
 * Invalid Jinja templates, garbage image buffers, image reload,
 * multi-turn tool calling, custom chatTemplate, backend idempotency
 */
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { loadModel, createModel, detectGpu } from '../src/index.ts';
import { loadBinding } from '../src/binding-loader.ts';
import type { LlmModel, ImageModel, TtsModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');
const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
const EMBED_MODEL = path.join(MODELS_DIR, 'nomic-embed-text-v1-5-q4_k_m', 'nomic-embed-text-v1.5.Q4_K_M.gguf');
const TTS_DIR = path.join(MODELS_DIR, 'qwen3-tts');
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

// ─── Invalid Jinja Chat Template ───

async function testInvalidChatTemplate() {
  console.log('\n=== Invalid Chat Templates ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  const binding = loadBinding();

  await test('load with broken Jinja template', async () => {
    try {
      await (binding['createLlmContext'] as Function)(LLM_MODEL, {
        contextSize: 2048, gpuLayers: 0, flashAttn: false,
        embeddings: false, batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16',
        chatTemplate: '{% for msg in messages %}{{ msg.invalid_field }}{% endfor %}{{ raise_exception("boom") }}',
      });
    } catch (e: any) {
      // May or may not fail at load, but shouldn't crash
    }
  });

  await test('load with empty Jinja template (uses model default)', async () => {
    const ctx = await (binding['createLlmContext'] as Function)(LLM_MODEL, {
      contextSize: 2048, gpuLayers: 0, flashAttn: false,
      embeddings: false, batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16',
      chatTemplate: '',
    });
    await (ctx.unload as Function)();
  });

  await test('complete with custom (valid) template', async () => {
    const ctx = await (binding['createLlmContext'] as Function)(LLM_MODEL, {
      contextSize: 2048, gpuLayers: 0, flashAttn: false,
      embeddings: false, batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16',
      chatTemplate: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}",
    });
    try {
      const r = await (ctx.complete as Function)(
        [{ role: 'user', content: 'Say OK' }],
        { temperature: 0, maxTokens: 5, stopSequences: [], thinkingBudget: 0 }
      );
    } catch (e: any) {
      // Template rendering error is fine
    }
    await (ctx.unload as Function)();
  });

  await test('complete triggers template error at runtime', async () => {
    try {
      const ctx = await (binding['createLlmContext'] as Function)(LLM_MODEL, {
        contextSize: 2048, gpuLayers: 0, flashAttn: false,
        embeddings: false, batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16',
        chatTemplate: '{% if messages|length == 0 %}{{ raise_exception("no messages") }}{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}',
      });
      // Try with messages that trigger the error
      await (ctx.complete as Function)([], { temperature: 0, maxTokens: 5, stopSequences: [] });
    } catch (e: any) {
      // Template error is fine, crash is not
    }
  });
}

// ─── Video initImage with garbage data ───

async function testGarbageInitImage() {
  console.log('\n=== Garbage initImage Data ===');

  const hasWan = existsSync(WAN_MODEL) && existsSync(WAN_VAE) && existsSync(UMT5_PATH);
  if (!hasWan) { console.log('  SKIP: No WAN model'); return; }

  let model: ImageModel;

  await test('load WAN model', async () => {
    model = createModel(WAN_MODEL, 'image');
    await model.load({
      t5xxlPath: UMT5_PATH, vaePath: WAN_VAE,
      flashAttn: true, vaeDecodeOnly: true, keepVaeOnCpu: true,
    });
  });

  await test('generateVideo with random bytes as initImage', async () => {
    const garbage = Buffer.alloc(1000);
    for (let i = 0; i < garbage.length; i++) garbage[i] = Math.random() * 256;
    try {
      await model.generateVideo('test', {
        videoFrames: 5, steps: 2, width: 480, height: 272,
        initImage: garbage,
      });
    } catch (e: any) {
      // Error or degrade gracefully (stbi returns null, ignored)
    }
  });

  await test('generateVideo with empty initImage buffer', async () => {
    try {
      await model.generateVideo('test', {
        videoFrames: 5, steps: 2, width: 480, height: 272,
        initImage: Buffer.alloc(0),
      });
    } catch (e: any) {
      // Fine
    }
  });

  await test('generateVideo with 1-byte initImage', async () => {
    try {
      await model.generateVideo('test', {
        videoFrames: 5, steps: 2, width: 480, height: 272,
        initImage: Buffer.from([0xFF]),
      });
    } catch (e: any) {
      // Fine
    }
  });

  // Verify model still works after bad inputs
  await test('normal generateVideo after garbage inputs', async () => {
    const frames = await model.generateVideo('a sunset', {
      videoFrames: 5, steps: 2, width: 480, height: 272, seed: 42,
    });
    if (frames.length === 0) throw new Error('No frames');
    if (frames[0][0] !== 0x89) throw new Error('Invalid PNG');
  });

  await model.unload();
}

// ─── Image model reload cycle ───

async function testImageReload() {
  console.log('\n=== Image Model Reload Cycle ===');

  const hasFlux = existsSync(FLUX_MODEL) && existsSync(FLUX_LLM) && existsSync(FLUX_VAE);
  if (!hasFlux) { console.log('  SKIP'); return; }

  await test('image load → generate → unload → reload → generate', async () => {
    const model = createModel(FLUX_MODEL, 'image');

    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
    const png1 = await model.generate('test 1', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
    if (png1[0] !== 0x89) throw new Error('First gen failed');

    await model.unload();
    if (model.loaded) throw new Error('Should be unloaded');

    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
    const png2 = await model.generate('test 2', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
    if (png2[0] !== 0x89) throw new Error('Second gen failed');

    await model.unload();
  });
}

// ─── Multi-turn tool calling flow ───

async function testMultiTurnToolCalling() {
  console.log('\n=== Multi-Turn Tool Calling ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  let llm: LlmModel;

  await test('load LLM', async () => {
    llm = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 4096, gpuLayers: -1 }) as LlmModel;
  });

  const weatherTool = {
    name: 'get_weather',
    description: 'Get the current weather for a city',
    parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
  };

  await test('full tool calling round-trip', async () => {
    // Step 1: user asks, model calls tool
    const r1 = await llm.complete(
      [{ role: 'user', content: 'What is the weather in Tokyo?' }],
      { maxTokens: 128, temperature: 0, tools: [weatherTool], toolChoice: 'required' }
    );

    if (!r1.toolCalls || r1.toolCalls.length === 0) {
      // Some models may not call tools; that's not a crash
      return;
    }

    // Step 2: provide tool response, get final answer
    const tc = r1.toolCalls[0];
    const r2 = await llm.complete([
      { role: 'user', content: 'What is the weather in Tokyo?' },
      { role: 'assistant', content: r1.content || '', tool_calls: [tc] },
      { role: 'tool', content: '{"temperature": 22, "condition": "sunny"}', tool_call_id: tc.id, name: tc.name },
    ], { maxTokens: 128, temperature: 0, tools: [weatherTool], thinkingBudget: 0 });

    // Should produce a text response about the weather
    if (!r2.content && !r2.reasoning) throw new Error('No response after tool result');
  });

  await test('tool calling with multiple tools', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'What is the weather in Tokyo and translate hello to Japanese?' }],
        {
          maxTokens: 128, temperature: 0, thinkingBudget: 0,
          tools: [
            weatherTool,
            { name: 'translate', description: 'Translate text', parameters: { type: 'object', properties: { text: { type: 'string' }, language: { type: 'string' } } } },
          ],
          toolChoice: 'required',
        }
      );
    } catch (e: any) {
      // Grammar errors are fine
    }
  });

  await llm.unload();
}

// ─── Multiple LLM backend inits ───

async function testBackendIdempotency() {
  console.log('\n=== Backend Init Idempotency ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  await test('create 3 LLM contexts sequentially (shared backend)', async () => {
    const binding = loadBinding();
    const opts = {
      contextSize: 2048, gpuLayers: 0, flashAttn: false,
      embeddings: false, batchSize: 512, cacheTypeK: 'f16', cacheTypeV: 'f16', chatTemplate: '',
    };

    const ctx1 = await (binding['createLlmContext'] as Function)(LLM_MODEL, opts);
    const ctx2 = await (binding['createLlmContext'] as Function)(LLM_MODEL, opts);
    const ctx3 = await (binding['createLlmContext'] as Function)(LLM_MODEL, opts);

    // Use all three
    const r1 = await (ctx1.complete as Function)([{ role: 'user', content: 'A' }], { temperature: 0, maxTokens: 3, stopSequences: [], thinkingBudget: 0 });
    const r2 = await (ctx2.complete as Function)([{ role: 'user', content: 'B' }], { temperature: 0, maxTokens: 3, stopSequences: [], thinkingBudget: 0 });
    const r3 = await (ctx3.complete as Function)([{ role: 'user', content: 'C' }], { temperature: 0, maxTokens: 3, stopSequences: [], thinkingBudget: 0 });

    // Unload in reverse order
    await (ctx3.unload as Function)();
    await (ctx2.unload as Function)();
    await (ctx1.unload as Function)();
  });
}

// ─── All model types reload cycle ───

async function testAllModelReload() {
  console.log('\n=== All Model Types Reload ===');

  if (existsSync(TTS_DIR)) {
    await test('TTS load → use → unload → reload → use', async () => {
      const model = createModel(TTS_DIR, 'tts');
      await model.load();
      await model.speak('Hello');
      await model.unload();
      await model.load();
      await model.speak('World');
      await model.unload();
    });
  }

  if (existsSync(LLM_MODEL)) {
    await test('LLM load(CPU) → unload → load(GPU) → use → unload', async () => {
      const model = createModel(LLM_MODEL, 'llm');
      await model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false });
      await model.unload();
      await model.load({ contextSize: 2048, gpuLayers: -1 });
      const r = await model.complete(
        [{ role: 'user', content: 'OK' }],
        { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
      );
      if (!r.content) throw new Error('No content');
      await model.unload();
    });
  }
}

// ─── Throw from stream consumer ───

async function testStreamConsumerThrow() {
  console.log('\n=== Stream Consumer Error Recovery ===');

  if (!existsSync(LLM_MODEL)) { console.log('  SKIP'); return; }

  let llm: LlmModel;

  await test('load LLM', async () => {
    llm = await loadModel(LLM_MODEL, { type: 'llm', contextSize: 2048, gpuLayers: -1 }) as LlmModel;
  });

  await test('throw from stream consumer (for-await catch)', async () => {
    try {
      for await (const chunk of llm.stream(
        [{ role: 'user', content: 'Count forever' }],
        { maxTokens: 100, temperature: 0, thinkingBudget: 0 }
      )) {
        throw new Error('Consumer error!');
      }
    } catch (e: any) {
      if (e.message !== 'Consumer error!') throw e;
    }
  });

  // Model should still be usable after consumer threw
  await test('complete after consumer throw', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Still alive?' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('Dead after consumer throw');
  });

  await llm.unload();
}

// ─── Run ───

async function main() {
  const gpu = detectGpu();
  console.log('========================================');
  console.log('  CRASH TEST ROUND 7 — Final Sweep');
  console.log(`  GPU: ${gpu.name} (${gpu.backend})`);
  console.log('========================================');

  await testInvalidChatTemplate();
  await testBackendIdempotency();
  await testMultiTurnToolCalling();
  await testStreamConsumerThrow();
  await testAllModelReload();
  await testImageReload();
  await testGarbageInitImage();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed`);
  console.log('========================================');

  if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
