/**
 * CUDA-specific crash tests — exercises GPU code paths
 */
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { loadModel, createModel, detectGpu } from '../src/index.ts';
import type { LlmModel, ChatMessage } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');
const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
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
    console.log(`FAIL: ${err.message}`);
    failed++;
  }
}

const gpu = detectGpu();
console.log('========================================');
console.log('  CUDA CRASH TEST SUITE');
console.log(`  GPU: ${gpu.name} (${gpu.backend})`);
console.log(`  VRAM: ${gpu.vramBytes ? Math.round(gpu.vramBytes / 1024 / 1024) + ' MiB' : 'unknown'}`);
console.log('========================================');

if (gpu.backend !== 'cuda') {
  console.log('SKIP: No CUDA GPU detected');
  process.exit(0);
}

async function testLLMCuda() {
  console.log('\n=== LLM on CUDA (all GPU layers) ===');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  let llm: LlmModel;

  await test('load LLM with all GPU layers (gpuLayers=-1)', async () => {
    llm = await loadModel(LLM_MODEL, {
      type: 'llm',
      contextSize: 4096,
      gpuLayers: -1, // All layers on GPU
    }) as LlmModel;
    if (!llm.loaded) throw new Error('Should be loaded');
  });

  await test('basic GPU completion', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Say hello in one word.' }],
      { maxTokens: 16, temperature: 0, thinkingBudget: 0 }
    );
    console.log(`[${r.content}] `);
    if (!r.content) throw new Error('No content');
  });

  await test('GPU completion with reasoning enabled', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'What is 2+2?' }],
      { maxTokens: 128, temperature: 0 }
    );
    // With thinking enabled, may have reasoning and/or content
    const hasOutput = r.content || r.reasoning;
    if (!hasOutput) throw new Error('No output');
  });

  await test('GPU completion with thinkingBudget=10', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'What is the capital of France?' }],
      { maxTokens: 64, temperature: 0, thinkingBudget: 10 }
    );
    const hasOutput = r.content || r.reasoning;
    if (!hasOutput) throw new Error('No output');
  });

  await test('GPU stream', async () => {
    const chunks: string[] = [];
    for await (const chunk of llm.stream(
      [{ role: 'user', content: 'Count from 1 to 5.' }],
      { maxTokens: 50, temperature: 0, thinkingBudget: 0 }
    )) {
      if (chunk.content) chunks.push(chunk.content);
      if (chunk.done) break;
    }
    if (chunks.length === 0) throw new Error('No stream chunks');
  });

  await test('GPU stream with mid-abort', async () => {
    const controller = new AbortController();
    let count = 0;
    for await (const chunk of llm.stream(
      [{ role: 'user', content: 'Write a very long essay about everything.' }],
      { maxTokens: 500, temperature: 0.5, thinkingBudget: 0, signal: controller.signal }
    )) {
      count++;
      if (count >= 5) controller.abort();
      if (chunk.done) break;
    }
  });

  await test('GPU recovery after abort', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content after abort recovery');
  });

  await test('concurrent GPU completions (3x, serialized by mutex)', async () => {
    const promises = [1, 2, 3].map(i =>
      llm.complete(
        [{ role: 'user', content: `Number ${i}` }],
        { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
      )
    );
    const results = await Promise.all(promises);
    for (const r of results) {
      if (!r.content && r.content !== '') throw new Error('Missing content');
    }
  });

  await test('rapid GPU completions (10x sequential)', async () => {
    for (let i = 0; i < 10; i++) {
      await llm.complete(
        [{ role: 'user', content: `Quick ${i}` }],
        { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
      );
    }
  });

  await test('long GPU generation (256 tokens)', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Write a short poem about the ocean.' }],
      { maxTokens: 256, temperature: 0.7, thinkingBudget: 0 }
    );
    if (r.usage.outputTokens < 10) throw new Error('Too few tokens generated');
  });

  await test('GPU tool calling', async () => {
    try {
      const r = await llm.complete(
        [{ role: 'user', content: 'What is the weather in Tokyo?' }],
        {
          maxTokens: 128, temperature: 0, thinkingBudget: 0,
          tools: [{
            name: 'get_weather',
            description: 'Get weather for a city',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
          }],
          toolChoice: 'required',
        }
      );
      if (r.toolCalls && r.toolCalls.length > 0) {
        console.log(`[tool: ${r.toolCalls[0].name}(${r.toolCalls[0].args})] `);
      }
    } catch (e: any) {
      // Template errors are OK, just shouldn't crash
    }
  });

  await test('deep conversation on GPU', async () => {
    const messages: ChatMessage[] = [];
    for (let i = 0; i < 10; i++) {
      messages.push({ role: 'user', content: `Turn ${i}: tell me a fact.` });
      messages.push({ role: 'assistant', content: `Fact ${i}: The sky is blue.` });
    }
    messages.push({ role: 'user', content: 'Summarize.' });
    try {
      await llm.complete(messages, { maxTokens: 32, temperature: 0, thinkingBudget: 0 });
    } catch (e: any) {
      // Context overflow is OK
    }
  });

  await test('GPU unload and reload', async () => {
    await llm.unload();
    await llm.load({ contextSize: 2048, gpuLayers: -1 });
    const r = await llm.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content after reload');
  });

  await test('double unload GPU', async () => {
    await llm.unload();
    await llm.unload();
  });
}

async function testLLMCudaOomFallback() {
  console.log('\n=== CUDA OOM Fallback ===');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  await test('huge context size triggers OOM fallback', async () => {
    // Request enormous context — should trigger the binary search OOM fallback
    try {
      const llm = await loadModel(LLM_MODEL, {
        type: 'llm',
        contextSize: 131072, // 128K context — will likely OOM on 8GB card
        gpuLayers: -1,
      }) as LlmModel;
      // If it loads (maybe with reduced GPU layers), that's fine
      const r = await llm.complete(
        [{ role: 'user', content: 'Hi' }],
        { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
      );
      await llm.unload();
    } catch (e: any) {
      // OOM that we couldn't recover from — as long as no crash, that's OK
      console.log(`[${e.message.substring(0, 60)}] `);
    }
  });
}

async function main() {
  await testLLMCuda();
  await testLLMCudaOomFallback();

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
