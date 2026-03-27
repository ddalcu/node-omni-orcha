/**
 * Crash test ROUND 4 — Image mutex, abandoned streams, video generation,
 * cross-type method abuse
 */
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { loadModel, createModel, detectGpu } from '../src/index.ts';
import type { LlmModel, ImageModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');

const LLM_MODEL = path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');
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
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('TIMEOUT')), timeoutMs)
      ),
    ]);
    console.log('PASS');
    passed++;
  } catch (err: any) {
    console.log(`FAIL: ${err.message?.substring(0, 120)}`);
    failed++;
  }
}

// ─── Image concurrent + unload tests ───

async function testImageConcurrency() {
  console.log('\n=== Image Model Concurrency (FLUX 2) ===');

  const hasFlux = existsSync(FLUX_MODEL) && existsSync(FLUX_LLM) && existsSync(FLUX_VAE);
  if (!hasFlux) {
    console.log('  SKIP: No FLUX model');
    return;
  }

  let model: ImageModel;

  await test('load FLUX model', async () => {
    model = createModel(FLUX_MODEL, 'image');
    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
  });

  // Concurrent image generation — previously would crash without mutex
  await test('concurrent image generate (2x)', async () => {
    const [png1, png2] = await Promise.all([
      model.generate('a red circle', { width: 256, height: 256, steps: 2, cfgScale: 1.0, seed: 1 }),
      model.generate('a blue square', { width: 256, height: 256, steps: 2, cfgScale: 1.0, seed: 2 }),
    ]);
    if (png1[0] !== 0x89 || png2[0] !== 0x89) throw new Error('Invalid PNG');
  });

  // Sequential after concurrent to verify state
  await test('sequential generate after concurrent', async () => {
    const png = await model.generate('a green triangle', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
    if (png[0] !== 0x89) throw new Error('Invalid PNG');
  });

  // Unload during generation
  await test('unload during image generation', async () => {
    const genPromise = model.generate('a long detailed landscape', {
      width: 512, height: 512, steps: 8, cfgScale: 1.0,
    });

    // Let it start
    await new Promise(r => setTimeout(r, 100));

    // Unload should wait for generation to complete (via mutex)
    await model.unload();

    try {
      const png = await genPromise;
      // If we get here, generation completed before unload freed context
      if (png[0] !== 0x89) throw new Error('Invalid PNG');
    } catch (e: any) {
      // Error is acceptable
    }
  });

  // Reload and use after unload-during-gen
  await test('reload after unload-during-gen', async () => {
    model = createModel(FLUX_MODEL, 'image');
    await model.load({ llmPath: FLUX_LLM, vaePath: FLUX_VAE, keepVaeOnCpu: true });
    const png = await model.generate('recovery test', { width: 256, height: 256, steps: 2, cfgScale: 1.0 });
    if (png[0] !== 0x89) throw new Error('Invalid PNG');
    await model.unload();
  });
}

// ─── Abandoned stream tests ───

async function testAbandonedStreams() {
  console.log('\n=== Abandoned Streams ===');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  let llm: LlmModel;

  await test('load LLM', async () => {
    llm = await loadModel(LLM_MODEL, {
      type: 'llm', contextSize: 2048, gpuLayers: -1,
    }) as LlmModel;
  });

  // Break from stream early (without abort signal)
  await test('break from stream after 2 chunks', async () => {
    let count = 0;
    for await (const chunk of llm.stream(
      [{ role: 'user', content: 'Count from 1 to 100' }],
      { maxTokens: 200, temperature: 0, thinkingBudget: 0 }
    )) {
      count++;
      if (count >= 2) break; // Early exit without abort
    }
  });

  // Model should still be usable after abandoned stream
  await test('complete after abandoned stream', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('No content');
  });

  // Multiple abandoned streams in sequence
  await test('3 sequential abandoned streams', async () => {
    for (let i = 0; i < 3; i++) {
      for await (const chunk of llm.stream(
        [{ role: 'user', content: `Iteration ${i}` }],
        { maxTokens: 100, temperature: 0, thinkingBudget: 0 }
      )) {
        break; // Abandon immediately
      }
    }
    // Verify model is still alive
    const r = await llm.complete(
      [{ role: 'user', content: 'Still alive?' }],
      { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
    );
    if (!r.content) throw new Error('Dead after abandoned streams');
  });

  // Abandoned stream + concurrent complete
  await test('abandoned stream then immediate complete', async () => {
    // Start stream, take 1 chunk, break, immediately complete
    for await (const chunk of llm.stream(
      [{ role: 'user', content: 'Long story' }],
      { maxTokens: 50, temperature: 0, thinkingBudget: 0 }
    )) {
      break;
    }
    // This should wait for the stream's native worker to finish (mutex)
    const r = await llm.complete(
      [{ role: 'user', content: 'Quick' }],
      { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
    );
  });

  // Stream with error in template (empty messages)
  await test('stream with empty messages (template error)', async () => {
    try {
      for await (const chunk of llm.stream([], { maxTokens: 5 })) {
        // Should not get here
      }
    } catch (e: any) {
      // Error is expected
    }
    // Model should still work
    const r = await llm.complete(
      [{ role: 'user', content: 'OK?' }],
      { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
    );
  });

  await llm.unload();
}

// ─── Extreme parameter values on CUDA ───

async function testExtremeParams() {
  console.log('\n=== Extreme Parameters on CUDA ===');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  let llm: LlmModel;

  await test('load LLM on CUDA', async () => {
    llm = await loadModel(LLM_MODEL, {
      type: 'llm', contextSize: 4096, gpuLayers: -1,
    }) as LlmModel;
  });

  await test('maxTokens = MAX_SAFE_INTEGER', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'Hi' }],
        { maxTokens: Number.MAX_SAFE_INTEGER, temperature: 0, thinkingBudget: 0 }
      );
    } catch (e: any) {
      // Fine
    }
  });

  await test('contextSize boundary: fill to near capacity', async () => {
    // Create a long prompt that nearly fills the 4096 context
    const longContent = 'word '.repeat(3000); // ~3000 tokens
    try {
      const r = await llm.complete(
        [{ role: 'user', content: longContent }],
        { maxTokens: 5, temperature: 0, thinkingBudget: 0 }
      );
    } catch (e: any) {
      // Context overflow is fine
    }
  });

  await test('thinkingBudget = MAX_SAFE_INTEGER', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'Hi' }],
        { maxTokens: 10, temperature: 0, thinkingBudget: Number.MAX_SAFE_INTEGER }
      );
    } catch (e: any) {
      // Fine
    }
  });

  await test('temperature = Number.MIN_VALUE', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Say OK' }],
      { maxTokens: 5, temperature: Number.MIN_VALUE, thinkingBudget: 0 }
    );
  });

  await test('all stop sequences match everything', async () => {
    const r = await llm.complete(
      [{ role: 'user', content: 'Hello' }],
      { maxTokens: 50, temperature: 0, thinkingBudget: 0, stopSequences: ['a', 'e', 'i', 'o', 'u', ' ', '\n'] }
    );
  });

  // Tool with deeply nested JSON schema
  await test('tool with complex nested schema', async () => {
    try {
      await llm.complete(
        [{ role: 'user', content: 'Do something' }],
        {
          maxTokens: 64, temperature: 0, thinkingBudget: 0,
          tools: [{
            name: 'complex_tool',
            description: 'A tool with deeply nested params',
            parameters: {
              type: 'object',
              properties: {
                level1: {
                  type: 'object',
                  properties: {
                    level2: {
                      type: 'object',
                      properties: {
                        level3: {
                          type: 'array',
                          items: {
                            type: 'object',
                            properties: {
                              name: { type: 'string' },
                              values: { type: 'array', items: { type: 'number' } },
                            },
                          },
                        },
                      },
                    },
                  },
                },
              },
            },
          }],
          toolChoice: 'required',
        }
      );
    } catch (e: any) {
      // Template/grammar errors are fine
    }
  });

  await llm.unload();
}

// ─── Double-loading race (concurrent loads) ───

async function testConcurrentLoadRace() {
  console.log('\n=== Concurrent Load Race ===');

  if (!existsSync(LLM_MODEL)) {
    console.log('  SKIP: No LLM model');
    return;
  }

  await test('3 concurrent loads on same model object', async () => {
    const model = createModel(LLM_MODEL, 'llm');
    // Fire 3 loads concurrently — the `if (loaded) return` guard may race
    await Promise.all([
      model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false }),
      model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false }),
      model.load({ contextSize: 2048, gpuLayers: 0, flashAttn: false }),
    ]);
    // Should have loaded exactly once
    if (!model.loaded) throw new Error('Not loaded');
    const r = await model.complete(
      [{ role: 'user', content: 'Hi' }],
      { maxTokens: 3, temperature: 0, thinkingBudget: 0 }
    );
    await model.unload();
  });
}

// ─── Run all ───

async function main() {
  const gpu = detectGpu();
  console.log('========================================');
  console.log('  CRASH TEST ROUND 4');
  console.log(`  GPU: ${gpu.name} (${gpu.backend})`);
  console.log('========================================');

  await testImageConcurrency();
  await testAbandonedStreams();
  await testExtremeParams();
  await testConcurrentLoadRace();

  console.log('\n========================================');
  console.log(`  RESULTS: ${passed} passed, ${failed} failed`);
  console.log('========================================');

  if (failed > 0) process.exit(1);
}

main().catch(err => {
  console.error('FATAL CRASH:', err);
  process.exit(3);
});
