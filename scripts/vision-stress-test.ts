/**
 * Vision/Multimodal stress test — tries to break the multimodal pipeline.
 *
 * Tests:
 *   1. Rapid sequential vision inferences
 *   2. Large images
 *   3. Many images in one prompt
 *   4. Mixed text+image conversations back-to-back
 *   5. Streaming + abort during vision inference
 *   6. Memory pressure: load, infer, unload, repeat
 *   7. Zero-dimension and tiny images
 *   8. Various image formats
 *
 * Usage:
 *   bash scripts/download-test-models.sh --vision
 *   node scripts/vision-stress-test.ts
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import * as path from 'node:path';
import { loadModel, createModel } from '../src/index.ts';
import type { LlmModel, ChatMessage } from '../src/types.ts';

const MODELS = process.env['MODELS_DIR'] || `${process.env['HOME']}/.orcha/workspace/.models`;
const VISION_DIR = path.join(MODELS, 'qwen2-vl-2b');
const MODEL_PATH = path.join(VISION_DIR, 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf');
const MMPROJ_PATH = path.join(VISION_DIR, 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf');
const TEST_IMAGE = path.join(VISION_DIR, 'test-image.png');

let passed = 0;
let failed = 0;

async function run(name: string, fn: () => Promise<void>) {
  process.stdout.write(`\n[STRESS] ${name} ... `);
  const start = Date.now();
  try {
    await fn();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`PASSED (${elapsed}s)`);
    passed++;
  } catch (err) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`FAILED (${elapsed}s)`);
    console.error(`  ${(err as Error).stack}\n`);
    failed++;
  }
}

// Check prerequisites
for (const f of [MODEL_PATH, MMPROJ_PATH, TEST_IMAGE]) {
  if (!existsSync(f)) {
    console.error(`Missing: ${f}`);
    console.error('Run: bash scripts/download-test-models.sh --vision');
    process.exit(1);
  }
}

console.log('=== Vision Stress Test ===\n');

const imageData = readFileSync(TEST_IMAGE);
let model: LlmModel;

// Load model once for most tests
model = await loadModel(MODEL_PATH, {
  type: 'llm',
  contextSize: 4096,
  mmprojPath: MMPROJ_PATH,
}) as LlmModel;

// ─── 1. Rapid sequential inferences ───

await run('Rapid sequential: 5 vision inferences back-to-back', async () => {
  for (let i = 0; i < 5; i++) {
    const result = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: `Iteration ${i + 1}: What is this?` },
        ],
      },
    ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });
    if (!result.content) throw new Error(`No content at iteration ${i + 1}`);
    process.stdout.write('.');
  }
});

// ─── 2. Large image ───

await run('Large image: 1024x1024 synthetic RGB', async () => {
  const w = 1024, h = 1024;
  const rgbData = Buffer.alloc(w * h * 3);
  // Create a gradient pattern
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 3;
      rgbData[idx] = (x * 255 / w) | 0;
      rgbData[idx + 1] = (y * 255 / h) | 0;
      rgbData[idx + 2] = 128;
    }
  }

  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', rgbData, width: w, height: h },
        { type: 'text', text: 'What do you see?' },
      ],
    },
  ], { temperature: 0, maxTokens: 30, thinkingBudget: 0 });
  if (!result.content) throw new Error('No content for large image');
});

// ─── 3. Tiny image (1x1) ───

await run('Tiny image: 1x1 pixel RGB (should error, not crash)', async () => {
  const rgbData = Buffer.from([255, 0, 0]); // single red pixel
  try {
    await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', rgbData, width: 1, height: 1 },
          { type: 'text', text: 'What color?' },
        ],
      },
    ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });
    throw new Error('Should have thrown for tiny image');
  } catch (err) {
    const msg = (err as Error).message;
    if (msg.includes('too small') || msg.includes('preprocessing') || msg.includes('failed') || msg.includes('14x14')) {
      // Clean error — expected
    } else {
      throw err; // Unexpected error
    }
  }
});

// ─── 4. Alternating text and image inferences ───

await run('Alternating: text-only then image, repeated 3x', async () => {
  for (let i = 0; i < 3; i++) {
    // Text-only
    const r1 = await model.complete([
      { role: 'user', content: `What is ${i + 1} + ${i + 1}?` },
    ], { temperature: 0, maxTokens: 10, thinkingBudget: 0 });
    if (!r1.content) throw new Error(`No text content at ${i}`);

    // Image
    const r2 = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: 'Brief description.' },
        ],
      },
    ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });
    if (!r2.content) throw new Error(`No image content at ${i}`);
  }
});

// ─── 5. Streaming + abort ───

await run('Stream abort mid-generation', async () => {
  const controller = new AbortController();
  let chunks = 0;

  for await (const chunk of model.stream([
    {
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: 'Write a very long detailed description of everything you see.' },
      ],
    },
  ], { temperature: 0, maxTokens: 200, thinkingBudget: 0, signal: controller.signal })) {
    chunks++;
    if (chunks >= 3) {
      controller.abort();
      break;
    }
  }

  if (chunks < 2) throw new Error('Expected at least a couple chunks before abort');
});

// ─── 6. Rapid stream+complete interleave ───

await run('Stream then complete, repeated 3x', async () => {
  for (let i = 0; i < 3; i++) {
    // Stream
    let streamContent = '';
    for await (const chunk of model.stream([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: 'One word description.' },
        ],
      },
    ], { temperature: 0, maxTokens: 10, thinkingBudget: 0 })) {
      if (chunk.content) streamContent += chunk.content;
    }
    if (!streamContent) throw new Error(`Stream ${i} empty`);

    // Complete
    const result = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: 'What is this?' },
        ],
      },
    ], { temperature: 0, maxTokens: 10, thinkingBudget: 0 });
    if (!result.content) throw new Error(`Complete ${i} empty`);
  }
});

// ─── 7. Memory test: unload and reload ───

await run('Load/unload cycle 3x', async () => {
  await model.unload();

  for (let i = 0; i < 3; i++) {
    model = await loadModel(MODEL_PATH, {
      type: 'llm',
      contextSize: 4096,
      mmprojPath: MMPROJ_PATH,
    }) as LlmModel;

    const result = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: 'What?' },
        ],
      },
    ], { temperature: 0, maxTokens: 10, thinkingBudget: 0 });
    if (!result.content) throw new Error(`Cycle ${i} no content`);

    await model.unload();
    process.stdout.write('.');
  }

  // Reload for remaining tests
  model = await loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 4096,
    mmprojPath: MMPROJ_PATH,
  }) as LlmModel;
});

// ─── 8. Long conversation with multiple image turns ───

await run('Multi-turn: 3 turns with images', async () => {
  const messages: ChatMessage[] = [];

  for (let turn = 0; turn < 3; turn++) {
    messages.push({
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: `Turn ${turn + 1}: describe this.` },
      ],
    });

    const result = await model.complete(messages, {
      temperature: 0,
      maxTokens: 30,
      thinkingBudget: 0,
    });
    if (!result.content) throw new Error(`Turn ${turn + 1} empty`);

    messages.push({ role: 'assistant', content: result.content });
    process.stdout.write('.');
  }
});

// ─── 9. Concurrent promise creation (should serialize, not crash) ───

await run('Concurrent: 3 complete() calls queue properly', async () => {
  const promises = [];
  for (let i = 0; i < 3; i++) {
    promises.push(
      model.complete([
        {
          role: 'user',
          content: [
            { type: 'image', data: imageData },
            { type: 'text', text: `Query ${i}` },
          ],
        },
      ], { temperature: 0, maxTokens: 10, thinkingBudget: 0 })
    );
  }

  const results = await Promise.all(promises);
  for (let i = 0; i < results.length; i++) {
    if (!results[i].content) throw new Error(`Concurrent ${i} empty`);
  }
});

// ─── Cleanup ───

await model.unload();

// ─── Summary ───

console.log('\n\n' + '='.repeat(60));
console.log(`Stress Test: ${passed} passed, ${failed} failed`);
console.log('='.repeat(60));

if (failed > 0) process.exit(1);
