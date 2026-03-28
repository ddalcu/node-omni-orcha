/**
 * Vision/Multimodal integration test — exercises the mtmd pipeline with real models.
 *
 * Usage:
 *   bash scripts/download-test-models.sh --vision  # download models first
 *   node scripts/vision-integration-test.ts
 */

import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { loadModel } from '../src/index.ts';
import { saveTestOutput } from '../test/test-output-helper.ts';
import type { LlmModel, ChatMessage } from '../src/types.ts';

const MODELS = process.env['MODELS_DIR'] || `${process.env['HOME']}/.orcha/workspace/.models`;
const VISION_DIR = path.join(MODELS, 'qwen2-vl-2b');
const MODEL_PATH = path.join(VISION_DIR, 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf');
const MMPROJ_PATH = path.join(VISION_DIR, 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf');
const TEST_IMAGE = path.join(VISION_DIR, 'test-image.png');

let passed = 0;
let failed = 0;

async function run(name: string, fn: () => Promise<void>) {
  process.stdout.write(`\n[${'='.repeat(60)}]\n${name}\n[${'='.repeat(60)}]\n`);
  const start = Date.now();
  try {
    await fn();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log(`PASSED (${elapsed}s)\n`);
    passed++;
  } catch (err) {
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.error(`FAILED (${elapsed}s): ${(err as Error).stack}\n`);
    failed++;
  }
}

// Check prerequisites
if (!existsSync(MODEL_PATH)) {
  console.error(`Vision model not found: ${MODEL_PATH}`);
  console.error('Run: bash scripts/download-test-models.sh --vision');
  process.exit(1);
}
if (!existsSync(MMPROJ_PATH)) {
  console.error(`mmproj not found: ${MMPROJ_PATH}`);
  process.exit(1);
}
if (!existsSync(TEST_IMAGE)) {
  console.error(`Test image not found: ${TEST_IMAGE}`);
  process.exit(1);
}

console.log('=== Vision/Multimodal Integration Test ===');
console.log(`Model:   ${MODEL_PATH}`);
console.log(`mmproj:  ${MMPROJ_PATH}`);
console.log(`Image:   ${TEST_IMAGE}\n`);

let model: LlmModel;

// ─── Load ───

await run('Load vision model with mmproj', async () => {
  model = await loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 4096,
    mmprojPath: MMPROJ_PATH,
  }) as LlmModel;
  console.log(`  loaded: ${model.loaded}`);
  console.log(`  hasVision: ${model.hasVision}`);
  if (!model.hasVision) throw new Error('Expected hasVision=true');
});

// ─── Basic image description ───

await run('Describe image from Buffer', async () => {
  const imageData = readFileSync(TEST_IMAGE);
  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: 'Describe what you see in this image. Be brief.' },
      ],
    },
  ], { temperature: 0, maxTokens: 150, thinkingBudget: 0 });

  console.log(`  content: ${result.content}`);
  console.log(`  usage: ${JSON.stringify(result.usage)}`);
  if (!result.content) throw new Error('No content in response');
  if (result.usage.inputTokens < 50) throw new Error('Expected more input tokens (image should add many)');

  saveTestOutput('vision', 'describe-buffer', {}, result.content, '.txt');
});

// ─── Image from file path ───

await run('Describe image from file path', async () => {
  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', path: TEST_IMAGE },
        { type: 'text', text: 'What colors do you see?' },
      ],
    },
  ], { temperature: 0, maxTokens: 100, thinkingBudget: 0 });

  console.log(`  content: ${result.content}`);
  if (!result.content) throw new Error('No content');
});

// ─── Text-only with vision model (backward compat) ───

await run('Text-only message on vision model', async () => {
  const result = await model.complete([
    { role: 'user', content: 'What is 2+2? Answer with just the number.' },
  ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });

  console.log(`  content: ${result.content}`);
  if (!result.content) throw new Error('No content');
});

// ─── Text ContentPart array (no images) ───

await run('Text-only ContentPart array on vision model', async () => {
  const result = await model.complete([
    {
      role: 'user',
      content: [{ type: 'text', text: 'Say hello.' }],
    },
  ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });

  console.log(`  content: ${result.content}`);
  if (!result.content) throw new Error('No content');
});

// ─── Streaming with images ───

await run('Stream image description', async () => {
  const imageData = readFileSync(TEST_IMAGE);
  let content = '';
  let chunks = 0;
  let finalUsage: any = null;

  for await (const chunk of model.stream([
    {
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: 'Describe this image in one sentence.' },
      ],
    },
  ], { temperature: 0, maxTokens: 80, thinkingBudget: 0 })) {
    if (chunk.content) content += chunk.content;
    if (chunk.usage) finalUsage = chunk.usage;
    chunks++;
  }

  console.log(`  content: ${content}`);
  console.log(`  chunks: ${chunks}`);
  console.log(`  usage: ${JSON.stringify(finalUsage)}`);
  if (!content) throw new Error('No streamed content');
  if (chunks < 2) throw new Error('Expected multiple stream chunks');
});

// ─── Multi-turn conversation with images ───

await run('Multi-turn conversation with image', async () => {
  const imageData = readFileSync(TEST_IMAGE);

  // Turn 1: describe
  const result1 = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: 'What is in this image?' },
      ],
    },
  ], { temperature: 0, maxTokens: 100, thinkingBudget: 0 });

  console.log(`  turn1: ${result1.content.substring(0, 80)}...`);

  // Turn 2: follow-up (text only, referencing the previous image)
  const result2 = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: 'What is in this image?' },
      ],
    },
    { role: 'assistant', content: result1.content },
    { role: 'user', content: 'Can you be more specific about the colors?' },
  ], { temperature: 0, maxTokens: 100, thinkingBudget: 0 });

  console.log(`  turn2: ${result2.content.substring(0, 80)}...`);
  if (!result2.content) throw new Error('No turn 2 content');
});

// ─── Multiple images ───

await run('Multiple images in one message', async () => {
  const imageData = readFileSync(TEST_IMAGE);

  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'text', text: 'I am showing you two copies of the same image.' },
        { type: 'image', data: imageData },
        { type: 'image', data: imageData },
        { type: 'text', text: 'Are these two images identical? Answer yes or no.' },
      ],
    },
  ], { temperature: 0, maxTokens: 50, thinkingBudget: 0 });

  console.log(`  content: ${result.content}`);
  if (!result.content) throw new Error('No content');
});

// ─── Raw RGB pixels ───

await run('Image from raw RGB data', async () => {
  // Create a 64x64 solid red image (RGB) — minimum 14x14 for vision models
  const width = 64;
  const height = 64;
  const rgbData = Buffer.alloc(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    rgbData[i * 3] = 255;     // R
    rgbData[i * 3 + 1] = 0;   // G
    rgbData[i * 3 + 2] = 0;   // B
  }

  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', rgbData, width, height },
        { type: 'text', text: 'What color is this image?' },
      ],
    },
  ], { temperature: 0, maxTokens: 50, thinkingBudget: 0 });

  console.log(`  content: ${result.content}`);
  if (!result.content) throw new Error('No content');
});

// ─── Cleanup ───

await model.unload();

// ─── Summary ───

console.log('\n' + '='.repeat(60));
console.log(`Vision Integration: ${passed} passed, ${failed} failed`);
console.log('='.repeat(60));

if (failed > 0) process.exit(1);
