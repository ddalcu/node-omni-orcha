/**
 * Vision crash hardening test — intentionally tries to crash the multimodal pipeline.
 *
 * Tests invalid/hostile inputs that should produce clean errors, not segfaults.
 *
 * Usage:
 *   bash scripts/download-test-models.sh --vision
 *   node scripts/vision-crash-test.ts
 */

import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { loadModel, createModel } from '../src/index.ts';
import type { LlmModel, ChatMessage } from '../src/types.ts';

const MODELS = process.env['MODELS_DIR'] || `${process.env['HOME']}/.orcha/workspace/.models`;
const VISION_DIR = path.join(MODELS, 'qwen2-vl-2b');
const MODEL_PATH = path.join(VISION_DIR, 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf');
const MMPROJ_PATH = path.join(VISION_DIR, 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf');
const TEXT_MODEL_PATH = path.join(MODELS, 'tinyllama', 'tinyllama.gguf');
const TEST_IMAGE = path.join(VISION_DIR, 'test-image.png');

let passed = 0;
let failed = 0;

async function expectError(name: string, fn: () => Promise<unknown>, expectedPattern?: RegExp) {
  process.stdout.write(`[CRASH] ${name} ... `);
  try {
    await fn();
    console.log('FAILED (should have thrown)');
    failed++;
  } catch (err) {
    const msg = (err as Error).message;
    if (expectedPattern && !expectedPattern.test(msg)) {
      console.log(`FAILED (wrong error: ${msg})`);
      failed++;
    } else {
      console.log(`PASSED (caught: ${msg.substring(0, 80)})`);
      passed++;
    }
  }
}

async function expectSuccess(name: string, fn: () => Promise<unknown>) {
  process.stdout.write(`[CRASH] ${name} ... `);
  try {
    await fn();
    console.log('PASSED');
    passed++;
  } catch (err) {
    console.log(`FAILED: ${(err as Error).message}`);
    failed++;
  }
}

// Check prerequisites
for (const f of [MODEL_PATH, MMPROJ_PATH]) {
  if (!existsSync(f)) {
    console.error(`Missing: ${f}`);
    console.error('Run: bash scripts/download-test-models.sh --vision');
    process.exit(1);
  }
}

console.log('=== Vision Crash Hardening Test ===\n');
console.log('Each test should produce a clean error or handle gracefully.\n');

// ─── 1. Invalid mmproj path ───

await expectError(
  'Invalid mmproj path',
  () => loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 2048,
    mmprojPath: '/nonexistent/mmproj.gguf',
  }),
  /Failed to load multimodal projector/,
);

// ─── 2. Empty mmproj path (should work, no vision) ───

await expectSuccess('Empty mmproj path = text-only model', async () => {
  const m = await loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 2048,
  }) as LlmModel;
  if (m.hasVision) throw new Error('Expected no vision without mmproj');
  await m.unload();
});

// ─── 3. Load vision model for remaining crash tests ───

let model: LlmModel;
await expectSuccess('Load vision model', async () => {
  model = await loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 4096,
    mmprojPath: MMPROJ_PATH,
  }) as LlmModel;
});

// ─── 4. Invalid image data ───

await expectError(
  'Invalid image data: random bytes',
  () => model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: Buffer.from([0xDE, 0xAD, 0xBE, 0xEF]) },
        { type: 'text', text: 'What?' },
      ],
    },
  ], { maxTokens: 10, thinkingBudget: 0 }),
);

// ─── 5. Zero-length image buffer ───

await expectError(
  'Zero-length image buffer',
  () => model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: Buffer.alloc(0) },
        { type: 'text', text: 'What?' },
      ],
    },
  ], { maxTokens: 10, thinkingBudget: 0 }),
);

// ─── 6. Missing image file ───

await expectError(
  'Image path: nonexistent file',
  () => model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', path: '/nonexistent/image.png' },
        { type: 'text', text: 'What?' },
      ],
    },
  ], { maxTokens: 10, thinkingBudget: 0 }),
  /Failed to open image file/,
);

// ─── 7. Image marker mismatch (images in content but text has no markers) ───
// This tests internal consistency — the C++ binding inserts markers automatically,
// so this shouldn't happen normally. But if someone passes images without text
// that would have markers, it should still handle gracefully.

// ─── 8. Very large number of images (should fail gracefully with context overflow) ───

await expectSuccess('Many images: 20 images in one prompt (should not crash)', async () => {
  const img = existsSync(TEST_IMAGE) ? readFileSync(TEST_IMAGE) : Buffer.alloc(100);
  const parts: any[] = [{ type: 'text', text: 'Describe all these:' }];
  for (let i = 0; i < 20; i++) {
    parts.push({ type: 'image', data: img });
  }
  parts.push({ type: 'text', text: 'Go.' });
  try {
    await model.complete([
      { role: 'user', content: parts },
    ], { maxTokens: 10, thinkingBudget: 0 });
    // May succeed or fail with context overflow — both are fine, as long as no crash
  } catch {
    // Clean error is fine
  }
});

// ─── 9. RGB data with wrong size (width * height * 3 != data.length) ───

await expectError(
  'RGB data size mismatch: 10x10 but only 9 bytes',
  () => model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', rgbData: Buffer.alloc(9), width: 10, height: 10 },
        { type: 'text', text: 'What?' },
      ],
    },
  ], { maxTokens: 10, thinkingBudget: 0 }),
);

// ─── 10. Use after unload ───

await expectSuccess('Use after unload: should throw cleanly', async () => {
  const m = await loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 2048,
    mmprojPath: MMPROJ_PATH,
  }) as LlmModel;
  await m.unload();

  try {
    await m.complete([{ role: 'user', content: 'hello' }], { maxTokens: 10 });
    throw new Error('Should have thrown');
  } catch (err) {
    if (!(err as Error).message.includes('not loaded')) {
      throw new Error(`Wrong error: ${(err as Error).message}`);
    }
  }
});

// ─── 11. Double unload ───

await expectSuccess('Double unload: should not crash', async () => {
  const m = await loadModel(MODEL_PATH, {
    type: 'llm',
    contextSize: 2048,
    mmprojPath: MMPROJ_PATH,
  }) as LlmModel;
  await m.unload();
  await m.unload(); // Second unload should be no-op
});

// ─── 12. Empty messages array with images ───

await expectSuccess('Empty messages array (should not crash)', async () => {
  try {
    await model.complete([], { maxTokens: 10, thinkingBudget: 0 });
    // May succeed with empty content or throw — both fine
  } catch {
    // Clean error is fine
  }
});

// ─── 13. Only image, no text ───

await expectSuccess('Image-only content (no text parts)', async () => {
  if (!existsSync(TEST_IMAGE)) return;
  const img = readFileSync(TEST_IMAGE);
  // Some models handle this, some don't — should not crash either way
  try {
    const result = await model.complete([
      {
        role: 'user',
        content: [{ type: 'image', data: img }],
      },
    ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });
    // If it succeeds, that's fine
  } catch {
    // If it fails with a clean error, that's also fine
  }
});

// ─── 14. maxTokens = 0 ───

await expectSuccess('maxTokens=0 should produce empty/short response', async () => {
  if (!existsSync(TEST_IMAGE)) return;
  const img = readFileSync(TEST_IMAGE);
  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: img },
        { type: 'text', text: 'Describe.' },
      ],
    },
  ], { temperature: 0, maxTokens: 0, thinkingBudget: 0 });
  // Should not crash, may have empty content
});

// ─── 15. Cross-platform path separators ───

await expectSuccess('Path with forward slashes (cross-platform)', async () => {
  if (!existsSync(TEST_IMAGE)) return;
  // Force forward slashes even on Windows
  const fwdSlashPath = TEST_IMAGE.replace(/\\/g, '/');
  const result = await model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', path: fwdSlashPath },
        { type: 'text', text: 'What is this?' },
      ],
    },
  ], { temperature: 0, maxTokens: 20, thinkingBudget: 0 });
  if (!result.content) throw new Error('No content');
});

// ─── 16. Truncated JPEG (partial file) ───

await expectError(
  'Truncated JPEG: first 100 bytes of a JPEG header',
  async () => {
    // JPEG magic bytes + partial data
    const truncated = Buffer.alloc(100);
    truncated[0] = 0xFF;
    truncated[1] = 0xD8;
    truncated[2] = 0xFF;
    truncated[3] = 0xE0;
    await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', data: truncated },
          { type: 'text', text: 'What?' },
        ],
      },
    ], { maxTokens: 10, thinkingBudget: 0 });
  },
);

// ─── 17. Non-image file (text file as "image") ───

await expectError(
  'Non-image file: plain text as image data',
  () => model.complete([
    {
      role: 'user',
      content: [
        { type: 'image', data: Buffer.from('This is not an image\n'.repeat(100)) },
        { type: 'text', text: 'What?' },
      ],
    },
  ], { maxTokens: 10, thinkingBudget: 0 }),
);

// ─── Cleanup ───

await model.unload();

// ─── Summary ───

console.log('\n' + '='.repeat(60));
console.log(`Crash Hardening: ${passed} passed, ${failed} failed`);
console.log('='.repeat(60));
console.log(`\nIf we got here without a segfault, the error handling is solid.`);

if (failed > 0) process.exit(1);
