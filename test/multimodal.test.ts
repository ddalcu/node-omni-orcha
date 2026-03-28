import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { loadModel, createModel } from '../src/index.ts';
import type { LlmModel, ChatMessage, ContentPart, ImageContentPart } from '../src/types.ts';
import { saveTestOutput } from './test-output-helper.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || path.join(process.env['HOME'] || process.env['USERPROFILE'] || '', '.orcha', 'workspace', '.models');
const TEXT_MODEL_PATH = path.join(MODELS_DIR, 'tinyllama', 'tinyllama.gguf');
const hasTextModel = existsSync(TEXT_MODEL_PATH);

// Vision model paths — these are optional, tests that need them are skipped
const VISION_DIR = path.join(MODELS_DIR, 'qwen2-vl-2b');
const VISION_MODEL_PATH = path.join(VISION_DIR, 'Qwen2-VL-2B-Instruct-Q4_K_M.gguf');
const MMPROJ_PATH = path.join(VISION_DIR, 'mmproj-Qwen2-VL-2B-Instruct-f16.gguf');
const hasVisionModel = existsSync(VISION_MODEL_PATH) && existsSync(MMPROJ_PATH);

// ─── Unit tests (no model needed) ───

describe('Multimodal Types', () => {
  it('ChatMessage accepts string content', () => {
    const msg: ChatMessage = {
      role: 'user',
      content: 'Hello world',
    };
    assert.equal(msg.content, 'Hello world');
  });

  it('ChatMessage accepts ContentPart array', () => {
    const msg: ChatMessage = {
      role: 'user',
      content: [
        { type: 'text', text: 'What is in this image?' },
        { type: 'image', data: Buffer.from([0xFF, 0xD8]) },
      ],
    };
    assert.ok(Array.isArray(msg.content));
    assert.equal(msg.content.length, 2);
    assert.equal(msg.content[0].type, 'text');
    assert.equal(msg.content[1].type, 'image');
  });

  it('ImageContentPart supports data, path, and rgbData variants', () => {
    const fromData: ImageContentPart = {
      type: 'image',
      data: Buffer.from([0xFF, 0xD8, 0xFF, 0xE0]),
    };
    const fromPath: ImageContentPart = {
      type: 'image',
      path: '/tmp/test.png',
    };
    const fromRgb: ImageContentPart = {
      type: 'image',
      rgbData: Buffer.alloc(3 * 64 * 64),
      width: 64,
      height: 64,
    };
    assert.ok(fromData.data);
    assert.ok(fromPath.path);
    assert.ok(fromRgb.rgbData);
    assert.equal(fromRgb.width, 64);
  });

  it('ContentPart array can mix text and multiple images', () => {
    const parts: ContentPart[] = [
      { type: 'text', text: 'Compare these two images:' },
      { type: 'image', data: Buffer.from([0x89, 0x50, 0x4E, 0x47]) }, // PNG magic
      { type: 'image', data: Buffer.from([0xFF, 0xD8, 0xFF, 0xE0]) }, // JPEG magic
      { type: 'text', text: 'Which one is better?' },
    ];
    const imageCount = parts.filter(p => p.type === 'image').length;
    const textCount = parts.filter(p => p.type === 'text').length;
    assert.equal(imageCount, 2);
    assert.equal(textCount, 2);
  });
});

// ─── Text model tests: verify multimodal messages work with text-only model ───

describe('Multimodal with text-only model', {
  skip: !hasTextModel ? `No test model at ${TEXT_MODEL_PATH}` : undefined,
}, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(TEXT_MODEL_PATH, { type: 'llm', contextSize: 2048 }) as LlmModel;
  });

  after(async () => {
    await model?.unload();
  });

  it('hasVision is false when no mmproj loaded', () => {
    assert.equal(model.hasVision, false);
  });

  it('still handles string content normally', async () => {
    const result = await model.complete([
      { role: 'user', content: 'Say hi' },
    ], { temperature: 0, maxTokens: 10 });
    assert.ok(result.content.length > 0);
  });

  it('handles ContentPart array with text-only parts', async () => {
    // Even without vision, text-only content parts should work
    const result = await model.complete([
      {
        role: 'user',
        content: [{ type: 'text', text: 'Say hi' }],
      },
    ], { temperature: 0, maxTokens: 10 });
    assert.ok(result.content.length > 0);
  });
});

// ─── Vision model tests (requires downloading vision model) ───

describe('Multimodal Vision Model', {
  skip: !hasVisionModel ? `No vision model at ${VISION_MODEL_PATH}` : undefined,
}, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(VISION_MODEL_PATH, {
      type: 'llm',
      contextSize: 4096,
      mmprojPath: MMPROJ_PATH,
    }) as LlmModel;
  });

  after(async () => {
    await model?.unload();
  });

  it('hasVision is true when mmproj loaded', () => {
    assert.equal(model.hasVision, true);
  });

  it('describes an image from buffer', async () => {
    // Create a simple 2x2 red PNG for testing
    const testImagePath = path.join(VISION_DIR, 'test-image.png');
    if (!existsSync(testImagePath)) {
      console.log(`    Skipping: no test image at ${testImagePath}`);
      return;
    }

    const imageData = readFileSync(testImagePath);
    const result = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: 'Describe this image briefly.' },
        ],
      },
    ], { temperature: 0, maxTokens: 100, thinkingBudget: 0 });

    assert.ok(result.content.length > 0, 'Should produce a description');
    assert.ok(result.usage.inputTokens > 100, 'Image tokens should inflate input count');

    const text = `Vision result:\n${result.content}\n\nUsage: ${JSON.stringify(result.usage)}`;
    const outPath = saveTestOutput('multimodal', 'vision-describe', {}, text, '.txt');
    console.log(`    Saved: ${outPath}`);
  });

  it('describes an image from file path', async () => {
    const testImagePath = path.join(VISION_DIR, 'test-image.png');
    if (!existsSync(testImagePath)) {
      console.log(`    Skipping: no test image at ${testImagePath}`);
      return;
    }

    const result = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'image', path: testImagePath },
          { type: 'text', text: 'What do you see?' },
        ],
      },
    ], { temperature: 0, maxTokens: 100, thinkingBudget: 0 });

    assert.ok(result.content.length > 0);
  });

  it('handles multiple images in one message', async () => {
    const testImagePath = path.join(VISION_DIR, 'test-image.png');
    if (!existsSync(testImagePath)) {
      console.log(`    Skipping: no test image at ${testImagePath}`);
      return;
    }

    const imageData = readFileSync(testImagePath);
    const result = await model.complete([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Compare these two images:' },
          { type: 'image', data: imageData },
          { type: 'image', data: imageData },
          { type: 'text', text: 'Are they the same?' },
        ],
      },
    ], { temperature: 0, maxTokens: 50, thinkingBudget: 0 });

    assert.ok(result.content.length > 0);
  });

  it('streams vision responses', async () => {
    const testImagePath = path.join(VISION_DIR, 'test-image.png');
    if (!existsSync(testImagePath)) {
      console.log(`    Skipping: no test image at ${testImagePath}`);
      return;
    }

    const imageData = readFileSync(testImagePath);
    let content = '';
    let chunkCount = 0;

    for await (const chunk of model.stream([
      {
        role: 'user',
        content: [
          { type: 'image', data: imageData },
          { type: 'text', text: 'Describe this image in one sentence.' },
        ],
      },
    ], { temperature: 0, maxTokens: 50, thinkingBudget: 0 })) {
      if (chunk.content) content += chunk.content;
      chunkCount++;
    }

    assert.ok(content.length > 0, 'Stream should produce content');
    assert.ok(chunkCount > 1, 'Should have multiple chunks');
  });
});

// ─── Error handling tests ───

describe('Multimodal Error Handling', {
  skip: !hasTextModel ? `No test model at ${TEXT_MODEL_PATH}` : undefined,
}, () => {
  it('rejects loading with invalid mmproj path', async () => {
    await assert.rejects(
      () => loadModel(TEXT_MODEL_PATH, {
        type: 'llm',
        contextSize: 2048,
        mmprojPath: '/nonexistent/mmproj.gguf',
      }),
      /Failed to load multimodal projector/,
    );
  });
});
