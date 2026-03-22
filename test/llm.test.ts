import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { loadModel, createModel } from '../src/index.ts';
import type { LlmModel, ToolDefinition } from '../src/types.ts';

const TEST_MODEL_PATH = new URL('./fixtures/tinyllama.gguf', import.meta.url).pathname;
const hasModel = existsSync(TEST_MODEL_PATH);

describe('LlmModel', { skip: !hasModel ? 'No test model at test/fixtures/tinyllama.gguf — run scripts/download-test-models.sh' : undefined }, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(TEST_MODEL_PATH, { type: 'llm', contextSize: 2048 }) as LlmModel;
  });

  after(async () => {
    await model?.unload();
  });

  it('loads and reports as loaded', () => {
    assert.equal(model.type, 'llm');
    assert.equal(model.loaded, true);
    assert.ok(model.metadata, 'Should have metadata');
    assert.ok(model.metadata!.contextLength > 0, 'Should have context length');
  });

  it('completes a simple prompt', async () => {
    const result = await model.complete([
      { role: 'user', content: 'Say hello in one word.' },
    ], { temperature: 0, maxTokens: 32 });

    assert.ok(result.content.length > 0, 'Should have content');
    assert.ok(result.usage.inputTokens > 0, 'Should have input tokens');
    assert.ok(result.usage.outputTokens > 0, 'Should have output tokens');
    assert.equal(result.usage.totalTokens, result.usage.inputTokens + result.usage.outputTokens);
  });

  it('respects maxTokens', async () => {
    const result = await model.complete([
      { role: 'user', content: 'Count from 1 to 100.' },
    ], { temperature: 0, maxTokens: 5 });

    assert.ok(result.usage.outputTokens <= 10, 'Should stop around maxTokens');
  });

  it('handles system messages', async () => {
    const result = await model.complete([
      { role: 'system', content: 'You are a helpful assistant. Respond in exactly one word.' },
      { role: 'user', content: 'Hello!' },
    ], { temperature: 0, maxTokens: 16 });

    assert.ok(result.content.length > 0, 'Should have content');
  });
});

describe('createModel', { skip: !hasModel ? 'No test model' : undefined }, () => {
  it('creates unloaded model', () => {
    const model = createModel(TEST_MODEL_PATH, 'llm');
    assert.equal(model.loaded, false);
    assert.equal(model.type, 'llm');
  });

  it('can be loaded manually', async () => {
    const model = createModel(TEST_MODEL_PATH, 'llm');
    assert.equal(model.loaded, false);

    await model.load({ contextSize: 2048 });
    assert.equal(model.loaded, true);

    await model.unload();
    assert.equal(model.loaded, false);
  });
});

// ─── Tool Calling Tests (require a model with tool-calling template) ───
const TOOL_MODEL_PATH = new URL('./fixtures/Qwen3.5-4B-IQ4_NL.gguf', import.meta.url).pathname;
const hasToolModel = existsSync(TOOL_MODEL_PATH);

const weatherTool: ToolDefinition = {
  name: 'get_weather',
  description: 'Get the current weather for a city',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string', description: 'City name' },
    },
    required: ['city'],
  },
};

describe('LlmModel tool calling', { skip: !hasToolModel ? 'No Qwen3.5-4B model for tool calling tests' : undefined }, () => {
  let model: LlmModel;

  before(async () => {
    model = await loadModel(TOOL_MODEL_PATH, { type: 'llm', contextSize: 2048 }) as LlmModel;
  });

  after(async () => {
    await model?.unload();
  });

  it('completes normally without tools (no regression)', async () => {
    const result = await model.complete([
      { role: 'user', content: 'Say hello in one word.' },
    ], { temperature: 0, maxTokens: 128 });

    assert.ok(result.content.length > 0 || (result.reasoning && result.reasoning.length > 0),
      'Should have content or reasoning');
    assert.equal(result.toolCalls, undefined, 'Should not have tool calls');
  });

  it('returns tool calls when tools are provided', async () => {
    const result = await model.complete([
      { role: 'user', content: 'What is the weather in Tokyo?' },
    ], {
      temperature: 0,
      maxTokens: 256,
      tools: [weatherTool],
      toolChoice: 'required',
    });

    assert.ok(result.toolCalls, 'Should have toolCalls');
    assert.ok(result.toolCalls!.length > 0, 'Should have at least one tool call');

    const tc = result.toolCalls![0]!;
    assert.equal(tc.name, 'get_weather', 'Should call get_weather');
    assert.ok(tc.args, 'Should have args');

    // Args should be valid JSON
    const args = JSON.parse(tc.args);
    assert.ok(args.city, 'Should have city argument');
    console.log(`    Tool call: ${tc.name}(${tc.args})`);
  });

  it('toolChoice none produces plain text', async () => {
    const result = await model.complete([
      { role: 'user', content: 'What is the weather in Paris?' },
    ], {
      temperature: 0,
      maxTokens: 256,
      tools: [weatherTool],
      toolChoice: 'none',
    });

    assert.ok(result.content.length > 0 || (result.reasoning && result.reasoning.length > 0),
      'Should have text content or reasoning');
    assert.equal(result.toolCalls, undefined, 'Should not have tool calls with toolChoice none');
  });
});
