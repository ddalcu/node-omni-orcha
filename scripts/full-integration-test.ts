/**
 * Full integration test — exercises every engine with real models.
 * Outputs everything to test-output/ with datetime and model params in filenames.
 *
 * Usage: node scripts/full-integration-test.ts
 */

import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { createModel, loadModel } from '../src/index.ts';
import { saveTestOutput, saveVideoFrames } from '../test/test-output-helper.ts';
import type { LlmModel, ImageModel, TtsModel } from '../src/types.ts';

const MODELS = process.env['MODELS_DIR'] || `${process.env['HOME']}/.orcha/workspace/.models`;
const FIXTURES = path.resolve(import.meta.dirname!, '..', 'test', 'fixtures');

let passed = 0;
let failed = 0;
let skipped = 0;

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
    console.error(`FAILED (${elapsed}s): ${(err as Error).message}\n`);
    failed++;
  }
}

function skip(name: string, reason: string) {
  console.log(`\nSKIP: ${name} — ${reason}`);
  skipped++;
}

// ─── LLM: Qwen3.5-4B ───

const llmPath = path.join(MODELS, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf');

if (existsSync(llmPath)) {
  await run('LLM — Qwen3.5-4B completion', async () => {
    const model = await loadModel(llmPath, { type: 'llm', contextSize: 2048 }) as LlmModel;

    const opts = { temperature: 0, maxTokens: 64 };
    const result = await model.complete([
      { role: 'user', content: 'What is 2+2? Reply with just the number.' },
    ], opts);

    console.log(`  Content: ${JSON.stringify(result.content.substring(0, 200))}`);
    console.log(`  Reasoning: ${result.reasoning ? result.reasoning.substring(0, 100) + '...' : 'none'}`);
    console.log(`  Usage: ${JSON.stringify(result.usage)}`);

    const text = `Prompt: What is 2+2?\nContent: ${result.content}\nReasoning: ${result.reasoning ?? ''}\nUsage: ${JSON.stringify(result.usage)}`;
    const out = saveTestOutput('llm', 'Qwen3.5-4B-IQ4_NL', opts, text, '.txt');
    console.log(`  Output: ${out}`);

    await model.unload();
  });

  await run('LLM — Qwen3.5-4B streaming', async () => {
    const model = await loadModel(llmPath, { type: 'llm', contextSize: 2048 }) as LlmModel;

    const opts = { temperature: 0.7, maxTokens: 128 };
    let content = '';
    let chunks = 0;
    for await (const chunk of model.stream([
      { role: 'user', content: 'Count to 5 briefly.' },
    ], opts)) {
      if (chunk.content && !chunk.done) {
        content += chunk.content;
        chunks++;
      }
      if (chunk.done) break;
    }

    console.log(`  Streamed ${chunks} chunks, total: ${content.length} chars`);
    console.log(`  Content: ${JSON.stringify(content.substring(0, 200))}`);

    const text = `Prompt: Count to 5 briefly.\nStreamed chunks: ${chunks}\nContent: ${content}`;
    const out = saveTestOutput('llm', 'Qwen3.5-4B-IQ4_NL', { ...opts, mode: 'stream' }, text, '.txt');
    console.log(`  Output: ${out}`);

    await model.unload();
  });

  await run('LLM — Qwen3.5-4B tool calling', async () => {
    const model = await loadModel(llmPath, { type: 'llm', contextSize: 2048 }) as LlmModel;

    const opts = {
      temperature: 0,
      maxTokens: 256,
      tools: [{
        name: 'get_weather',
        description: 'Get current weather for a city',
        parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
      }],
      toolChoice: 'required' as const,
    };

    const result = await model.complete([
      { role: 'user', content: 'What is the weather in Tokyo?' },
    ], opts);

    console.log(`  Tool calls: ${JSON.stringify(result.toolCalls)}`);
    console.log(`  Content: ${result.content.substring(0, 100)}`);

    const text = `Prompt: What is the weather in Tokyo?\nTool calls: ${JSON.stringify(result.toolCalls, null, 2)}\nContent: ${result.content}\nUsage: ${JSON.stringify(result.usage)}`;
    const out = saveTestOutput('llm', 'Qwen3.5-4B-IQ4_NL', { temperature: 0, maxTokens: 256, tool: 'get_weather' }, text, '.txt');
    console.log(`  Output: ${out}`);

    await model.unload();
  });
} else {
  skip('LLM', `Model not found: ${llmPath}`);
}

// ─── TTS: Qwen3-TTS ───

const ttsDir = path.join(MODELS, 'qwen3-tts');
const samuelWav = path.join(FIXTURES, 'Samuel.wav');

if (existsSync(path.join(ttsDir, 'qwen3-tts-0.6b-f16.gguf'))) {
  await run('TTS — Qwen3-TTS default voice', async () => {
    const model = createModel(ttsDir, 'tts') as TtsModel;
    await model.load();

    const wav = await model.speak('Hello world, this is a test of Qwen3 text to speech.');
    const durationSec = ((wav.length - 44) / (24000 * 2)).toFixed(2);

    console.log(`  WAV: ${wav.length} bytes, ~${durationSec}s`);

    const out = saveTestOutput('tts', 'qwen3-0.6b-f16', { voice: 'default' }, wav, '.wav');
    console.log(`  Output: ${out}`);

    await model.unload();
  });

  if (existsSync(samuelWav)) {
    await run('TTS — Qwen3-TTS Samuel L. Jackson clone', async () => {
      const model = createModel(ttsDir, 'tts') as TtsModel;
      await model.load();

      const wav = await model.speak(
        'I have had it with these snakes on this plane. Everybody strap in.',
        { referenceAudioPath: samuelWav },
      );
      const durationSec = ((wav.length - 44) / (24000 * 2)).toFixed(2);

      console.log(`  WAV: ${wav.length} bytes, ~${durationSec}s`);

      const out = saveTestOutput('tts', 'qwen3-0.6b-f16', { voice: 'samuel-clone' }, wav, '.wav');
      console.log(`  Output: ${out}`);

      await model.unload();
    });
  } else {
    skip('TTS voice clone', 'No Samuel.wav reference audio');
  }
} else {
  skip('TTS', `Model not found: ${ttsDir}`);
}

// ─── Image: FLUX 2 Klein ───

const fluxDir = path.join(MODELS, 'flux2-klein');
const fluxModel = path.join(fluxDir, 'flux-2-klein-4b-Q4_K_M.gguf');
const fluxLlm = path.join(fluxDir, 'Qwen3-4B-Q4_K_M.gguf');
const fluxVae = path.join(fluxDir, 'flux2-vae.safetensors');

if (existsSync(fluxModel) && existsSync(fluxLlm) && existsSync(fluxVae)) {
  await run('Image — FLUX 2 Klein 4B generation', async () => {
    const model = createModel(fluxModel, 'image') as ImageModel;
    await model.load({ llmPath: fluxLlm, vaePath: fluxVae, keepVaeOnCpu: true });

    const opts = { width: 512, height: 512, steps: 4, cfgScale: 1.0, sampleMethod: 'euler' as const };
    const png = await model.generate('a red sports car parked on a mountain road at sunset', opts);

    console.log(`  PNG: ${png.length} bytes (${(png.length / 1024).toFixed(0)}KB)`);

    const out = saveTestOutput('image', 'flux2-klein-4b-Q4_K_M', opts, png, '.png');
    console.log(`  Output: ${out}`);

    await model.unload();
  });
} else {
  skip('Image (FLUX 2)', `Missing model files in ${fluxDir}`);
}

// ─── Video: WAN 2.2 ───

const wanDir = path.join(MODELS, 'wan22-5b');
const wanModel = path.join(wanDir, 'Wan2.2-TI2V-5B-Q4_K_M.gguf');
const wanVae = path.join(wanDir, 'Wan2.2_VAE.safetensors');
const wanT5 = path.join(wanDir, 'umt5-xxl-encoder-Q8_0.gguf');

if (existsSync(wanModel) && existsSync(wanVae) && existsSync(wanT5)) {
  await run('Video — WAN 2.2 5B (5 frames)', async () => {
    const model = createModel(wanModel, 'image') as ImageModel;
    await model.load({ t5xxlPath: wanT5, vaePath: wanVae, flashAttn: true, vaeDecodeOnly: true });

    const params = { width: 480, height: 288, videoFrames: 5, steps: 6, cfgScale: 6.0, flowShift: 3.0, seed: 42 };
    const frames = await model.generateVideo('a woman walking on a beach', params);

    console.log(`  Frames: ${frames.length}`);
    console.log(`  Sizes: ${frames.map(f => `${(f.length / 1024).toFixed(0)}KB`).join(', ')}`);

    const outDir = saveVideoFrames('Wan2.2-TI2V-5B-Q4_K_M', params, frames);
    console.log(`  Output: ${outDir}`);

    await model.unload();
  });
} else {
  skip('Video (WAN 2.2)', `Missing model files in ${wanDir}`);
}

// ─── Summary ───

console.log('\n' + '='.repeat(60));
console.log(`RESULTS: ${passed} passed, ${failed} failed, ${skipped} skipped`);
console.log('='.repeat(60));

console.log('\nOutput files:');
const { readdirSync, statSync } = await import('node:fs');
try {
  for (const f of readdirSync('test-output').sort()) {
    const s = statSync(path.join('test-output', f));
    if (s.isDirectory()) {
      console.log(`  ${f}/`);
    } else {
      console.log(`  ${f} (${(s.size / 1024).toFixed(0)}KB)`);
    }
  }
} catch { /* empty */ }

if (failed > 0) process.exit(1);
