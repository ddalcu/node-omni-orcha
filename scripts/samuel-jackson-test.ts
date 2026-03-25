/**
 * Samuel L. Jackson-themed integration test across all engines.
 * Tests LLM with and without reasoning, TTS voice cloning, STT transcription of TTS output.
 * Outputs everything to test-output/ with datetime and model params in filenames.
 *
 * Usage: node scripts/samuel-jackson-test.ts
 */

import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { createModel } from '../src/index.ts';
import { saveTestOutput } from '../test/test-output-helper.ts';
import type { LlmModel, TtsModel, SttModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || `${process.env['HOME']}/.orcha/workspace/.models`;
const FIXTURES = path.resolve(import.meta.dirname!, '..', 'test', 'fixtures');

function resolveModel(...candidates: string[]): string | null {
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return null;
}

// Resample 24kHz WAV → 16kHz 16-bit PCM for whisper
function wavToWhisperPcm(wav: Buffer): Buffer {
  const pcm24k = wav.subarray(44); // skip WAV header
  const samplesIn = pcm24k.length / 2;
  const ratio = 16000 / 24000;
  const samplesOut = Math.floor(samplesIn * ratio);
  const pcm16k = Buffer.alloc(samplesOut * 2);
  for (let i = 0; i < samplesOut; i++) {
    const srcIdx = Math.min(Math.floor(i / ratio), samplesIn - 1);
    pcm16k.writeInt16LE(pcm24k.readInt16LE(srcIdx * 2), i * 2);
  }
  return pcm16k;
}

async function main() {
  console.log('Samuel L. Jackson Integration Test\n');

  const ttsWavBuffers: { phrase: string; wav: Buffer; filename: string }[] = [];

  // ─── 1. LLM ───

  console.log('=== LLM: Samuel L. Jackson (with and without reasoning) ===');
  const llmPath = resolveModel(path.join(MODELS_DIR, 'qwen3-5-4b', 'Qwen3.5-4B-IQ4_NL.gguf'));

  if (llmPath) {
    try {
      const llm = createModel(llmPath, 'llm');
      await llm.load({ contextSize: 4096 });

      const systemMsg = {
        role: 'system' as const,
        content: 'You are Samuel L. Jackson. Be brief, intense, and in character. No hedging, no disclaimers.',
      };

      const prompts = [
        'Give a motivational speech about never giving up. Keep it under 100 words.',
        'What is your opinion on snakes on a plane? Respond in 50 words.',
      ];

      // Test WITH capped reasoning (budget=64 tokens, then respond)
      console.log('\n  --- With reasoning (thinkingBudget=64, maxTokens=512) ---');
      for (let i = 0; i < prompts.length; i++) {
        console.log(`\n  Prompt ${i + 1}: ${prompts[i]!.slice(0, 60)}...`);
        const opts = { temperature: 0.8, maxTokens: 512, thinkingBudget: 64 };
        const result = await llm.complete(
          [systemMsg, { role: 'user', content: prompts[i]! }],
          opts,
        );
        const contentPreview = result.content.trim().slice(0, 300) || '(empty)';
        const reasoningLen = result.reasoning?.length ?? 0;
        console.log(`  Content (${result.content.length} chars): ${contentPreview}`);
        console.log(`  Reasoning: ${reasoningLen} chars`);
        console.log(`  Usage: ${result.usage.outputTokens} output tokens`);

        const text = `Prompt: ${prompts[i]}\n\nContent:\n${result.content}\n\nReasoning:\n${result.reasoning ?? ''}\n\nUsage: ${JSON.stringify(result.usage)}`;
        saveTestOutput('llm', 'Qwen3.5-4B-IQ4_NL', { ...opts, prompt: `samuel-reasoning-${i + 1}` }, text, '.txt');
      }

      // Test WITHOUT reasoning (thinkingBudget=0, direct response)
      console.log('\n  --- Without reasoning (thinkingBudget=0, maxTokens=512) ---');
      for (let i = 0; i < prompts.length; i++) {
        console.log(`\n  Prompt ${i + 1}: ${prompts[i]!.slice(0, 60)}...`);
        const opts = { temperature: 0.8, maxTokens: 512, thinkingBudget: 0 };
        const result = await llm.complete(
          [systemMsg, { role: 'user', content: prompts[i]! }],
          opts,
        );
        const contentPreview = result.content.trim().slice(0, 300) || '(empty)';
        const reasoningLen = result.reasoning?.length ?? 0;
        console.log(`  Content (${result.content.length} chars): ${contentPreview}`);
        console.log(`  Reasoning: ${reasoningLen} chars`);
        console.log(`  Usage: ${result.usage.outputTokens} output tokens`);

        const text = `Prompt: ${prompts[i]}\n\nContent:\n${result.content}\n\nReasoning:\n${result.reasoning ?? ''}\n\nUsage: ${JSON.stringify(result.usage)}`;
        saveTestOutput('llm', 'Qwen3.5-4B-IQ4_NL', { ...opts, prompt: `samuel-noreason-${i + 1}` }, text, '.txt');
      }

      await llm.unload();
      console.log('\n  LLM done.\n');
    } catch (err) {
      console.error('  LLM error:', (err as Error).message);
    }
  } else {
    console.log('  SKIP: No Qwen3.5-4B model found\n');
  }

  // ─── 2. Image ───

  console.log('=== Image: Generate Samuel L. Jackson (FLUX 2 Klein) ===');
  const fluxDir = path.join(MODELS_DIR, 'flux2-klein');
  const fluxModel = path.join(fluxDir, 'flux-2-klein-4b-Q4_K_M.gguf');
  const fluxLlm = path.join(fluxDir, 'Qwen3-4B-Q4_K_M.gguf');
  const fluxVae = path.join(fluxDir, 'flux2-vae.safetensors');

  if (existsSync(fluxModel) && existsSync(fluxLlm) && existsSync(fluxVae)) {
    try {
      const img = createModel(fluxModel, 'image');
      await img.load({ llmPath: fluxLlm, vaePath: fluxVae, keepVaeOnCpu: true, flashAttn: true });

      const prompt = 'A cartoon caricature of Samuel L. Jackson looking intense, bright colors, comic style';
      const opts = { width: 512, height: 512, steps: 4, cfgScale: 1.0, sampleMethod: 'euler' as const };

      console.log(`\n  Generating: ${prompt.slice(0, 50)}...`);
      const startTime = Date.now();
      const png = await img.generate(prompt, opts);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

      const outPath = saveTestOutput('image', 'flux2-klein-4b-Q4_K_M', opts, png, '.png');
      console.log(`  Saved: ${outPath} (${(png.length / 1024).toFixed(0)}KB, ${elapsed}s)`);

      await img.unload();
      console.log('\n  Image done.\n');
    } catch (err) {
      console.error('  Image error:', (err as Error).message);
    }
  } else {
    console.log('  SKIP: FLUX 2 Klein models not found\n');
  }

  // ─── 3. TTS ───

  console.log('=== TTS: Voice Clone (Samuel L. Jackson) ===');
  const qwen3TtsDir = resolveModel(path.join(MODELS_DIR, 'qwen3-tts'));
  const samuelWav = path.join(FIXTURES, 'Samuel.wav');

  if (qwen3TtsDir && existsSync(samuelWav)) {
    try {
      const tts = createModel(qwen3TtsDir, 'tts');
      await tts.load();

      const phrases = [
        'I have had it with these snakes on this plane. Everybody strap in, I am about to open some windows.',
        'English, do you speak it? Then you know what I am saying.',
      ];

      for (let i = 0; i < phrases.length; i++) {
        console.log(`\n  Speaking ${i + 1}: ${phrases[i]!.slice(0, 50)}...`);
        const startTime = Date.now();
        const wav = await tts.speak(phrases[i]!, { referenceAudioPath: samuelWav });
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const durationSec = ((wav.length - 44) / (24000 * 2)).toFixed(1);

        const outPath = saveTestOutput('tts', 'qwen3-0.6b-f16', { voice: `samuel-clone-${i + 1}` }, wav, '.wav');
        console.log(`  Saved: ${outPath} (${(wav.length / 1024).toFixed(0)}KB, ${durationSec}s audio, ${elapsed}s)`);

        ttsWavBuffers.push({ phrase: phrases[i]!, wav, filename: path.basename(outPath) });
      }

      await tts.unload();
      console.log('\n  TTS done.\n');
    } catch (err) {
      console.error('  TTS error:', (err as Error).message);
    }
  } else {
    console.log('  SKIP: Qwen3 TTS models or Samuel.wav not found\n');
  }

  // ─── 4. STT: Transcribe the TTS output ───

  console.log('=== STT: Transcribe TTS Output (Whisper) ===');
  const whisperPath = path.join(FIXTURES, 'whisper-tiny.bin');

  if (existsSync(whisperPath) && ttsWavBuffers.length > 0) {
    try {
      const stt = createModel(whisperPath, 'stt');
      await stt.load();

      for (const { phrase, wav, filename } of ttsWavBuffers) {
        const pcm16k = wavToWhisperPcm(wav);
        console.log(`\n  Transcribing: ${filename}`);
        console.log(`  Original:     "${phrase}"`);

        const result = await stt.transcribe(pcm16k, { language: 'en' });
        console.log(`  Transcribed:  "${result.text.trim()}"`);
        console.log(`  Language: ${result.language}, Segments: ${result.segments.length}`);

        const text = [
          `Source: ${filename}`,
          `Original:      ${phrase}`,
          `Transcription: ${result.text.trim()}`,
          `Language: ${result.language}`,
          `Segments:\n${JSON.stringify(result.segments, null, 2)}`,
        ].join('\n');
        saveTestOutput('stt', 'whisper-tiny', { lang: 'en', source: filename.replace('.wav', '') }, text, '.txt');
      }

      await stt.unload();
      console.log('\n  STT done.\n');
    } catch (err) {
      console.error('  STT error:', (err as Error).message);
    }
  } else if (!existsSync(whisperPath)) {
    console.log('  SKIP: No whisper model. Run: bash scripts/download-test-models.sh --whisper\n');
  } else {
    console.log('  SKIP: No TTS output to transcribe.\n');
  }

  // ─── Summary ───

  console.log('=== Output Files ===');
  const { readdirSync, statSync } = await import('node:fs');
  try {
    for (const f of readdirSync('test-output').sort()) {
      if (f.startsWith('.')) continue;
      const s = statSync(path.join('test-output', f));
      if (s.isDirectory()) {
        console.log(`  ${f}/`);
      } else {
        console.log(`  ${f} (${(s.size / 1024).toFixed(0)}KB)`);
      }
    }
  } catch { /* empty */ }

  console.log('\nDone!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
