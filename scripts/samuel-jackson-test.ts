/**
 * Samuel L. Jackson-themed integration test across all engines.
 * Generates output to test-output/<datetime>/
 *
 * Usage:
 *   node scripts/samuel-jackson-test.ts
 *
 * Models used (symlinked/copied from agent-orcha):
 *   LLM:   Qwen3.5-4B-IQ4_NL.gguf
 *   Image: flux-2-klein-4b-Q8_0.gguf + clip_l + t5xxl + vae
 *   STT:   whisper-tiny.bin (optional)
 *   TTS:   kokoro-no-espeak-q8.gguf (Kokoro)
 */

import { mkdir, writeFile, copyFile, symlink, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { createModel } from '../src/index.ts';
import type { LlmModel, ImageModel } from '../src/types.ts';

const ORCHA_MODELS = process.env['ORCHA_MODELS'] || path.resolve(import.meta.dirname!, '..', '..', 'agent-orcha', 'templates', '.models');
const OUTPUT_BASE = path.resolve(import.meta.dirname!, '..', 'test-output');

function timestamp(): string {
  const d = new Date();
  return d.toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

async function ensureModel(src: string, dest: string): Promise<boolean> {
  if (existsSync(dest)) return true;
  if (!existsSync(src)) {
    console.log(`  SKIP: ${path.basename(src)} not found`);
    return false;
  }
  try {
    await symlink(src, dest);
  } catch {
    // symlink may fail, try copy
    console.log(`  Copying ${path.basename(src)}...`);
    await copyFile(src, dest);
  }
  return true;
}

async function main() {
  const outDir = path.join(OUTPUT_BASE, timestamp());
  await mkdir(outDir, { recursive: true });
  console.log(`Output: ${outDir}\n`);

  const fixturesDir = path.resolve(import.meta.dirname!, '..', 'test', 'fixtures');
  await mkdir(fixturesDir, { recursive: true });

  // ─── 1. LLM: Talk like Samuel L. Jackson ───
  console.log('=== LLM: Talk like Samuel L. Jackson ===');
  const llmPath = path.join(fixturesDir, 'Qwen3.5-4B-IQ4_NL.gguf');
  const hasLlm = await ensureModel(
    path.join(ORCHA_MODELS, 'Qwen3.5-4B-IQ4_NL.gguf'),
    llmPath,
  );

  if (hasLlm) {
    try {
      const llm = createModel(llmPath, 'llm');
      await llm.load({ contextSize: 2048 });

      const systemMsg = { role: 'system' as const, content: 'You are Samuel L. Jackson. Respond directly without any thinking or reasoning tags. Be brief, intense, and in character.' };

      const prompts = [
        'Give a motivational speech about never giving up. Keep it under 100 words.',
        'What is your opinion on snakes on a plane? Respond in 50 words.',
        'Give a review of a royale with cheese in 30 words.',
      ];

      for (let i = 0; i < prompts.length; i++) {
        console.log(`\n  Prompt ${i + 1}: ${prompts[i]!.slice(0, 60)}...`);
        const result = await llm.complete(
          [systemMsg, { role: 'user', content: prompts[i]! }],
          { temperature: 0.8, maxTokens: 512 },
        );
        console.log(`  Response (${result.usage.outputTokens} tokens):\n    ${result.content.trim().slice(0, 200)}`);
        await writeFile(
          path.join(outDir, `llm-samuel-${i + 1}.txt`),
          `Prompt: ${prompts[i]}\n\nResponse:\n${result.content}\n\nTokens: ${JSON.stringify(result.usage)}`,
        );
      }

      await llm.unload();
      console.log('\n  LLM done.\n');
    } catch (err) {
      console.error('  LLM error:', (err as Error).message);
    }
  }

  // ─── 2. Image: Generate Samuel L. Jackson ───
  console.log('=== Image: Generate Samuel L. Jackson (FLUX.2 Klein 4B) ===');
  const fluxModelPath = path.join(fixturesDir, 'flux-2-klein-4b-Q8_0.gguf');
  const llmEncoderPath = path.join(fixturesDir, 'Qwen3-4B-Q8_0.gguf');
  const vaePath = path.join(fixturesDir, 'flux2-ae.safetensors');

  const hasFlux = (
    await ensureModel(path.join(ORCHA_MODELS, 'flux-2-klein-4b-Q8_0.gguf'), fluxModelPath) &&
    await ensureModel(path.join(ORCHA_MODELS, 'Qwen3-4B-Q8_0.gguf'), llmEncoderPath) &&
    await ensureModel(path.join(ORCHA_MODELS, 'flux2-ae.safetensors'), vaePath)
  );

  if (hasFlux) {
    try {
      const img = createModel(fluxModelPath, 'image');
      await img.load({
        llmPath: llmEncoderPath,
        vaePath,
        offloadToCpu: true,
        flashAttn: true,
      });

      // Generate one image per context to avoid sd.cpp context reuse crash
      const imagePrompts = [
        'A cartoon caricature of Samuel L. Jackson looking intense, bright colors, comic style',
      ];

      for (let i = 0; i < imagePrompts.length; i++) {
        console.log(`\n  Generating image ${i + 1}: ${imagePrompts[i]!.slice(0, 50)}...`);
        const startTime = Date.now();
        const png = await img.generate(imagePrompts[i]!, {
          width: 512,
          height: 512,
          steps: 4,
          cfgScale: 1.0,
          sampleMethod: 'euler',
        });
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const filename = `image-samuel-${i + 1}.png`;
        await writeFile(path.join(outDir, filename), png);
        console.log(`  Saved ${filename} (${(png.length / 1024).toFixed(0)}KB, ${elapsed}s)`);
      }

      await img.unload();
      console.log('\n  Image done.\n');
    } catch (err) {
      console.error('  Image error:', (err as Error).message);
    }
  }

  // ─── 3. TTS: Speak like Samuel L. Jackson (before STT so we can transcribe it back) ───
  console.log('=== TTS: Speak like Samuel L. Jackson (Kokoro) ===');
  const kokoroPath = path.join(fixturesDir, 'kokoro-no-espeak-q8.gguf');
  let ttsWavBuffer: Buffer | null = null;
  const ttsPhrase = 'Say what again. I dare you, I double dare you. Say what one more time.';

  if (existsSync(kokoroPath)) {
    try {
      const tts = createModel(kokoroPath, 'tts');
      await tts.load();

      const ttsPrompts = [
        ttsPhrase,
        'English, do you speak it? Then you know what I am saying.',
      ];

      for (let i = 0; i < ttsPrompts.length; i++) {
        console.log(`\n  Speaking ${i + 1}: ${ttsPrompts[i]!.slice(0, 50)}...`);
        const startTime = Date.now();
        const wav = await tts.speak(ttsPrompts[i]!);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const filename = `tts-samuel-${i + 1}.wav`;
        await writeFile(path.join(outDir, filename), wav);
        console.log(`  Saved ${filename} (${(wav.length / 1024).toFixed(0)}KB, ${elapsed}s)`);
        if (i === 0) ttsWavBuffer = wav;
      }

      await tts.unload();
      console.log('\n  TTS done.\n');
    } catch (err) {
      console.error('  TTS error:', (err as Error).message);
    }
  } else {
    console.log('  SKIP: No Kokoro model. Download from: https://huggingface.co/mmwillet2/Kokoro_GGUF\n');
  }

  // ─── 4. STT: Transcribe the TTS output back ───
  console.log('=== STT: Transcribe TTS Output (Whisper Tiny) ===');
  const whisperPath = path.join(fixturesDir, 'whisper-tiny.bin');
  if (existsSync(whisperPath) && ttsWavBuffer) {
    try {
      const stt = createModel(whisperPath, 'stt');
      await stt.load();

      // Extract PCM from WAV and resample 24kHz → 16kHz
      // WAV header is 44 bytes, then 16-bit signed PCM data
      const pcm24k = ttsWavBuffer.subarray(44);
      const samplesIn = pcm24k.length / 2;
      const ratio = 16000 / 24000; // 2/3
      const samplesOut = Math.floor(samplesIn * ratio);
      const pcm16k = Buffer.alloc(samplesOut * 2);

      for (let i = 0; i < samplesOut; i++) {
        const srcIdx = Math.floor(i / ratio);
        const clamped = Math.min(srcIdx, samplesIn - 1);
        pcm16k.writeInt16LE(pcm24k.readInt16LE(clamped * 2), i * 2);
      }

      console.log(`  Original TTS: "${ttsPhrase}"`);
      console.log(`  Resampled: ${samplesIn} samples @24kHz → ${samplesOut} samples @16kHz`);

      const result = await stt.transcribe(pcm16k, { language: 'en' });
      console.log(`  Transcribed:  "${result.text.trim()}"`);
      console.log(`  Language: ${result.language}, Segments: ${result.segments.length}`);

      await writeFile(
        path.join(outDir, 'stt-result.txt'),
        `Original:      ${ttsPhrase}\nTranscription: ${result.text.trim()}\nLanguage: ${result.language}\nSegments: ${JSON.stringify(result.segments, null, 2)}`,
      );

      await stt.unload();
      console.log('  STT done.\n');
    } catch (err) {
      console.error('  STT error:', (err as Error).message);
    }
  } else if (!existsSync(whisperPath)) {
    console.log('  SKIP: No whisper model. Run: bash scripts/download-test-models.sh --whisper\n');
  } else {
    console.log('  SKIP: No TTS output to transcribe.\n');
  }

  // ─── 5. Qwen3 TTS: Voice cloning ───
  console.log('=== Qwen3 TTS: Voice Clone (Samuel L. Jackson) ===');
  const qwen3ModelsDir = path.resolve(import.meta.dirname!, '..', 'deps', 'qwen3-tts.cpp', 'models');
  const qwen3ModelFile = path.join(qwen3ModelsDir, 'qwen3-tts-0.6b-f16.gguf');
  const samuelRefAudio = path.join(fixturesDir, 'Samuel.wav');

  if (existsSync(qwen3ModelFile) && existsSync(samuelRefAudio)) {
    try {
      const tts = createModel(qwen3ModelsDir, 'tts');
      await tts.load({ engine: 'qwen3' });

      console.log(`  Cloning Samuel L. Jackson's voice from: Samuel.wav`);
      const startTime = Date.now();
      const wav = await tts.speak(
        'I have had it with these snakes on this plane. Everybody strap in, I am about to open some windows.',
        { referenceAudioPath: samuelRefAudio },
      );
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const filename = 'tts-qwen3-cloned.wav';
      await writeFile(path.join(outDir, filename), wav);
      console.log(`  Saved ${filename} (${(wav.length / 1024).toFixed(0)}KB, ${elapsed}s)`);

      // Also generate without cloning for comparison
      const wav2 = await tts.speak('This is Qwen3 TTS without voice cloning.');
      await writeFile(path.join(outDir, 'tts-qwen3-default.wav'), wav2);
      console.log(`  Saved tts-qwen3-default.wav (${(wav2.length / 1024).toFixed(0)}KB)`);

      await tts.unload();
      console.log('  Qwen3 TTS done.\n');
    } catch (err) {
      console.error('  Qwen3 TTS error:', (err as Error).message);
    }
  } else {
    if (!existsSync(qwen3ModelFile)) {
      console.log('  SKIP: No Qwen3 TTS models. Run setup in deps/qwen3-tts.cpp/\n');
    } else {
      console.log('  SKIP: No Samuel.wav reference audio at test/fixtures/Samuel.wav\n');
    }
  }

  // ─── Summary ───
  console.log('=== Summary ===');
  console.log(`Output directory: ${outDir}`);
  const files = await import('node:fs/promises').then(fs => fs.readdir(outDir));
  for (const f of files) {
    const s = await stat(path.join(outDir, f));
    console.log(`  ${f} (${(s.size / 1024).toFixed(0)}KB)`);
  }
  console.log('\nDone!');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
