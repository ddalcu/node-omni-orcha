#!/usr/bin/env node
/**
 * Kokoro TTS integration test — benchmarks TTS speed and
 * tests roundtrip with Whisper STT.
 */

import { createModel } from '../src/index.ts';
import { writeFileSync } from 'node:fs';

const MODELS_DIR = process.env.MODELS_DIR ?? `${process.env.HOME}/.orcha/workspace/.models`;
const KOKORO_PATH = `${MODELS_DIR}/kokoro`;
const STT_PATH = `${MODELS_DIR}/whisper-tiny/whisper-tiny.bin`;

const TEST_PHRASES = [
  'Hello world, how are you today?',
  'The quick brown fox jumps over the lazy dog.',
  'Artificial intelligence is transforming the way we live and work.',
  'She sells seashells by the seashore.',
  'To be or not to be, that is the question.',
];

const VOICES_TO_TEST = ['af_heart', 'am_adam', 'bf_emma', 'bm_george'];

async function main() {
  console.log('═══════════════════════════════════════════');
  console.log('  Kokoro TTS Integration Test');
  console.log('═══════════════════════════════════════════\n');

  // ─── Load Kokoro ───
  console.log('Loading Kokoro TTS...');
  const ttsStart = performance.now();
  const kokoro = createModel(KOKORO_PATH, 'kokoro');
  await kokoro.load();
  const ttsLoadMs = performance.now() - ttsStart;
  console.log(`  Loaded in ${(ttsLoadMs / 1000).toFixed(2)}s`);
  console.log(`  Voices: ${kokoro.listVoices().length}`);
  console.log(`  Available: ${kokoro.listVoices().slice(0, 10).join(', ')}...\n`);

  // ─── Load STT ───
  console.log('Loading Whisper STT...');
  const sttStart = performance.now();
  const stt = createModel(STT_PATH, 'stt');
  await stt.load();
  const sttLoadMs = performance.now() - sttStart;
  console.log(`  Loaded in ${(sttLoadMs / 1000).toFixed(2)}s\n`);

  // ─── Benchmark: TTS speed per voice ───
  console.log('─── TTS Benchmarks (per voice) ───\n');
  const benchText = 'Hello, this is a test of the Kokoro text to speech system.';

  for (const voice of VOICES_TO_TEST) {
    const start = performance.now();
    const wav = await kokoro.speak(benchText, { voice });
    const ms = performance.now() - start;
    const seconds = wav.length > 44 ? (wav.length - 44) / 2 / 24000 : 0;
    console.log(`  ${voice.padEnd(12)} ${ms.toFixed(0).padStart(5)}ms → ${seconds.toFixed(1)}s audio (${wav.length} bytes)`);
  }

  // ─── Benchmark: TTS speed at different speeds ───
  console.log('\n─── Speed Control ───\n');
  for (const speed of [0.8, 1.0, 1.2, 1.5]) {
    const start = performance.now();
    const wav = await kokoro.speak(benchText, { voice: 'af_heart', speed });
    const ms = performance.now() - start;
    const seconds = wav.length > 44 ? (wav.length - 44) / 2 / 24000 : 0;
    console.log(`  speed=${speed.toFixed(1)} → ${ms.toFixed(0).padStart(5)}ms, ${seconds.toFixed(1)}s audio`);
  }

  // ─── TTS → STT Roundtrip ───
  console.log('\n─── TTS → STT Roundtrip (af_heart) ───\n');

  let totalTtsMs = 0;
  let totalSttMs = 0;
  let matches = 0;

  for (const phrase of TEST_PHRASES) {
    // TTS
    const t1 = performance.now();
    const wav = await kokoro.speak(phrase, { voice: 'af_heart' });
    const ttsMs = performance.now() - t1;
    totalTtsMs += ttsMs;

    // Extract PCM from WAV (skip 44-byte header, convert 16-bit LE to raw)
    // Kokoro outputs 24kHz, Whisper expects 16kHz — need to resample
    const pcm16bit = wav.subarray(44);
    const samples24k = new Int16Array(pcm16bit.buffer, pcm16bit.byteOffset, pcm16bit.length / 2);

    // Simple downsample 24kHz → 16kHz (every 3rd sample from 2 samples)
    const ratio = 24000 / 16000; // 1.5
    const outLen = Math.floor(samples24k.length / ratio);
    const samples16k = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = Math.min(Math.floor(i * ratio), samples24k.length - 1);
      samples16k[i] = samples24k[srcIdx];
    }
    const pcmBuf = Buffer.from(samples16k.buffer, samples16k.byteOffset, samples16k.byteLength);

    // STT
    const t2 = performance.now();
    const result = await stt.transcribe(pcmBuf, { language: 'en' });
    const sttMs = performance.now() - t2;
    totalSttMs += sttMs;

    const transcript = result.text.trim();
    const match = transcript.toLowerCase().includes(phrase.toLowerCase().substring(0, 20));
    if (match) matches++;

    console.log(`  Input:  "${phrase}"`);
    console.log(`  Output: "${transcript}"`);
    console.log(`  TTS: ${ttsMs.toFixed(0)}ms | STT: ${sttMs.toFixed(0)}ms | Match: ${match ? '✓' : '✗'}\n`);
  }

  // ─── Save sample WAV ───
  const sampleWav = await kokoro.speak('The future of voice AI is here.', { voice: 'af_heart' });
  const samplePath = '/tmp/kokoro-test-output.wav';
  writeFileSync(samplePath, sampleWav);
  console.log(`Sample WAV saved: ${samplePath}\n`);

  // ─── Summary ───
  console.log('═══════════════════════════════════════════');
  console.log('  SUMMARY');
  console.log('═══════════════════════════════════════════');
  console.log(`  Kokoro load:    ${(ttsLoadMs / 1000).toFixed(2)}s`);
  console.log(`  Whisper load:   ${(sttLoadMs / 1000).toFixed(2)}s`);
  console.log(`  Voices:         ${kokoro.listVoices().length}`);
  console.log(`  Dict words:     126,052`);
  console.log(`  Avg TTS:        ${(totalTtsMs / TEST_PHRASES.length).toFixed(0)}ms`);
  console.log(`  Avg STT:        ${(totalSttMs / TEST_PHRASES.length).toFixed(0)}ms`);
  console.log(`  Roundtrip:      ${matches}/${TEST_PHRASES.length} recognized`);
  console.log('═══════════════════════════════════════════');

  await kokoro.unload();
  await stt.unload();
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
