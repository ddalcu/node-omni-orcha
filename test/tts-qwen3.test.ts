import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { createModel } from '../src/index.ts';
import type { TtsModel } from '../src/types.ts';
import { saveTestOutput } from './test-output-helper.ts';

const QWEN3_MODELS_DIR = process.env['MODELS_DIR']
  ? `${process.env['MODELS_DIR']}/qwen3-tts`
  : `${process.env['HOME']}/.orcha/workspace/.models/qwen3-tts`;
const GLADOS_WAV = new URL('./fixtures/GLaDOS.wav', import.meta.url).pathname;
const SAMUEL_WAV = new URL('./fixtures/Samuel.wav', import.meta.url).pathname;

const hasModel = existsSync(`${QWEN3_MODELS_DIR}/qwen3-tts-0.6b-f16.gguf`);

const NEWS_TEXT = "Qwen TTS is an advanced family of open-source text-to-speech models developed by Alibaba Cloud's Qwen team. Released in January 2026, it is designed for high-quality, real-time speech generation across 10 major languages and supports features such as 3-second voice cloning, natural language voice design, and ultra-low latency streaming.";

describe('Qwen3 TTS voice cloning', { skip: !hasModel ? 'No Qwen3 TTS models' : undefined }, () => {
  let tts: TtsModel;

  before(async () => {
    tts = createModel(QWEN3_MODELS_DIR, 'tts');
    await tts.load();
  });

  after(async () => {
    await tts?.unload();
  });

  it('clones GLaDOS voice', { skip: !existsSync(GLADOS_WAV) ? 'No GLaDOS.wav' : undefined }, async () => {
    const wav = await tts.speak(
      'Adrian is a very bad boy, because he stays up all night when he should go to sleep.',
      { referenceAudioPath: GLADOS_WAV },
    );

    assert.ok(Buffer.isBuffer(wav), 'Should return a Buffer');
    assert.ok(wav.length > 1000, `WAV too small: ${wav.length} bytes`);
    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');

    const outPath = saveTestOutput('tts', 'qwen3-0.6b-f16', { voice: 'glados-clone' }, wav, '.wav');
    console.log(`    Saved: ${outPath} (${(wav.length / 1024).toFixed(0)}KB)`);
  });

  it('clones Samuel L. Jackson voice', { skip: !existsSync(SAMUEL_WAV) ? 'No Samuel.wav' : undefined }, async () => {
    const wav = await tts.speak(NEWS_TEXT, { referenceAudioPath: SAMUEL_WAV });

    assert.ok(Buffer.isBuffer(wav), 'Should return a Buffer');
    assert.ok(wav.length > 1000, `WAV too small: ${wav.length} bytes`);
    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');

    const outPath = saveTestOutput('tts', 'qwen3-0.6b-f16', { voice: 'samuel-clone' }, wav, '.wav');
    console.log(`    Saved: ${outPath} (${(wav.length / 1024).toFixed(0)}KB)`);
  });
});
