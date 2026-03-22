import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { writeFile, mkdir } from 'node:fs/promises';
import * as path from 'node:path';
import { createModel } from '../src/index.ts';
import type { TtsModel } from '../src/types.ts';

const QWEN3_MODELS_DIR = new URL('../deps/qwen3-tts.cpp/models', import.meta.url).pathname;
const GLADOS_WAV = new URL('./fixtures/GLaDOS.wav', import.meta.url).pathname;
const SAMUEL_WAV = new URL('./fixtures/Samuel.wav', import.meta.url).pathname;
const OUTPUT_DIR = new URL('../test-output', import.meta.url).pathname;

const hasModel = existsSync(`${QWEN3_MODELS_DIR}/qwen3-tts-0.6b-f16.gguf`);

const NEWS_TEXT = "Trump threatens to strike Iran's power plants if it doesn't reopen Strait of Hormuz. The president had earlier talked of winding down the war, while Israel said strikes will increase significantly.";

describe('Qwen3 TTS voice cloning', { skip: !hasModel ? 'No Qwen3 TTS models' : undefined }, () => {
  let tts: TtsModel;

  before(async () => {
    await mkdir(OUTPUT_DIR, { recursive: true });
    tts = createModel(QWEN3_MODELS_DIR, 'tts');
    await tts.load({ engine: 'qwen3' });
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

    const outPath = path.join(OUTPUT_DIR, 'tts-glados-clone.wav');
    await writeFile(outPath, wav);
    console.log(`    Saved: ${outPath} (${(wav.length / 1024).toFixed(0)}KB)`);
  });

  it('clones Samuel L. Jackson voice', { skip: !existsSync(SAMUEL_WAV) ? 'No Samuel.wav' : undefined }, async () => {
    const wav = await tts.speak(NEWS_TEXT, { referenceAudioPath: SAMUEL_WAV });

    assert.ok(Buffer.isBuffer(wav), 'Should return a Buffer');
    assert.ok(wav.length > 1000, `WAV too small: ${wav.length} bytes`);
    assert.equal(wav.toString('ascii', 0, 4), 'RIFF');

    const outPath = path.join(OUTPUT_DIR, 'tts-samuel-clone.wav');
    await writeFile(outPath, wav);
    console.log(`    Saved: ${outPath} (${(wav.length / 1024).toFixed(0)}KB)`);
  });
});
