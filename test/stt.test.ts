import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createModel } from '../src/index.ts';
import { loadBinding } from '../src/binding-loader.ts';
import type { SttModel } from '../src/types.ts';

// Binding loader test
describe('STT binding loader', () => {
  it('loads the stt binding from local build', () => {
    const binding = loadBinding('stt');
    assert.ok(binding, 'Binding should be loaded');
    assert.equal(typeof binding['createContext'], 'function', 'Should export createContext');
  });
});

// createModel test (no model needed)
describe('createModel stt', () => {
  it('creates unloaded stt model', () => {
    const model = createModel('/fake/whisper.gguf', 'stt');
    assert.equal(model.type, 'stt');
    assert.equal(model.loaded, false);
  });
});

// Integration tests (require model + audio)
const WHISPER_MODEL_PATH = new URL('./fixtures/whisper-tiny.bin', import.meta.url).pathname;
const AUDIO_PATH = new URL('./fixtures/test-audio.pcm', import.meta.url).pathname;
const hasModel = existsSync(WHISPER_MODEL_PATH);
const hasAudio = existsSync(AUDIO_PATH);

describe('SttModel', { skip: !hasModel ? 'No whisper model at test/fixtures/whisper-tiny.bin — run scripts/download-test-models.sh --whisper' : undefined }, () => {
  let model: SttModel;

  before(async () => {
    model = createModel(WHISPER_MODEL_PATH, 'stt');
    await model.load();
  });

  after(async () => {
    await model?.unload();
  });

  it('loads and reports as loaded', () => {
    assert.equal(model.type, 'stt');
    assert.equal(model.loaded, true);
  });

  it('transcribes audio', { skip: !hasAudio ? 'No test audio at test/fixtures/test-audio.pcm' : undefined }, async () => {
    const audio = await readFile(AUDIO_PATH);
    const result = await model.transcribe(audio, { language: 'en' });

    assert.ok(result.text.length > 0, 'Should produce text');
    assert.ok(result.segments.length > 0, 'Should have segments');
    assert.ok(result.segments[0]!.start >= 0, 'Segment should have start time');
    assert.ok(result.segments[0]!.end > result.segments[0]!.start, 'Segment end > start');
    assert.ok(result.segments[0]!.text.length > 0, 'Segment should have text');
  });

  it('detects language', { skip: !hasAudio ? 'No test audio' : undefined }, async () => {
    const audio = await readFile(AUDIO_PATH);
    const lang = await model.detectLanguage(audio);

    assert.equal(typeof lang, 'string');
    assert.ok(lang.length > 0, 'Should detect a language');
  });
});
