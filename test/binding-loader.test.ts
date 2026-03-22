import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { loadBinding } from '../src/binding-loader.ts';

describe('loadBinding', () => {
  it('loads the llm binding from local build', () => {
    const binding = loadBinding('llm');
    assert.ok(binding, 'Binding should be loaded');
    assert.equal(typeof binding['createContext'], 'function', 'Should export createContext');
  });

  it('caches bindings across calls', () => {
    const a = loadBinding('llm');
    const b = loadBinding('llm');
    assert.strictEqual(a, b, 'Should return same cached instance');
  });

  it('loads the image binding from local build', () => {
    const binding = loadBinding('image');
    assert.ok(binding, 'Binding should be loaded');
    assert.equal(typeof binding['createContext'], 'function', 'Should export createContext');
  });

  it('loads the stt binding from local build', () => {
    const binding = loadBinding('stt');
    assert.ok(binding, 'Binding should be loaded');
    assert.equal(typeof binding['createContext'], 'function');
  });

  it('loads the tts binding from local build', () => {
    const binding = loadBinding('tts');
    assert.ok(binding, 'Binding should be loaded');
    assert.equal(typeof binding['createContext'], 'function');
  });

  it('throws for unknown engine', () => {
    assert.throws(() => loadBinding('nonexistent'), /not found/i);
  });
});
