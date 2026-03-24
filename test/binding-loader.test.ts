import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { loadBinding } from '../src/binding-loader.ts';

describe('loadBinding', () => {
  it('loads the unified omni binding from local build', () => {
    const binding = loadBinding();
    assert.ok(binding, 'Binding should be loaded');
    assert.equal(typeof binding['createLlmContext'], 'function', 'Should export createLlmContext');
    assert.equal(typeof binding['createSttContext'], 'function', 'Should export createSttContext');
    assert.equal(typeof binding['createTtsContext'], 'function', 'Should export createTtsContext');
    assert.equal(typeof binding['createImageContext'], 'function', 'Should export createImageContext');
  });

  it('returns same cached instance across calls', () => {
    const a = loadBinding();
    const b = loadBinding();
    assert.strictEqual(a, b, 'Should return same cached instance');
  });
});
