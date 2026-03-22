import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { detectModelType } from '../src/model-detector.ts';

describe('detectModelType', () => {
  it('returns "image" for .safetensors files', async () => {
    const result = await detectModelType('/some/path/model.safetensors');
    assert.equal(result, 'image');
  });

  it('returns "image" for .ckpt files', async () => {
    const result = await detectModelType('/some/path/model.ckpt');
    assert.equal(result, 'image');
  });

  it('returns null for unknown extensions', async () => {
    const result = await detectModelType('/some/path/model.bin');
    assert.equal(result, null);
  });

  it('returns null for non-existent .gguf files', async () => {
    const result = await detectModelType('/nonexistent/model.gguf');
    assert.equal(result, null);
  });
});
