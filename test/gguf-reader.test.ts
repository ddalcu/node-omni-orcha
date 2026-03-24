import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { readGGUFMetadata, kvCacheBytesPerToken, calculateOptimalContextSize } from '../src/utils/gguf-reader.ts';

describe('readGGUFMetadata', () => {
  it('returns null for non-existent files', async () => {
    const result = await readGGUFMetadata('/nonexistent/model.gguf');
    assert.equal(result, null);
  });

  it('returns null for non-GGUF files', async () => {
    // package.json is not a GGUF file
    const result = await readGGUFMetadata(fileURLToPath(new URL('../package.json', import.meta.url)));
    assert.equal(result, null);
  });
});

describe('kvCacheBytesPerToken', () => {
  it('calculates correctly for typical model', () => {
    const info = {
      architecture: 'llama',
      contextLength: 4096,
      blockCount: 32,
      embeddingLength: 4096,
      headCount: 32,
      headCountKv: 8,
      fileSizeBytes: 4 * 1024 * 1024 * 1024,
    };
    const bytes = kvCacheBytesPerToken(info);
    // 2 * 32 * 8 * 128 * 2 = 131072 bytes per token
    assert.equal(bytes, 131072);
  });
});

describe('calculateOptimalContextSize', () => {
  it('returns at least 2048', () => {
    const info = {
      architecture: 'llama',
      contextLength: 128000,
      blockCount: 80,
      embeddingLength: 8192,
      headCount: 64,
      headCountKv: 8,
      fileSizeBytes: 100 * 1024 * 1024 * 1024, // huge model
    };
    const ctx = calculateOptimalContextSize(info);
    assert.ok(ctx >= 2048, `Context size ${ctx} should be >= 2048`);
  });

  it('respects 32K cap', () => {
    const info = {
      architecture: 'llama',
      contextLength: 128000,
      blockCount: 32,
      embeddingLength: 4096,
      headCount: 32,
      headCountKv: 8,
      fileSizeBytes: 1 * 1024 * 1024 * 1024, // small model, lots of RAM
    };
    const ctx = calculateOptimalContextSize(info);
    assert.ok(ctx <= 32768, `Context size ${ctx} should be <= 32768`);
  });

  it('is a multiple of 1024', () => {
    const info = {
      architecture: 'llama',
      contextLength: 8192,
      blockCount: 32,
      embeddingLength: 4096,
      headCount: 32,
      headCountKv: 8,
      fileSizeBytes: 4 * 1024 * 1024 * 1024,
    };
    const ctx = calculateOptimalContextSize(info);
    assert.equal(ctx % 1024, 0, `Context size ${ctx} should be multiple of 1024`);
  });
});
