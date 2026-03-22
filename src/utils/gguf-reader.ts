import * as fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import * as os from 'node:os';
import type { GGUFModelInfo } from '../types.ts';

const GGUF_MAGIC = 0x46554747;
const METADATA_BUFFER_SIZE = 1024 * 1024; // 1MB
const OS_RESERVED_BYTES = 4 * 1024 * 1024 * 1024; // 4GB

/**
 * Reads model metadata from a GGUF file header.
 * Only reads the first 1MB — no model loading required.
 */
export async function readGGUFMetadata(modelPath: string): Promise<GGUFModelInfo | null> {
  let handle: fs.FileHandle | null = null;
  try {
    handle = await fs.open(modelPath, 'r');
    const stat = await handle.stat();
    const buf = Buffer.alloc(METADATA_BUFFER_SIZE);
    const { bytesRead } = await handle.read(buf, 0, buf.length, 0);
    if (bytesRead < 24) return null;

    const magic = buf.readUInt32LE(0);
    if (magic !== GGUF_MAGIC) return null;

    const version = buf.readUInt32LE(4);
    if (version < 2 || version > 3) return null;

    const kvCount = Number(buf.readBigUInt64LE(16));
    let pos = 24;

    const info: Partial<GGUFModelInfo> = { fileSizeBytes: stat.size };
    const needed = new Set([
      'general.architecture',
      'context_length',
      'block_count',
      'embedding_length',
      'attention.head_count',
      'attention.head_count_kv',
    ]);

    for (let i = 0; i < kvCount && pos < bytesRead - 12 && needed.size > 0; i++) {
      if (pos + 8 > bytesRead) break;
      const keyLen = Number(buf.readBigUInt64LE(pos));
      pos += 8;

      if (pos + keyLen > bytesRead) break;
      const key = buf.toString('utf-8', pos, pos + keyLen);
      pos += keyLen;

      if (pos + 4 > bytesRead) break;
      const vtype = buf.readUInt32LE(pos);
      pos += 4;

      if (key === 'general.architecture') {
        info.architecture = readString(buf, pos, vtype) ?? 'unknown';
        needed.delete('general.architecture');
      } else if (key.endsWith('.context_length')) {
        info.contextLength = readScalar(buf, pos, vtype) ?? 0;
        needed.delete('context_length');
      } else if (key.endsWith('.block_count')) {
        info.blockCount = readScalar(buf, pos, vtype) ?? 0;
        needed.delete('block_count');
      } else if (key.endsWith('.embedding_length')) {
        info.embeddingLength = readScalar(buf, pos, vtype) ?? 0;
        needed.delete('embedding_length');
      } else if (key.endsWith('.attention.head_count_kv')) {
        info.headCountKv = readScalar(buf, pos, vtype) ?? 0;
        needed.delete('attention.head_count_kv');
      } else if (key.endsWith('.attention.head_count')) {
        info.headCount = readScalar(buf, pos, vtype) ?? 0;
        needed.delete('attention.head_count');
      }

      pos = skipValue(buf, pos, vtype, bytesRead);
      if (pos < 0) break;
    }

    if (!info.architecture) info.architecture = 'unknown';
    if (!info.contextLength) return null;

    return info as GGUFModelInfo;
  } catch {
    return null;
  } finally {
    await handle?.close();
  }
}

/**
 * Estimates KV cache bytes per token.
 * KV cache = 2 (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes (f16)
 */
export function kvCacheBytesPerToken(info: GGUFModelInfo): number {
  const headDim = info.embeddingLength / info.headCount;
  return 2 * info.blockCount * info.headCountKv * headDim * 2;
}

/**
 * Returns effective total memory, respecting container cgroup limits.
 */
function getEffectiveMemory(): number {
  const hostRam = os.totalmem();
  if (process.platform !== 'linux') return hostRam;
  try {
    const raw = readFileSync('/sys/fs/cgroup/memory.max', 'utf-8').trim();
    if (raw !== 'max') return Math.min(Number(raw), hostRam);
  } catch { /* not cgroup v2 */ }
  try {
    const raw = readFileSync('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'utf-8').trim();
    const limit = Number(raw);
    if (limit > 0 && limit < hostRam) return limit;
  } catch { /* not cgroup v1 */ }
  return hostRam;
}

/**
 * Calculates optimal context size based on available system RAM.
 */
export function calculateOptimalContextSize(info: GGUFModelInfo): number {
  const totalRam = getEffectiveMemory();
  const availableForModel = totalRam - OS_RESERVED_BYTES;
  const memAfterWeights = availableForModel - info.fileSizeBytes;

  if (memAfterWeights <= 0) return 2048;

  const bytesPerToken = kvCacheBytesPerToken(info);
  const maxCtxByRam = Math.floor((memAfterWeights * 0.5) / bytesPerToken);
  const MAX_CONTEXT_CAP = 32768;

  const optimal = Math.min(maxCtxByRam, info.contextLength, MAX_CONTEXT_CAP);
  return Math.max(2048, Math.floor(optimal / 1024) * 1024);
}

function readString(buf: Buffer, pos: number, vtype: number): string | null {
  if (vtype !== 8) return null;
  if (pos + 8 > buf.length) return null;
  const len = Number(buf.readBigUInt64LE(pos));
  if (pos + 8 + len > buf.length) return null;
  return buf.toString('utf-8', pos + 8, pos + 8 + len);
}

function readScalar(buf: Buffer, pos: number, vtype: number): number | null {
  switch (vtype) {
    case 0: return buf.readUInt8(pos);
    case 1: return buf.readInt8(pos);
    case 2: return buf.readUInt16LE(pos);
    case 3: return buf.readInt16LE(pos);
    case 4: return buf.readUInt32LE(pos);
    case 5: return buf.readInt32LE(pos);
    case 6: return buf.readFloatLE(pos);
    case 7: return buf.readUInt8(pos);
    case 10: return Number(buf.readBigUInt64LE(pos));
    case 11: return Number(buf.readBigInt64LE(pos));
    case 12: return buf.readDoubleLE(pos);
    default: return null;
  }
}

function skipValue(buf: Buffer, pos: number, vtype: number, limit: number): number {
  switch (vtype) {
    case 0: case 1: case 7: return pos + 1;
    case 2: case 3: return pos + 2;
    case 4: case 5: case 6: return pos + 4;
    case 10: case 11: case 12: return pos + 8;
    case 8: {
      if (pos + 8 > limit) return -1;
      const len = Number(buf.readBigUInt64LE(pos));
      return pos + 8 + len;
    }
    case 9: {
      if (pos + 12 > limit) return -1;
      const elemType = buf.readUInt32LE(pos);
      const count = Number(buf.readBigUInt64LE(pos + 4));
      pos += 12;
      for (let i = 0; i < count && pos < limit; i++) {
        pos = skipValue(buf, pos, elemType, limit);
        if (pos < 0) return -1;
      }
      return pos;
    }
    default: return -1;
  }
}
