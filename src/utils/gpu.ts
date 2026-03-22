import { execFileSync } from 'node:child_process';
import type { GpuInfo } from '../types.ts';

let cached: GpuInfo | null = null;

export function detectGpu(): GpuInfo {
  if (cached) return cached;

  if (process.platform === 'darwin') {
    cached = { backend: 'metal' };
    return cached;
  }

  const nvidia = detectNvidia();
  if (nvidia) {
    cached = { backend: 'cuda', name: nvidia.name, vramBytes: nvidia.vramBytes };
    return cached;
  }

  cached = { backend: 'cpu' };
  return cached;
}

function detectNvidia(): { name?: string; vramBytes?: number } | null {
  try {
    const output = execFileSync('nvidia-smi', [], {
      encoding: 'utf-8',
      timeout: 10_000,
    });
    if (!output.includes('CUDA Version')) return null;

    let name: string | undefined;
    try {
      name = execFileSync('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], {
        encoding: 'utf-8',
        timeout: 5_000,
      }).trim().split('\n')[0];
    } catch { /* optional */ }

    let vramBytes: number | undefined;
    try {
      const mem = execFileSync('nvidia-smi', [
        '--query-gpu=memory.total', '--format=csv,noheader,nounits',
      ], { encoding: 'utf-8', timeout: 5_000 }).trim();
      const mib = parseInt(mem.split('\n')[0]!, 10);
      if (!isNaN(mib)) vramBytes = mib * 1024 * 1024;
    } catch { /* optional */ }

    return { name, vramBytes };
  } catch {
    return null;
  }
}

export function resetGpuCache(): void {
  cached = null;
}
