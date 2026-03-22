import { createRequire } from 'node:module';
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { detectGpu } from './utils/gpu.ts';

const require = createRequire(import.meta.url);

interface NativeBinding {
  [key: string]: (...args: unknown[]) => unknown;
}

const loadedBindings = new Map<string, NativeBinding>();

/**
 * Load a native .node addon by engine name.
 * Resolution order:
 *   1. Optional platform package (@node-omni-orcha/{platform}-{arch}-{gpu})
 *   2. Optional platform package without GPU suffix
 *   3. Local cmake-js build (./build/Release/{engine}.node)
 */
export function loadBinding(engine: string): NativeBinding {
  const cached = loadedBindings.get(engine);
  if (cached) return cached;

  const platform = process.platform;
  const arch = process.arch;
  const gpu = detectGpu();

  // Try optional platform packages
  const gpuSuffix = gpu.backend !== 'cpu' ? `-${gpu.backend}` : '';
  const candidates = [
    `@node-omni-orcha/${platform}-${arch}${gpuSuffix}/${engine}.node`,
    `@node-omni-orcha/${platform}-${arch}/${engine}.node`,
  ];

  for (const candidate of candidates) {
    try {
      const binding = require(candidate) as NativeBinding;
      loadedBindings.set(engine, binding);
      return binding;
    } catch {
      // Not found, try next
    }
  }

  // Try local build
  const localPath = path.resolve(
    path.dirname(new URL(import.meta.url).pathname),
    '..',
    'build',
    'Release',
    `${engine}.node`,
  );

  if (existsSync(localPath)) {
    const binding = require(localPath) as NativeBinding;
    loadedBindings.set(engine, binding);
    return binding;
  }

  throw new Error(
    `Native binding "${engine}.node" not found. ` +
    `Install a platform package (@node-omni-orcha/${platform}-${arch}) ` +
    `or build from source: npm run build`,
  );
}
