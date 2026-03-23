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
 *   1. Custom bindings directory (NODE_OMNI_ORCHA_BINDINGS env var — for SEA / standalone)
 *   2. CUDA-specific platform package (@agent-orcha/node-omni-orcha-{platform}-{arch}-cuda)
 *   3. Default platform package (@agent-orcha/node-omni-orcha-{platform}-{arch})
 *   4. Local cmake-js build (./build/Release/{engine}.node)
 */
export function loadBinding(engine: string): NativeBinding {
  const cached = loadedBindings.get(engine);
  if (cached) return cached;

  const platform = process.platform;
  const arch = process.arch;
  const gpu = detectGpu();

  // 1. Custom bindings directory (SEA / standalone deployments)
  const bindingsDir = process.env.NODE_OMNI_ORCHA_BINDINGS;
  if (bindingsDir) {
    const customPath = path.join(bindingsDir, `${engine}.node`);
    if (existsSync(customPath)) {
      const binding = require(customPath) as NativeBinding;
      loadedBindings.set(engine, binding);
      return binding;
    }
  }

  // 2-3. Platform npm packages (CUDA-specific first, then default)
  const candidates: string[] = [];
  if (gpu.backend === 'cuda') {
    candidates.push(`@agent-orcha/node-omni-orcha-${platform}-${arch}-cuda/${engine}.node`);
  }
  candidates.push(`@agent-orcha/node-omni-orcha-${platform}-${arch}/${engine}.node`);

  for (const candidate of candidates) {
    try {
      const binding = require(candidate) as NativeBinding;
      loadedBindings.set(engine, binding);
      return binding;
    } catch {
      // Not found, try next
    }
  }

  // 4. Local cmake-js build (development)
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
    `Install a platform package (@agent-orcha/node-omni-orcha-${platform}-${arch}) ` +
    `or build from source: npm run build`,
  );
}
