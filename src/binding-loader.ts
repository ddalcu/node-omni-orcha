import { createRequire } from 'node:module';
import * as path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { detectGpu } from './utils/gpu.ts';

const require = createRequire(import.meta.url);

interface NativeBinding {
  [key: string]: (...args: unknown[]) => unknown;
}

let cachedBinding: NativeBinding | null = null;

/**
 * Load the unified omni.node native addon.
 * Resolution order:
 *   1. Custom bindings directory (NODE_OMNI_ORCHA_BINDINGS env var — for SEA / standalone)
 *   2. CUDA-specific platform package (@agent-orcha/node-omni-orcha-{platform}-{arch}-cuda)
 *   3. Default platform package (@agent-orcha/node-omni-orcha-{platform}-{arch})
 *   4. Local cmake-js build (./build/Release/omni.node)
 */
export function loadBinding(): NativeBinding {
  if (cachedBinding) return cachedBinding;

  const platform = process.platform;
  const arch = process.arch;
  const gpu = detectGpu();

  // 1. Custom bindings directory (SEA / standalone deployments)
  const bindingsDir = process.env.NODE_OMNI_ORCHA_BINDINGS;
  if (bindingsDir) {
    const customPath = path.join(bindingsDir, 'omni.node');
    if (existsSync(customPath)) {
      cachedBinding = require(customPath) as NativeBinding;
      return cachedBinding;
    }
  }

  // 2-3. Platform npm packages (CUDA-specific first, then default)
  const candidates: string[] = [];
  if (gpu.backend === 'cuda') {
    candidates.push(`@agent-orcha/node-omni-orcha-${platform}-${arch}-cuda/omni.node`);
  }
  candidates.push(`@agent-orcha/node-omni-orcha-${platform}-${arch}/omni.node`);

  for (const candidate of candidates) {
    try {
      cachedBinding = require(candidate) as NativeBinding;
      return cachedBinding;
    } catch {
      // Not found, try next
    }
  }

  // 4. Local cmake-js build (development)
  const localPath = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    '..',
    'build',
    'Release',
    'omni.node',
  );

  if (existsSync(localPath)) {
    cachedBinding = require(localPath) as NativeBinding;
    return cachedBinding;
  }

  throw new Error(
    `Native binding "omni.node" not found. ` +
    `Install a platform package (@agent-orcha/node-omni-orcha-${platform}-${arch}) ` +
    `or build from source: npm run build`,
  );
}
