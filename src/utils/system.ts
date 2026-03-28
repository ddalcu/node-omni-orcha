import * as os from 'node:os';
import { execFileSync } from 'node:child_process';
import { detectGpu } from './gpu.ts';
import type { SystemStatus } from '../types.ts';

function getPhysicalCores(): number {
  try {
    if (process.platform === 'darwin') {
      return parseInt(execFileSync('sysctl', ['-n', 'hw.physicalcpu'], { encoding: 'utf-8' }).trim(), 10);
    }
    if (process.platform === 'linux') {
      const out = execFileSync('lscpu', { encoding: 'utf-8' });
      const match = out.match(/^Core\(s\) per socket:\s+(\d+)/m);
      const sockets = out.match(/^Socket\(s\):\s+(\d+)/m);
      if (match && sockets) return parseInt(match[1]!, 10) * parseInt(sockets[1]!, 10);
    }
  } catch { /* fall through */ }
  return os.cpus().length;
}

/**
 * Returns a snapshot of system status: CPU, memory, GPU, process info.
 */
export function getSystemStatus(): SystemStatus {
  const cpus = os.cpus();
  const cpu0 = cpus[0];
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const usedMem = totalMem - freeMem;
  const mem = process.memoryUsage();

  return {
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    hostname: os.hostname(),
    cpu: {
      model: cpu0?.model?.trim() ?? 'unknown',
      cores: getPhysicalCores(),
      threads: cpus.length,
      speed: cpu0?.speed ?? 0,
    },
    memory: {
      totalBytes: totalMem,
      usedBytes: usedMem,
      freeBytes: freeMem,
      usagePercent: Math.round((usedMem / totalMem) * 10000) / 100,
    },
    processMemory: {
      rssBytes: mem.rss,
      heapTotalBytes: mem.heapTotal,
      heapUsedBytes: mem.heapUsed,
      externalBytes: mem.external,
    },
    gpu: detectGpu(),
    uptimeSeconds: Math.floor(process.uptime()),
    osUptimeSeconds: Math.floor(os.uptime()),
  };
}
