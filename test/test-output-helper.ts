import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const OUTPUT_DIR = join(import.meta.dirname, '..', 'test-output');

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

function sanitize(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]/g, '_').replace(/_+/g, '_').slice(0, 80);
}

/**
 * Build a descriptive output filename and write to test-output/.
 * Returns the full path written.
 *
 * Example:
 *   saveTestOutput('image', 'flux2-klein-4b-Q4_K_M', { width: 256, height: 256, steps: 4 }, pngBuffer, '.png')
 *   → test-output/image_flux2-klein-4b-Q4_K_M_256x256_4steps_2026-03-24T08-30-15.png
 */
export function saveTestOutput(
  engine: string,
  model: string,
  params: Record<string, string | number | boolean | undefined>,
  data: Buffer | string,
  ext: string,
): string {
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const paramParts: string[] = [];
  for (const [key, val] of Object.entries(params)) {
    if (val === undefined || val === '' || val === false) continue;
    if (key === 'width' && params['height'] != null) {
      paramParts.push(`${val}x${params['height']}`);
    } else if (key === 'height') {
      continue; // handled by width
    } else if (key === 'steps' || key === 'sampleSteps') {
      paramParts.push(`${val}steps`);
    } else if (key === 'videoFrames' || key === 'frames') {
      paramParts.push(`${val}f`);
    } else if (key === 'cfgScale' || key === 'cfg') {
      paramParts.push(`cfg${val}`);
    } else if (key === 'temperature' || key === 'temp') {
      paramParts.push(`t${val}`);
    } else if (key === 'maxTokens') {
      paramParts.push(`${val}tok`);
    } else if (key === 'language' || key === 'lang') {
      paramParts.push(`${val}`);
    } else if (key === 'voice') {
      paramParts.push(`${sanitize(String(val))}`);
    } else if (key === 'seed') {
      paramParts.push(`s${val}`);
    } else if (key === 'sampleMethod') {
      paramParts.push(`${val}`);
    } else if (typeof val === 'string') {
      paramParts.push(sanitize(val));
    } else if (typeof val === 'number') {
      paramParts.push(`${key}${val}`);
    }
  }

  const modelSlug = sanitize(model);
  const paramsSlug = paramParts.length > 0 ? '_' + paramParts.join('_') : '';
  const ts = timestamp();

  const filename = `${engine}_${modelSlug}${paramsSlug}_${ts}${ext}`;
  const fullPath = join(OUTPUT_DIR, filename);

  writeFileSync(fullPath, data);
  return fullPath;
}

/**
 * Save video frames to test-output/ and optionally encode MP4.
 * Returns the directory path.
 */
export function saveVideoFrames(
  model: string,
  params: Record<string, string | number | boolean | undefined>,
  frames: Buffer[],
): string {
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const paramParts: string[] = [];
  if (params['width'] && params['height']) paramParts.push(`${params['width']}x${params['height']}`);
  if (params['videoFrames'] || params['frames']) paramParts.push(`${params['videoFrames'] ?? params['frames']}f`);
  if (params['steps']) paramParts.push(`${params['steps']}steps`);
  if (params['cfgScale']) paramParts.push(`cfg${params['cfgScale']}`);
  if (params['seed'] != null) paramParts.push(`s${params['seed']}`);

  const modelSlug = sanitize(model);
  const paramsSlug = paramParts.length > 0 ? '_' + paramParts.join('_') : '';
  const ts = timestamp();

  const dirName = `video_${modelSlug}${paramsSlug}_${ts}`;
  const outDir = join(OUTPUT_DIR, dirName);
  mkdirSync(outDir, { recursive: true });

  for (let i = 0; i < frames.length; i++) {
    writeFileSync(join(outDir, `frame_${String(i).padStart(4, '0')}.png`), frames[i]!);
  }

  // Encode MP4 with ffmpeg (WAN 2.2 generates at ~16fps)
  try {
    const { execSync } = require('node:child_process') as typeof import('node:child_process');
    const mp4Path = join(OUTPUT_DIR, `${dirName}.mp4`);
    execSync(
      `ffmpeg -y -framerate 16 -i "${outDir}/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "${mp4Path}"`,
      { stdio: 'pipe' },
    );
    const { statSync } = require('node:fs') as typeof import('node:fs');
    console.log(`    MP4: ${mp4Path} (${(statSync(mp4Path).size / 1024).toFixed(0)}KB)`);
  } catch {
    console.log('    MP4: skipped (ffmpeg not found — install with: winget install Gyan.FFmpeg)');
  }

  return outDir;
}
