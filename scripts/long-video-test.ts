/**
 * Long video generation — generates a 60-second video by chaining 9-frame clips.
 * Uses the last frame of each clip as the init image for the next clip (TI2V).
 *
 * At 16fps, 60 seconds = 960 frames.
 * Each clip generates 9 frames, with 1 frame overlap → 8 new frames per clip.
 * Total clips needed: ceil(960 / 8) = 120 clips.
 *
 * Usage:
 *   node scripts/long-video-test.ts                    # 60s video
 *   node scripts/long-video-test.ts --duration 10      # 10s video
 *   node scripts/long-video-test.ts --duration 5 --fps 8   # 5s at 8fps
 */

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import * as path from 'node:path';
import { createModel } from '../src/index.ts';
import type { ImageModel } from '../src/types.ts';

const MODELS_DIR = process.env['MODELS_DIR'] || `${process.env['HOME'] || process.env['USERPROFILE']}/.orcha/workspace/.models`;
const OUTPUT_DIR = path.resolve(import.meta.dirname!, '..', 'test-output');

// Parse CLI args
function getArg(name: string, defaultValue: number): number {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx !== -1 && process.argv[idx + 1]) return Number(process.argv[idx + 1]);
  return defaultValue;
}

const TARGET_DURATION = getArg('duration', 60);
const FPS = getArg('fps', 16);
const FRAMES_PER_CLIP = 9;
const OVERLAP = 1; // last frame of clip N = init image for clip N+1
const NEW_FRAMES_PER_CLIP = FRAMES_PER_CLIP - OVERLAP;
const TOTAL_FRAMES = TARGET_DURATION * FPS;
const TOTAL_CLIPS = Math.ceil(TOTAL_FRAMES / NEW_FRAMES_PER_CLIP);

function getPromptArg(): string {
  const idx = process.argv.indexOf('--prompt');
  if (idx !== -1 && process.argv[idx + 1]) return process.argv[idx + 1]!;
  return 'a red sports car driving through a beautiful mountain road at sunset, cinematic, smooth motion';
}
const PROMPT = getPromptArg();

const WIDTH = 832;
const HEIGHT = 480;
const STEPS = 20;
const CFG_SCALE = 3.5;
const FLOW_SHIFT = 3.0;
const SEED = 42;

// Model paths
const modelPath = path.join(MODELS_DIR, 'wan22-5b', 'Wan2.2-TI2V-5B-Q4_K_M.gguf');
const vaePath = path.join(MODELS_DIR, 'wan22-5b', 'Wan2.2_VAE.safetensors');
const t5Path = path.join(MODELS_DIR, 'wan22-5b', 'umt5-xxl-encoder-Q8_0.gguf');

if (!existsSync(modelPath) || !existsSync(vaePath) || !existsSync(t5Path)) {
  console.error('TI2V-5B model files not found. Run: bash scripts/download-test-models.sh --video');
  process.exit(1);
}

console.log(`
Long Video Generation
=====================
Duration:     ${TARGET_DURATION}s at ${FPS}fps
Total frames: ${TOTAL_FRAMES}
Clips:        ${TOTAL_CLIPS} x ${FRAMES_PER_CLIP} frames (${OVERLAP} overlap)
Resolution:   ${WIDTH}x${HEIGHT}
Steps:        ${STEPS}, CFG: ${CFG_SCALE}
Prompt:       "${PROMPT.slice(0, 80)}..."
`);

const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const outDir = path.join(OUTPUT_DIR, `longvideo_${WIDTH}x${HEIGHT}_${TARGET_DURATION}s_${timestamp}`);
mkdirSync(outDir, { recursive: true });

console.log('Starting clip generation (fresh context per clip to avoid sd.cpp reuse crash)...\n');

const allFrames: Buffer[] = [];
let initImage: Buffer | undefined;
const startTime = Date.now();

for (let clip = 0; clip < TOTAL_CLIPS; clip++) {
  const clipStart = Date.now();
  const framesNeeded = Math.min(TOTAL_FRAMES - allFrames.length + OVERLAP, FRAMES_PER_CLIP);

  if (framesNeeded <= OVERLAP) break;

  // sd.cpp context reuse crashes on second generation — create fresh context per clip
  const clipModel = createModel(modelPath, 'image') as ImageModel;
  await clipModel.load({ t5xxlPath: t5Path, vaePath: vaePath, flashAttn: true, vaeDecodeOnly: true });

  const params: Record<string, unknown> = {
    width: WIDTH, height: HEIGHT,
    videoFrames: FRAMES_PER_CLIP,
    steps: STEPS, cfgScale: CFG_SCALE,
    flowShift: FLOW_SHIFT, seed: SEED + clip,
    ...(initImage ? { initImage } : {}),
  };

  const mode = initImage ? 'TI2V (continuation)' : 'T2V (first clip)';
  process.stdout.write(`  Clip ${clip + 1}/${TOTAL_CLIPS} [${mode}]...`);

  const frames = await clipModel.generateVideo(PROMPT, params as any);

  await clipModel.unload();

  const clipElapsed = ((Date.now() - clipStart) / 1000).toFixed(1);
  const skipFirst = (clip > 0 && initImage) ? OVERLAP : 0;

  // Save frames
  for (let i = skipFirst; i < frames.length && allFrames.length < TOTAL_FRAMES; i++) {
    const frameIdx = allFrames.length;
    writeFileSync(
      path.join(outDir, `frame_${String(frameIdx).padStart(5, '0')}.png`),
      frames[i]!
    );
    allFrames.push(frames[i]!);
  }

  // Use last frame as init for next clip
  initImage = frames[frames.length - 1];

  const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  const progress = ((allFrames.length / TOTAL_FRAMES) * 100).toFixed(1);
  console.log(` ${frames.length} frames in ${clipElapsed}s | ${allFrames.length}/${TOTAL_FRAMES} total (${progress}%) | ${totalElapsed}s elapsed`);
}

const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
console.log(`\nDone: ${allFrames.length} frames in ${totalElapsed}s`);
console.log(`Frames: ${outDir}`);

// Encode MP4 with ffmpeg
try {
  const { execSync } = await import('node:child_process');
  const mp4Path = `${outDir}.mp4`;

  // Try system ffmpeg first, then winget location
  const ffmpegCandidates = [
    'ffmpeg',
    'C:/Users/cyber/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe',
  ];

  let ffmpeg = '';
  for (const candidate of ffmpegCandidates) {
    try {
      execSync(`"${candidate}" -version`, { stdio: 'pipe' });
      ffmpeg = candidate;
      break;
    } catch { /* try next */ }
  }

  if (ffmpeg) {
    console.log(`\nEncoding MP4 at ${FPS}fps...`);
    execSync(
      `"${ffmpeg}" -y -framerate ${FPS} -i "${outDir}/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "${mp4Path}"`,
      { stdio: 'pipe' },
    );
    const { statSync } = await import('node:fs');
    const sizeMB = (statSync(mp4Path).size / (1024 * 1024)).toFixed(1);
    console.log(`MP4: ${mp4Path} (${sizeMB}MB)`);
  } else {
    console.log('\nMP4: skipped (ffmpeg not found)');
  }
} catch (err) {
  console.log(`\nMP4 encoding failed: ${(err as Error).message}`);
}

console.log(`\nStats:`);
console.log(`  Total time:    ${totalElapsed}s`);
console.log(`  Frames:        ${allFrames.length}`);
console.log(`  Duration:      ${(allFrames.length / FPS).toFixed(1)}s at ${FPS}fps`);
console.log(`  Avg per clip:  ${(parseFloat(totalElapsed) / TOTAL_CLIPS).toFixed(1)}s`);
console.log(`  Avg per frame: ${(parseFloat(totalElapsed) / allFrames.length).toFixed(2)}s`);
