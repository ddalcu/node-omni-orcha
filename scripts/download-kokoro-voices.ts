#!/usr/bin/env node
/**
 * Downloads Kokoro voice .bin files from HuggingFace and packs them
 * into a single voices.bin for the C++ engine.
 *
 * Usage:
 *   node scripts/download-kokoro-voices.ts <output_dir>
 *
 * Each voice .bin file is raw float32[510][256] (522,240 bytes).
 *
 * Output format (voices.bin):
 *   uint32_t  num_voices
 *   uint32_t  num_frames (510)
 *   uint32_t  style_dim (256)
 *   For each voice (sorted by name):
 *     uint16_t  name_len
 *     char[]    name (utf-8, no null terminator)
 *     float32[] embedding (num_frames * style_dim floats)
 */

const BASE_URL =
  'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices';

const VOICE_NAMES = [
  'af','af_alloy','af_aoede','af_bella','af_heart','af_jessica','af_kore',
  'af_nicole','af_nova','af_river','af_sarah','af_sky',
  'am_adam','am_echo','am_eric','am_fenrir','am_liam','am_michael',
  'am_onyx','am_puck','am_santa',
  'bf_alice','bf_emma','bf_isabella','bf_lily',
  'bm_daniel','bm_fable','bm_george','bm_lewis',
  'ef_dora','em_alex','em_santa',
  'ff_siwis',
  'hf_alpha','hf_beta','hm_omega','hm_psi',
  'if_sara','im_nicola',
  'jf_alpha','jf_gongitsune','jf_nezumi','jf_tebukuro','jm_kumo',
  'pf_dora','pm_alex','pm_santa',
  'zf_xiaobei','zf_xiaoni','zf_xiaoxiao',
];

const NUM_FRAMES = 510;
const STYLE_DIM = 256;
const EXPECTED_SIZE = NUM_FRAMES * STYLE_DIM * 4; // 522,240 bytes

async function main() {
  const outputDir = process.argv[2];
  if (!outputDir) {
    console.error('Usage: node scripts/download-kokoro-voices.ts <output_dir>');
    process.exit(1);
  }

  const { existsSync, writeFileSync, mkdirSync } = await import('node:fs');
  const outputPath = outputDir + '/voices.bin';

  if (existsSync(outputPath)) {
    console.log(`Already exists: ${outputPath}`);
    return;
  }

  // Download individual voice files
  const voicesDir = outputDir + '/voices_raw';
  mkdirSync(voicesDir, { recursive: true });

  const voices: Array<{ name: string; data: Buffer }> = [];

  for (const name of VOICE_NAMES) {
    const filePath = `${voicesDir}/${name}.bin`;
    let data: Buffer;

    if (existsSync(filePath)) {
      const { readFileSync } = await import('node:fs');
      data = readFileSync(filePath);
    } else {
      const url = `${BASE_URL}/${name}.bin`;
      process.stdout.write(`Downloading ${name}...`);
      const res = await fetch(url);
      if (!res.ok) {
        console.log(` FAILED (${res.status})`);
        continue;
      }
      data = Buffer.from(await res.arrayBuffer());
      writeFileSync(filePath, data);
      console.log(` ${data.length} bytes`);
    }

    if (data.length !== EXPECTED_SIZE) {
      console.warn(`  WARNING: ${name} size ${data.length} != expected ${EXPECTED_SIZE}, skipping`);
      continue;
    }

    voices.push({ name, data });
  }

  voices.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  console.log(`\nPacking ${voices.length} voices...`);

  // Write packed voices.bin
  const parts: Buffer[] = [];

  // Header: num_voices, num_frames, style_dim
  const header = Buffer.alloc(12);
  header.writeUInt32LE(voices.length, 0);
  header.writeUInt32LE(NUM_FRAMES, 4);
  header.writeUInt32LE(STYLE_DIM, 8);
  parts.push(header);

  for (const voice of voices) {
    const nameBuf = Buffer.from(voice.name, 'utf-8');
    const nameHeader = Buffer.alloc(2);
    nameHeader.writeUInt16LE(nameBuf.length, 0);
    parts.push(nameHeader);
    parts.push(nameBuf);
    parts.push(voice.data);
  }

  const output = Buffer.concat(parts);
  writeFileSync(outputPath, output);
  console.log(`Written ${output.length} bytes to ${outputPath}`);
  console.log(`  ${voices.length} voices, ${(output.length / 1024 / 1024).toFixed(1)} MB`);
}

main().catch(e => { console.error(e); process.exit(1); });
