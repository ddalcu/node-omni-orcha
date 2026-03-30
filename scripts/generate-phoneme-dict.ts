#!/usr/bin/env node
/**
 * Downloads CMU Pronouncing Dictionary and converts it to a compact binary
 * format with pre-computed Kokoro token IDs.
 *
 * Usage:
 *   node scripts/generate-phoneme-dict.ts <output_path>
 *
 * Binary format (phoneme_dict.bin):
 *   char[4]    magic ("PHON")
 *   uint16_t   version (1)
 *   uint32_t   num_entries
 *   Entry[]    entries (sorted by word)
 *
 * Entry:
 *   uint8_t    word_len
 *   char[]     word (lowercase ASCII, no null terminator)
 *   uint8_t    num_tokens
 *   int16_t[]  token_ids (Kokoro vocabulary IDs)
 */

const CMUDICT_URL =
  'https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict';

// ─── Kokoro vocabulary (must match engine/kokoro/vocab.h) ───

const SYMBOLS: string[] = [
  '$',
  // punctuation + space (1-16)
  ';', ':', ',', '.', '!', '?', '\u00A1', '\u00BF', '\u2014', '\u2026',
  '\u201C', '\u00AB', '\u00BB', '\u201D', '"', ' ',
  // A-Z (17-42)
  ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
  // a-z (43-68)
  ...'abcdefghijklmnopqrstuvwxyz'.split(''),
  // IPA (69+)
  '\u0251','\u0250','\u0252','\u00E6','\u0253','\u0299','\u03B2','\u0254',
  '\u0255','\u00E7','\u0257','\u0256','\u00F0','\u02A4','\u0259','\u0258',
  '\u025A','\u025B','\u025C','\u025D','\u025E','\u025F','\u0284','\u0261',
  '\u0260','\u0262','\u029B','\u0266','\u0267','\u0127','\u0265','\u029C',
  '\u0268','\u026A','\u029D','\u026D','\u026C','\u026B','\u026E','\u029F',
  '\u0271','\u026F','\u0270','\u014B','\u0273','\u0272','\u0274','\u00F8',
  '\u0275','\u0278','\u03B8','\u0153','\u0276','\u0298','\u0279','\u027A',
  '\u027E','\u027B','\u0280','\u0281','\u027D','\u0282','\u0283','\u0288',
  '\u02A7','\u0289','\u028A','\u028B','\u2C71','\u028C','\u0263','\u0264',
  '\u028D','\u03C7','\u028E','\u028F','\u0291','\u0290','\u0292','\u0294',
  '\u02A1','\u0295','\u02A2','\u01C0','\u01C1','\u01C2','\u01C3','\u02C8',
  '\u02CC','\u02D0','\u02D1','\u02BC','\u02B4','\u02B0','\u02B1','\u02B2',
  '\u02B7','\u02E0','\u02E4','\u02DE','\u2193','\u2191','\u2192','\u2197',
  '\u2198','\u0027','\u0303','\u0308','\u0304','\u0300','\u030A','\u1D7B',
];

const symbolToId = new Map<string, number>();
for (let i = 0; i < SYMBOLS.length; i++) {
  symbolToId.set(SYMBOLS[i], i);
}

function ipaToTokenIds(ipa: string): number[] {
  const ids: number[] = [];
  for (const ch of ipa) {
    const id = symbolToId.get(ch);
    if (id !== undefined) ids.push(id);
  }
  return ids;
}

// ─── ARPABET → IPA mapping ───
// Maps ARPABET phonemes (without stress digits) to IPA characters.
// Stress (0/1/2) is handled separately by prepending ˈ or ˌ before vowels.

// ARPABET → IPA mapping tuned to match espeak-ng American English output,
// which is what Kokoro-82M was trained on.
const ARPABET_TO_IPA: Record<string, string> = {
  // Vowels — include length marker ː for tense/long vowels (espeak-ng style)
  'AA': '\u0251\u02D0',   // ɑː (long open back)
  'AE': '\u00E6',         // æ
  'AH': '\u0259',         // ə (unstressed default; stressed → ʌ)
  'AO': '\u0254\u02D0',   // ɔː (long open-mid back rounded)
  'AW': 'a\u028A',        // aʊ
  'AY': 'a\u026A',        // aɪ
  'EH': '\u025B',         // ɛ
  'ER': '\u0259\u0279',   // əɹ (unstressed: schwa + approximant, espeak-ng style)
  'EY': 'e\u026A',        // eɪ
  'IH': '\u026A',         // ɪ
  'IY': 'i\u02D0',        // iː (long close front)
  'OW': 'o\u028A',        // oʊ
  'OY': '\u0254\u026A',   // ɔɪ
  'UH': '\u028A',         // ʊ
  'UW': 'u\u02D0',        // uː (long close back)

  // Consonants (no stress)
  'B':  'b',
  'CH': '\u02A7',          // ʧ
  'D':  'd',
  'DH': '\u00F0',          // ð
  'F':  'f',
  'G':  '\u0261',          // ɡ (IPA g)
  'HH': 'h',
  'JH': '\u02A4',          // ʤ
  'K':  'k',
  'L':  'l',
  'M':  'm',
  'N':  'n',
  'NG': '\u014B',          // ŋ
  'P':  'p',
  'R':  '\u0279',          // ɹ
  'S':  's',
  'SH': '\u0283',          // ʃ
  'T':  't',
  'TH': '\u03B8',          // θ
  'V':  'v',
  'W':  'w',
  'Y':  'j',               // IPA j (palatal approximant)
  'Z':  'z',
  'ZH': '\u0292',          // ʒ
};

// Vowel phonemes that carry stress markers
const VOWELS = new Set([
  'AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW',
]);

function arpabetToIpa(phonemes: string[]): string {
  // First pass: convert each phoneme to IPA with inline stress markers
  const parts: Array<{ ipa: string; isVowel: boolean; stress: number }> = [];

  for (const ph of phonemes) {
    const stressMatch = ph.match(/^([A-Z]+)(\d)?$/);
    if (!stressMatch) continue;

    const base = stressMatch[1];
    const stress = stressMatch[2] ? parseInt(stressMatch[2]) : -1;
    const isVowel = VOWELS.has(base);

    let ipa: string;
    if (base === 'AH' && stress >= 1) {
      ipa = '\u028C'; // ʌ (stressed AH)
    } else if (base === 'ER' && stress >= 1) {
      ipa = '\u025C\u02D0\u0279'; // ɜːɹ (stressed ER — espeak-ng style)
    } else {
      ipa = ARPABET_TO_IPA[base] ?? '';
    }

    parts.push({ ipa, isVowel, stress });
  }

  // Second pass: place stress markers at syllable onset (before the preceding
  // consonant cluster) rather than immediately before the vowel.
  // This matches espeak-ng behavior: "hello" → həˈloʊ not həlˈoʊ
  let result = '';
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];

    if (p.isVowel && (p.stress === 1 || p.stress === 2)) {
      const stressMark = p.stress === 1 ? '\u02C8' : '\u02CC'; // ˈ or ˌ

      // Look back to find the start of the onset consonant cluster
      // (consecutive consonants immediately before this vowel)
      let onsetStart = i;
      for (let j = i - 1; j >= 0; j--) {
        if (parts[j].isVowel) break;
        onsetStart = j;
      }

      if (onsetStart < i && onsetStart > 0) {
        // Insert stress before the consonant onset
        // We need to rebuild the result string — find where the onset IPA starts
        // by computing character positions
        let charsBeforeOnset = 0;
        for (let j = 0; j < onsetStart; j++) charsBeforeOnset += parts[j].ipa.length;
        result = result.slice(0, charsBeforeOnset) + stressMark + result.slice(charsBeforeOnset);
      } else {
        // Word-initial vowel or no preceding consonant — stress before vowel
        result += stressMark;
      }
    }

    result += p.ipa;
  }

  return result;
}

// ─── Main ───

async function main() {
  const outputPath = process.argv[2];
  if (!outputPath) {
    console.error('Usage: node scripts/generate-phoneme-dict.ts <output_path>');
    process.exit(1);
  }

  // Download CMU dict
  console.log('Downloading CMU Pronouncing Dictionary...');
  const response = await fetch(CMUDICT_URL);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  const text = await response.text();

  // Parse entries
  const entries: Array<{ word: string; tokens: number[] }> = [];
  let skipped = 0;

  for (const line of text.split('\n')) {
    // Skip comments and empty lines
    if (!line || line.startsWith(';;;')) continue;

    // cmudict.dict format: word PH1 PH2 PH3 ...
    // cmudict-0.7b format: WORD  PH1 PH2 PH3 ...
    // Variant pronunciations: word(2) PH1 PH2 ...
    const match = line.match(/^([a-zA-Z'][a-zA-Z'.-]*)(?:\(\d+\))?\s+(.+)$/);
    if (!match) continue;

    const word = match[1].toLowerCase();
    const phonemes = match[2].trim().split(/\s+/);

    // Convert to IPA
    const ipa = arpabetToIpa(phonemes);
    if (!ipa) { skipped++; continue; }

    // Convert IPA to Kokoro token IDs
    const tokens = ipaToTokenIds(ipa);
    if (tokens.length === 0) { skipped++; continue; }

    // Only keep first pronunciation variant for each word
    if (entries.length > 0 && entries[entries.length - 1].word === word) continue;

    entries.push({ word, tokens });
  }

  // Sort by word
  entries.sort((a, b) => a.word < b.word ? -1 : a.word > b.word ? 1 : 0);

  console.log(`Parsed ${entries.length} entries (${skipped} skipped)`);

  // Write binary file
  const parts: Buffer[] = [];

  // Header
  const header = Buffer.alloc(10);
  header.write('PHON', 0, 4, 'ascii');
  header.writeUInt16LE(1, 4);           // version
  header.writeUInt32LE(entries.length, 6);
  parts.push(header);

  // Entries
  for (const entry of entries) {
    const wordBuf = Buffer.from(entry.word, 'utf-8');
    const wordLenBuf = Buffer.alloc(1);
    wordLenBuf.writeUInt8(wordBuf.length, 0);
    parts.push(wordLenBuf);
    parts.push(wordBuf);

    const tokenBuf = Buffer.alloc(1 + entry.tokens.length * 2);
    tokenBuf.writeUInt8(entry.tokens.length, 0);
    for (let i = 0; i < entry.tokens.length; i++) {
      tokenBuf.writeInt16LE(entry.tokens[i], 1 + i * 2);
    }
    parts.push(tokenBuf);
  }

  const { writeFileSync } = await import('node:fs');
  const output = Buffer.concat(parts);
  writeFileSync(outputPath, output);

  console.log(`Written ${output.length} bytes to ${outputPath}`);
  console.log(`  ${entries.length} words, avg ${(output.length / entries.length).toFixed(1)} bytes/entry`);
}

main().catch(e => { console.error(e); process.exit(1); });
