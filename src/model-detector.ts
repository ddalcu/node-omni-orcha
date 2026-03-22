import * as path from 'node:path';
import { readGGUFMetadata } from './utils/gguf-reader.ts';
import type { ModelType } from './types.ts';

// GGUF architectures that map to LLM
const LLM_ARCHITECTURES = new Set([
  'llama', 'mistral', 'gemma', 'gemma2', 'phi', 'phi2', 'phi3',
  'qwen', 'qwen2', 'qwen2moe', 'qwen3', 'qwen3moe',
  'deepseek', 'deepseek2',
  'starcoder', 'starcoder2',
  'gpt2', 'gptneox', 'gptj',
  'falcon', 'mpt', 'bloom', 'baichuan',
  'internlm', 'internlm2',
  'yi', 'orion', 'minicpm',
  'command-r', 'cohere',
  'olmo', 'olmoe',
  'dbrx', 'jais', 'chatglm',
  'exaone', 'arctic', 'nemotron',
  'rwkv', 'rwkv6', 'mamba',
  'granite', 'granitemo',
  'plamo', 'openelm', 'bitnet',
  't5', 't5encoder',
  'nomic-bert', 'bert', 'jina-bert-v2',
  'roberta',
]);

// GGUF architectures for STT
const STT_ARCHITECTURES = new Set(['whisper']);

// GGUF architectures for TTS
const TTS_ARCHITECTURES = new Set(['kokoro', 'parler', 'parler-tts', 'dia', 'orpheus']);

/**
 * Detect model type from file path.
 * For GGUF files, reads the architecture metadata key.
 * For safetensors, assumes image generation.
 */
export async function detectModelType(filePath: string): Promise<ModelType | null> {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === '.safetensors' || ext === '.ckpt') {
    return 'image';
  }

  if (ext === '.gguf') {
    const meta = await readGGUFMetadata(filePath);
    if (!meta) return null;

    const arch = meta.architecture.toLowerCase();

    if (LLM_ARCHITECTURES.has(arch)) return 'llm';
    if (STT_ARCHITECTURES.has(arch)) return 'stt';
    if (TTS_ARCHITECTURES.has(arch)) return 'tts';

    return null;
  }

  return null;
}
