import { createLlmModel } from './llm/llm-model.ts';
import { createImageModel } from './image/image-model.ts';
import { createSttModel } from './stt/stt-model.ts';
import { createTtsModel } from './tts/tts-model.ts';
import { createKokoroModel } from './tts/kokoro-model.ts';
import type {
  Model,
  LlmModel,
  ImageModel,
  SttModel,
  TtsModel,
  KokoroModel,
  ModelType,
  LoadModelOptions,
} from './types.ts';

// Re-export all types
export type {
  Model,
  LlmModel,
  ImageModel,
  SttModel,
  TtsModel,
  KokoroModel,
  ModelType,
  LoadModelOptions,
  GGUFModelInfo,
  GpuInfo,
  ChatMessage,
  CompletionOptions,
  CompletionResult,
  StreamChunk,
  LlmLoadOptions,
  ToolDefinition,
  ToolCall,
  ContentPart,
  TextContentPart,
  ImageContentPart,
  ImageOptions,
  ImageLoadOptions,
  VideoOptions,
  SampleMethod,
  Scheduler,
  SttLoadOptions,
  TranscribeOptions,
  TranscribeResult,
  TranscribeSegment,
  TtsLoadOptions,
  SpeakOptions,
  KokoroLoadOptions,
  KokoroSpeakOptions,
  // Status types
  SystemStatus,
  CpuInfo,
  MemoryInfo,
  ProcessMemoryInfo,
  ModelStatus,
  LlmModelStatus,
  ImageModelStatus,
  SttModelStatus,
  TtsModelStatus,
  KokoroModelStatus,
} from './types.ts';

// Re-export utilities
export { readGGUFMetadata } from './utils/gguf-reader.ts';
export { detectGpu } from './utils/gpu.ts';
export { getSystemStatus } from './utils/system.ts';

/**
 * Load a model from a file path. Type is required.
 * Returns a loaded, ready-to-use model instance.
 */
export async function loadModel(filePath: string, options: LoadModelOptions & { type: 'llm' }): Promise<LlmModel>;
export async function loadModel(filePath: string, options: LoadModelOptions & { type: 'image' }): Promise<ImageModel>;
export async function loadModel(filePath: string, options: LoadModelOptions & { type: 'stt' }): Promise<SttModel>;
export async function loadModel(filePath: string, options: LoadModelOptions & { type: 'tts' }): Promise<TtsModel>;
export async function loadModel(filePath: string, options: LoadModelOptions & { type: 'kokoro' }): Promise<KokoroModel>;
export async function loadModel(filePath: string, options: LoadModelOptions & { type: ModelType }): Promise<Model> {
  switch (options.type) {
    case 'llm': {
      const model = createLlmModel(filePath);
      await model.load(options);
      return model;
    }
    case 'image': {
      const model = createImageModel(filePath);
      await model.load(options as any);
      return model;
    }
    case 'stt': {
      const model = createSttModel(filePath);
      await model.load(options);
      return model;
    }
    case 'tts': {
      const model = createTtsModel(filePath);
      await model.load(options);
      return model;
    }
    case 'kokoro': {
      const model = createKokoroModel(filePath);
      await model.load(options);
      return model;
    }
    default:
      throw new Error(`Unknown model type: ${(options as any).type}`);
  }
}

/**
 * Create a model instance without loading it.
 * Call .load() manually to load into memory.
 */
export function createModel(filePath: string, type: 'llm'): LlmModel;
export function createModel(filePath: string, type: 'image'): ImageModel;
export function createModel(filePath: string, type: 'stt'): SttModel;
export function createModel(filePath: string, type: 'tts'): TtsModel;
export function createModel(filePath: string, type: 'kokoro'): KokoroModel;
export function createModel(filePath: string, type: ModelType): Model;
export function createModel(filePath: string, type: ModelType): Model {
  switch (type) {
    case 'llm':
      return createLlmModel(filePath);
    case 'image':
      return createImageModel(filePath);
    case 'stt':
      return createSttModel(filePath);
    case 'tts':
      return createTtsModel(filePath);
    case 'kokoro':
      return createKokoroModel(filePath);
    default:
      throw new Error(`Unknown model type: ${type}`);
  }
}
