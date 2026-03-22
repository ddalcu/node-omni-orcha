// Model types
export const MODEL_TYPES = {
  llm: 'llm',
  image: 'image',
  stt: 'stt',
  tts: 'tts',
} as const;

export type ModelType = (typeof MODEL_TYPES)[keyof typeof MODEL_TYPES];

// GPU
export interface GpuInfo {
  backend: 'metal' | 'cuda' | 'cpu';
  name?: string;
  vramBytes?: number;
}

// GGUF metadata
export interface GGUFModelInfo {
  architecture: string;
  contextLength: number;
  blockCount: number;
  embeddingLength: number;
  headCount: number;
  headCountKv: number;
  fileSizeBytes: number;
}

// --- LLM Types ---

export interface LlmLoadOptions {
  contextSize?: number;
  gpuLayers?: number;
  flashAttn?: boolean;
  batchSize?: number;
  cacheTypeK?: 'f16' | 'q8_0' | 'q4_0';
  cacheTypeV?: 'f16' | 'q8_0' | 'q4_0';
  /** Override the chat template (Jinja format). Use when the model GGUF doesn't include one. */
  chatTemplate?: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>; // JSON Schema object
}

export interface ToolCall {
  id: string;
  name: string;
  args: string; // JSON string of arguments
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_call_id?: string;
  name?: string;
  tool_calls?: ToolCall[];
}

export interface CompletionOptions {
  temperature?: number;
  maxTokens?: number;
  stopSequences?: string[];
  signal?: AbortSignal;
  tools?: ToolDefinition[];
  toolChoice?: 'auto' | 'required' | 'none';
  /** Max tokens the model can spend on reasoning/thinking before being forced to respond. -1 = unlimited (default). */
  thinkingBudget?: number;
}

export interface CompletionResult {
  content: string;
  reasoning?: string;
  toolCalls?: ToolCall[];
  usage: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
}

export interface StreamChunk {
  content?: string;
  reasoning?: string;
  toolCalls?: ToolCall[];
  usage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  done: boolean;
}

export interface LlmModel {
  readonly type: 'llm';
  readonly modelPath: string;
  readonly loaded: boolean;
  readonly metadata: GGUFModelInfo | null;

  load(options?: LlmLoadOptions): Promise<void>;
  complete(messages: ChatMessage[], options?: CompletionOptions): Promise<CompletionResult>;
  stream(messages: ChatMessage[], options?: CompletionOptions): AsyncIterable<StreamChunk>;
  embed(text: string): Promise<Float64Array>;
  embedBatch(texts: string[]): Promise<Float64Array[]>;
  unload(): Promise<void>;
}

// --- Image Types ---

export const SAMPLE_METHODS = {
  euler: 'euler',
  euler_a: 'euler_a',
  heun: 'heun',
  dpm2: 'dpm2',
  dpmpp2s_a: 'dpmpp2s_a',
  dpmpp2m: 'dpmpp2m',
  dpmpp2mv2: 'dpmpp2mv2',
  lcm: 'lcm',
} as const;

export type SampleMethod = (typeof SAMPLE_METHODS)[keyof typeof SAMPLE_METHODS];

export const SCHEDULERS = {
  discrete: 'discrete',
  karras: 'karras',
  exponential: 'exponential',
  ays: 'ays',
  sgm_uniform: 'sgm_uniform',
  simple: 'simple',
} as const;

export type Scheduler = (typeof SCHEDULERS)[keyof typeof SCHEDULERS];

export interface ImageLoadOptions {
  /** For FLUX.1: path to CLIP-L text encoder */
  clipLPath?: string;
  /** For FLUX.1/WAN: path to T5-XXL or UMT5 text encoder */
  t5xxlPath?: string;
  /** For FLUX.2: path to LLM text encoder (Qwen3 4B for Klein, Mistral for Dev) */
  llmPath?: string;
  /** Path to VAE model (required for FLUX/WAN, optional for SD) */
  vaePath?: string;
  /** Number of threads (defaults to physical core count) */
  threads?: number;
  /** Keep VAE on CPU to save VRAM */
  keepVaeOnCpu?: boolean;
  /** Offload model params to CPU (useful for large models on limited VRAM) */
  offloadToCpu?: boolean;
  /** Enable flash attention */
  flashAttn?: boolean;
  /** Only load VAE decoder (default: true). Set to false for video generation or image-to-video. */
  vaeDecodeOnly?: boolean;
}

export interface ImageOptions {
  width?: number;
  height?: number;
  steps?: number;
  seed?: number;
  /** CFG scale — use 1.0 for FLUX, 7.0 for SD (defaults based on model) */
  cfgScale?: number;
  /** Negative prompt (not used by FLUX) */
  negativePrompt?: string;
  /** Sampling method (defaults to euler) */
  sampleMethod?: SampleMethod;
  /** Scheduler (auto-detected from model if not specified) */
  scheduler?: Scheduler;
  /** CLIP skip layers */
  clipSkip?: number;
}

export interface VideoOptions {
  width?: number;
  height?: number;
  /** Number of video frames to generate (default: 33) */
  videoFrames?: number;
  steps?: number;
  seed?: number;
  /** CFG scale (default: 6.0 for WAN) */
  cfgScale?: number;
  /** Flow shift for WAN models (default: 3.0) */
  flowShift?: number;
  /** Negative prompt */
  negativePrompt?: string;
  /** Sampling method (defaults to euler) */
  sampleMethod?: SampleMethod;
  /** Scheduler */
  scheduler?: Scheduler;
  /** CLIP skip layers */
  clipSkip?: number;
}

export interface ImageModel {
  readonly type: 'image';
  readonly modelPath: string;
  readonly loaded: boolean;

  load(options?: ImageLoadOptions): Promise<void>;
  generate(prompt: string, options?: ImageOptions): Promise<Buffer>;
  /** Generate video frames. Returns array of PNG buffers (one per frame). */
  generateVideo(prompt: string, options?: VideoOptions): Promise<Buffer[]>;
  unload(): Promise<void>;
}

// --- STT Types ---

export interface TranscribeOptions {
  language?: string;
}

export interface TranscribeSegment {
  start: number;
  end: number;
  text: string;
}

export interface TranscribeResult {
  text: string;
  language: string;
  segments: TranscribeSegment[];
}

export interface SttModel {
  readonly type: 'stt';
  readonly modelPath: string;
  readonly loaded: boolean;

  load(): Promise<void>;
  transcribe(audio: Buffer, options?: TranscribeOptions): Promise<TranscribeResult>;
  detectLanguage(audio: Buffer): Promise<string>;
  unload(): Promise<void>;
}

// --- TTS Types ---

export const TTS_ENGINES = {
  kokoro: 'kokoro',
  qwen3: 'qwen3',
} as const;

export type TtsEngine = (typeof TTS_ENGINES)[keyof typeof TTS_ENGINES];

export interface TtsLoadOptions {
  /** TTS engine: 'kokoro' (Kokoro/Parler/Dia) or 'qwen3' (Qwen3-TTS with voice cloning) */
  engine?: TtsEngine;
}

export interface SpeakOptions {
  /** Voice name — for Kokoro: 'af_heart', 'am_adam', etc. */
  voice?: string;
  speed?: number;
  /** Path to reference audio WAV for voice cloning (Qwen3 engine only, 24kHz mono recommended) */
  referenceAudioPath?: string;
  /** Sampling temperature (Qwen3: default 0.9, Kokoro: default 1.0) */
  temperature?: number;
}

export interface TtsModel {
  readonly type: 'tts';
  readonly modelPath: string;
  readonly loaded: boolean;

  readonly engine: TtsEngine;

  load(options?: TtsLoadOptions): Promise<void>;
  speak(text: string, options?: SpeakOptions): Promise<Buffer>;
  unload(): Promise<void>;
}

// Union
export type Model = LlmModel | ImageModel | SttModel | TtsModel;

// Load options with type hint
export type LoadModelOptions = (LlmLoadOptions | ImageLoadOptions) & { type?: ModelType };
