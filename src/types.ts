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
  /** Enable embedding mode (required for embed/embedBatch). Default: false. */
  embeddings?: boolean;
  batchSize?: number;
  cacheTypeK?: 'f16' | 'q8_0' | 'q4_0';
  cacheTypeV?: 'f16' | 'q8_0' | 'q4_0';
  /** Override the chat template (Jinja format). Use when the model GGUF doesn't include one. */
  chatTemplate?: string;
  /** Path to multimodal projector GGUF (mmproj). Required for vision models. */
  mmprojPath?: string;
  /** Minimum tokens per image for dynamic-resolution vision models. Default: read from model metadata. */
  imageMinTokens?: number;
  /** Maximum tokens per image for dynamic-resolution vision models. Default: read from model metadata. */
  imageMaxTokens?: number;
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

/** A text content part in a multimodal message */
export interface TextContentPart {
  type: 'text';
  text: string;
}

/** An image content part — provide file bytes, a file path, or raw RGB pixels */
export interface ImageContentPart {
  type: 'image';
  /** Image file bytes (PNG, JPEG, BMP, GIF, WebP) */
  data?: Buffer;
  /** Path to an image file */
  path?: string;
  /** Raw RGB pixel data (3 bytes per pixel, row-major). Must also set width/height. */
  rgbData?: Buffer;
  /** Width in pixels (required when using rgbData) */
  width?: number;
  /** Height in pixels (required when using rgbData) */
  height?: number;
}

export type ContentPart = TextContentPart | ImageContentPart;

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  /** Text string, or array of content parts for multimodal messages */
  content: string | ContentPart[];
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
  /** Control reasoning/thinking. -1 = unlimited (default), 0 = disabled, N>0 = max N tokens of reasoning before responding. */
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
  readonly loading: boolean;
  readonly busy: boolean;
  readonly metadata: GGUFModelInfo | null;
  /** Whether this model has a vision encoder loaded (mmproj). */
  readonly hasVision: boolean;

  load(options?: LlmLoadOptions): Promise<void>;
  complete(messages: ChatMessage[], options?: CompletionOptions): Promise<CompletionResult>;
  stream(messages: ChatMessage[], options?: CompletionOptions): AsyncIterable<StreamChunk>;
  embed(text: string): Promise<Float64Array>;
  embedBatch(texts: string[]): Promise<Float64Array[]>;
  unload(): Promise<void>;
  getStatus(): LlmModelStatus;
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
  /** For WAN2.2 turbo: path to high-noise diffusion model */
  highNoiseDiffusionModelPath?: string;
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
  /** Init image for I2V / TI2V — PNG or JPEG buffer used as first frame */
  initImage?: Buffer;
  /** End image for FLF2V (first-last-frame-to-video) — PNG or JPEG buffer */
  endImage?: Buffer;
  /** High-noise sampling steps (WAN2.2 MoE / turbo) */
  highNoiseSteps?: number;
  /** High-noise CFG scale (WAN2.2 MoE / turbo) */
  highNoiseCfgScale?: number;
  /** High-noise sampling method (WAN2.2 MoE / turbo) */
  highNoiseSampleMethod?: SampleMethod;
}

export interface ImageModel {
  readonly type: 'image';
  readonly modelPath: string;
  readonly loaded: boolean;
  readonly loading: boolean;
  readonly busy: boolean;

  load(options?: ImageLoadOptions): Promise<void>;
  generate(prompt: string, options?: ImageOptions): Promise<Buffer>;
  /** Generate video frames. Returns array of PNG buffers (one per frame). */
  generateVideo(prompt: string, options?: VideoOptions): Promise<Buffer[]>;
  unload(): Promise<void>;
  getStatus(): ImageModelStatus;
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
  readonly loading: boolean;
  readonly busy: boolean;

  load(): Promise<void>;
  transcribe(audio: Buffer, options?: TranscribeOptions): Promise<TranscribeResult>;
  detectLanguage(audio: Buffer): Promise<string>;
  unload(): Promise<void>;
  getStatus(): SttModelStatus;
}

// --- TTS Types ---

export interface TtsLoadOptions {
  /** Reserved for future use */
}

export interface SpeakOptions {
  /** Path to reference audio for voice cloning — supports WAV, MP3, FLAC, OGG (24kHz mono recommended) */
  referenceAudioPath?: string;
  /** Sampling temperature (default 0.9) */
  temperature?: number;
  /** Maximum audio duration in seconds (default: 90, max: 180). Prevents runaway generation. */
  maxDurationSeconds?: number;
}

export interface TtsModel {
  readonly type: 'tts';
  readonly modelPath: string;
  readonly loaded: boolean;
  readonly loading: boolean;
  readonly busy: boolean;

  load(options?: TtsLoadOptions): Promise<void>;
  speak(text: string, options?: SpeakOptions): Promise<Buffer>;
  unload(): Promise<void>;
  getStatus(): TtsModelStatus;
}

// Union
export type Model = LlmModel | ImageModel | SttModel | TtsModel;

// Load options with type hint
export type LoadModelOptions = (LlmLoadOptions | ImageLoadOptions) & { type?: ModelType };

// --- Status Types ---

export interface CpuInfo {
  model: string;
  cores: number;
  threads: number;
  /** MHz */
  speed: number;
}

export interface MemoryInfo {
  totalBytes: number;
  usedBytes: number;
  freeBytes: number;
  usagePercent: number;
}

export interface ProcessMemoryInfo {
  rssBytes: number;
  heapTotalBytes: number;
  heapUsedBytes: number;
  externalBytes: number;
}

export interface SystemStatus {
  platform: string;
  arch: string;
  nodeVersion: string;
  hostname: string;
  cpu: CpuInfo;
  memory: MemoryInfo;
  processMemory: ProcessMemoryInfo;
  gpu: GpuInfo;
  /** Process uptime in seconds */
  uptimeSeconds: number;
  /** OS uptime in seconds */
  osUptimeSeconds: number;
}

export interface ModelStatus {
  type: ModelType;
  modelPath: string;
  loaded: boolean;
  loading: boolean;
  busy: boolean;
}

export interface LlmModelStatus extends ModelStatus {
  type: 'llm';
  metadata: GGUFModelInfo | null;
  hasVision: boolean;
}

export interface ImageModelStatus extends ModelStatus {
  type: 'image';
}

export interface SttModelStatus extends ModelStatus {
  type: 'stt';
}

export interface TtsModelStatus extends ModelStatus {
  type: 'tts';
}
