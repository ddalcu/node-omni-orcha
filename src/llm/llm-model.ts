import { loadBinding } from '../binding-loader.ts';
import { readGGUFMetadata, calculateOptimalContextSize } from '../utils/gguf-reader.ts';
import { detectGpu } from '../utils/gpu.ts';
import type {
  LlmModel,
  LlmModelStatus,
  LlmLoadOptions,
  ChatMessage,
  CompletionOptions,
  CompletionResult,
  StreamChunk,
  GGUFModelInfo,
} from '../types.ts';

// ─── Stream token sanitizer ───
// Some models (Gemma 4) emit special tokens in streaming that span multiple chunks.
// This state machine suppresses tool call blocks and channel markers from stream output.
// The actual tool calls are parsed post-generation and delivered in the final done chunk.

class StreamSanitizer {
  private inToolCall = false;
  private inChannel = false;

  /** Process a text chunk — returns the cleaned text to emit (may be empty). */
  process(text: string): string {
    // Strip standalone single-token markers
    let out = text.replaceAll('<|"|>', '');

    let result = '';
    for (const ch of out) {
      if (this.inToolCall) {
        // Suppress everything until we see the closing marker
        // <tool_call|> arrives as a single token, so check after accumulation
        if (out.includes('<tool_call|>')) {
          this.inToolCall = false;
          return ''; // entire chunk is part of tool call block
        }
        return ''; // still inside tool call, suppress
      }
      if (this.inChannel) {
        if (ch === '\n') {
          this.inChannel = false;
          continue; // swallow the newline that ends the channel marker
        }
        continue; // suppress channel marker content
      }
      result += ch;
    }

    // Check for markers that START in this chunk
    if (result.includes('<|tool_call>')) {
      this.inToolCall = true;
      // Return anything before the marker
      return result.slice(0, result.indexOf('<|tool_call>'));
    }
    if (result.includes('<|channel>')) {
      this.inChannel = true;
      return result.slice(0, result.indexOf('<|channel>'));
    }

    return result;
  }

  /** Flush any remaining state — call before the final done chunk. */
  flush(): string {
    this.inToolCall = false;
    this.inChannel = false;
    return '';
  }
}

/**
 * Creates an LlmModel instance for the given GGUF file.
 * The model is NOT loaded into memory — call load() first.
 */
export function createLlmModel(modelPath: string): LlmModel {
  let nativeCtx: any = null;
  let loaded = false;
  let loading = false;
  let isBusy = false;
  let metadata: GGUFModelInfo | null = null;

  // Mutex to serialize access — llama.cpp contexts are not thread-safe.
  // Concurrent operations on the same context corrupt KV cache state and cause segfaults.
  let busyPromise: Promise<void> = Promise.resolve();
  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const prev = busyPromise;
    let resolve: () => void;
    busyPromise = new Promise<void>(r => { resolve = r; });
    return prev.then(() => { isBusy = true; return fn(); }).finally(() => { isBusy = false; resolve!(); });
  }

  const model: LlmModel = {
    type: 'llm',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },
    get loading() { return loading; },
    get busy() { return isBusy; },
    get metadata() { return metadata; },
    get hasVision() { return loaded && nativeCtx?.hasVision?.() === true; },

    async load(options?: LlmLoadOptions) {
      if (loaded || loading) return;
      loading = true;

      try {
        const binding = loadBinding();
        metadata = await readGGUFMetadata(modelPath);

        const gpu = detectGpu();
        const contextSize = options?.contextSize
          ?? (metadata ? calculateOptimalContextSize(metadata) : 4096);
        const isEmbedding = options?.embeddings ?? false;
        const useGpu = options?.useGpu ?? (gpu.backend !== 'cpu');
        // Embedding models (e.g. nomic-bert) are small and fast on CPU — default to 0 GPU layers
        // to avoid competing for VRAM with the main LLM. Users can still override with gpuLayers.
        const defaultGpuLayers = !useGpu ? 0 : (isEmbedding ? 0 : -1);
        const requestedGpuLayers = options?.gpuLayers ?? defaultGpuLayers;
        const totalLayers = metadata?.blockCount ?? 0;

        const buildOpts = (gpuLayers: number) => ({
          contextSize,
          gpuLayers,
          flashAttn: options?.flashAttn ?? useGpu,
          embeddings: options?.embeddings ?? false,
          batchSize: options?.batchSize ?? (useGpu ? 4096 : 512),
          cacheTypeK: options?.cacheTypeK ?? (gpu.backend === 'metal' && useGpu ? 'q8_0' : 'f16'),
          cacheTypeV: options?.cacheTypeV ?? (gpu.backend === 'metal' && useGpu ? 'q8_0' : 'f16'),
          chatTemplate: options?.chatTemplate ?? '',
          ...(options?.mmprojPath ? { mmprojPath: options.mmprojPath } : {}),
          ...(options?.imageMinTokens != null ? { imageMinTokens: options.imageMinTokens } : {}),
          ...(options?.imageMaxTokens != null ? { imageMaxTokens: options.imageMaxTokens } : {}),
        });

        // When gpuLayers is -1 (all), retry with fewer layers on CUDA OOM
        // so the model spills to system RAM instead of crashing.
        let gpuLayers = requestedGpuLayers;
        if (gpu.backend !== 'cpu' && gpuLayers === -1 && totalLayers > 0) {
          // Try all layers first, then binary-search down on OOM
          let hi = totalLayers + 1; // +1 for output layer
          let lo = 0;
          let lastError: Error | null = null;

          // First attempt: all layers
          try {
            nativeCtx = await (binding['createLlmContext'] as Function)(modelPath, buildOpts(-1));
            loaded = true;
            return;
          } catch (err: any) {
            if (!isOomError(err)) throw err;
            lastError = err;
            console.warn(`[omni] CUDA OOM with all ${hi} GPU layers, reducing...`);
          }

          // Binary search for max layers that fit
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            try {
              nativeCtx = await (binding['createLlmContext'] as Function)(modelPath, buildOpts(mid));
              // Succeeded — try more layers
              lo = mid + 1;
              // Keep this context as the best so far
              break;
            } catch (err: any) {
              if (!isOomError(err)) throw err;
              lastError = err;
              hi = mid;
            }
          }

          if (nativeCtx) {
            console.warn(`[omni] Loaded with ${lo > 0 ? lo - 1 : 0}/${totalLayers} GPU layers (remaining on CPU)`);
            loaded = true;
            return;
          }

          // Last resort: CPU only
          try {
            console.warn('[omni] Falling back to CPU-only (0 GPU layers)');
            nativeCtx = await (binding['createLlmContext'] as Function)(modelPath, buildOpts(0));
            loaded = true;
            return;
          } catch (err: any) {
            throw lastError ?? err;
          }
        }

        nativeCtx = await (binding['createLlmContext'] as Function)(modelPath, buildOpts(gpuLayers));
        loaded = true;
      } finally {
        loading = false;
      }
    },

    async complete(messages: ChatMessage[], options?: CompletionOptions): Promise<CompletionResult> {
      assertLoaded();
      return serialize(async () => {
        const nativeOpts: Record<string, unknown> = {
          temperature: options?.temperature ?? 0.7,
          maxTokens: options?.maxTokens ?? 2048,
          stopSequences: options?.stopSequences ?? [],
        };

        // Serialize tools for C++ (parameters must be JSON string)
        if (options?.tools?.length) {
          nativeOpts.tools = options.tools.map(t => ({
            name: t.name,
            description: t.description,
            parameters: JSON.stringify(t.parameters),
          }));
          nativeOpts.toolChoice = options.toolChoice ?? 'auto';
        }

        if (options?.thinkingBudget != null && options.thinkingBudget >= 0) {
          nativeOpts.thinkingBudget = options.thinkingBudget;
        }

        const result = await (nativeCtx.complete as Function)(
          formatMessages(messages),
          nativeOpts,
        );
        return result as CompletionResult;
      });
    },

    async *stream(messages: ChatMessage[], options?: CompletionOptions): AsyncIterable<StreamChunk> {
      assertLoaded();

      // Acquire the mutex before starting the stream — hold it until stream completes.
      const prev = busyPromise;
      let releaseMutex: () => void;
      busyPromise = new Promise<void>(r => { releaseMutex = r; });
      await prev;
      isBusy = true;

      const nativeOpts: Record<string, unknown> = {
        temperature: options?.temperature ?? 0.7,
        maxTokens: options?.maxTokens ?? 2048,
        stopSequences: options?.stopSequences ?? [],
      };

      if (options?.tools?.length) {
        nativeOpts.tools = options.tools.map(t => ({
          name: t.name,
          description: t.description,
          parameters: JSON.stringify(t.parameters),
        }));
        nativeOpts.toolChoice = options.toolChoice ?? 'auto';
      }

      if (options?.thinkingBudget != null && options.thinkingBudget >= 0) {
        nativeOpts.thinkingBudget = options.thinkingBudget;
      }

      // Buffer chunks from the native callback, yield them as an async iterable
      const pending: StreamChunk[] = [];
      let notify: (() => void) | null = null;
      let streamDone = false;
      let streamError: Error | null = null;

      const onChunk = (chunk: StreamChunk) => {
        pending.push(chunk);
        if (notify) { notify(); notify = null; }
      };

      const streamPromise = (nativeCtx.stream as Function)(
        formatMessages(messages),
        nativeOpts,
        onChunk,
      ).catch((err: Error) => {
        // If the native worker errors (e.g. template failure), wake the consumer
        // so it doesn't hang forever waiting for chunks that will never arrive.
        streamError = err;
        if (notify) { notify(); notify = null; }
      });

      const contentSanitizer = new StreamSanitizer();
      const reasoningSanitizer = new StreamSanitizer();

      try {
        while (!streamDone) {
          if (pending.length === 0) {
            await new Promise<void>(r => { notify = r; });
          }
          if (streamError) throw streamError;
          while (pending.length > 0) {
            const chunk = pending.shift()!;

            if (chunk.done) {
              contentSanitizer.flush();
              reasoningSanitizer.flush();
              yield chunk;
              streamDone = true;
              break;
            }

            // Sanitize content and reasoning streams to strip model-specific special tokens
            if (chunk.content) {
              const clean = contentSanitizer.process(chunk.content);
              if (clean) yield { content: clean, done: false } as StreamChunk;
            } else if (chunk.reasoning) {
              const clean = reasoningSanitizer.process(chunk.reasoning);
              if (clean) yield { reasoning: clean, done: false } as StreamChunk;
            } else {
              yield chunk;
            }

            if (options?.signal?.aborted) {
              (nativeCtx.abort as Function)?.();
              streamDone = true;
              break;
            }
          }
        }
      } finally {
        await streamPromise;
        isBusy = false;
        releaseMutex!();
      }
    },

    async embed(text: string): Promise<Float64Array> {
      assertLoaded();
      return serialize(async () =>
        await (nativeCtx.embed as Function)(text) as Float64Array
      );
    },

    async embedBatch(texts: string[]): Promise<Float64Array[]> {
      assertLoaded();
      return serialize(async () =>
        await (nativeCtx.embedBatch as Function)(texts) as Float64Array[]
      );
    },

    async unload() {
      if (!loaded || !nativeCtx) return;
      // Wait for any in-flight operation to complete before destroying the context.
      // Destroying while an AsyncWorker is using the context causes a segfault.
      await serialize(async () => {
        if (!nativeCtx) return;
        await (nativeCtx.unload as Function)();
        nativeCtx = null;
        loaded = false;
        metadata = null;
      });
    },

    getStatus(): LlmModelStatus {
      return {
        type: 'llm',
        modelPath,
        loaded,
        loading,
        busy: isBusy,
        metadata,
        hasVision: loaded && nativeCtx?.hasVision?.() === true,
      };
    },
  };

  function assertLoaded(): void {
    if (!loaded || !nativeCtx) {
      throw new Error('Model not loaded. Call load() first.');
    }
  }

  return model;
}

/**
 * Detect CUDA out-of-memory errors from the native binding.
 */
function isOomError(err: unknown): boolean {
  const msg = String((err as Error)?.message ?? err).toLowerCase();
  return msg.includes('out of memory') || msg.includes('cuda error') || msg.includes('alloc');
}

/**
 * Format ChatMessage[] into the structure expected by the C++ binding.
 * Content can be a string or array of content parts for multimodal messages.
 */
function formatMessages(messages: ChatMessage[]): Array<Record<string, unknown>> {
  return messages.map(m => ({
    role: m.role,
    content: m.content, // string or ContentPart[] — C++ ParseMultimodalMessages handles both
    ...(m.tool_call_id ? { tool_call_id: m.tool_call_id } : {}),
    ...(m.name ? { name: m.name } : {}),
    ...(m.tool_calls?.length ? { tool_calls: m.tool_calls } : {}),
  }));
}
