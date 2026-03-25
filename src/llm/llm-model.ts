import { loadBinding } from '../binding-loader.ts';
import { readGGUFMetadata, calculateOptimalContextSize } from '../utils/gguf-reader.ts';
import { detectGpu } from '../utils/gpu.ts';
import type {
  LlmModel,
  LlmLoadOptions,
  ChatMessage,
  CompletionOptions,
  CompletionResult,
  StreamChunk,
  GGUFModelInfo,
} from '../types.ts';

/**
 * Creates an LlmModel instance for the given GGUF file.
 * The model is NOT loaded into memory — call load() first.
 */
export function createLlmModel(modelPath: string): LlmModel {
  let nativeCtx: any = null;
  let loaded = false;
  let metadata: GGUFModelInfo | null = null;

  const model: LlmModel = {
    type: 'llm',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },
    get metadata() { return metadata; },

    async load(options?: LlmLoadOptions) {
      if (loaded) return;

      const binding = loadBinding();
      metadata = await readGGUFMetadata(modelPath);

      const gpu = detectGpu();
      const contextSize = options?.contextSize
        ?? (metadata ? calculateOptimalContextSize(metadata) : 4096);
      const gpuLayers = options?.gpuLayers ?? (gpu.backend !== 'cpu' ? -1 : 0);

      nativeCtx = await (binding['createLlmContext'] as Function)(modelPath, {
        contextSize,
        gpuLayers,
        flashAttn: options?.flashAttn ?? (gpu.backend !== 'cpu'),
        embeddings: options?.embeddings ?? false,
        batchSize: options?.batchSize ?? (gpu.backend !== 'cpu' ? 4096 : 512),
        cacheTypeK: options?.cacheTypeK ?? (gpu.backend === 'metal' ? 'q8_0' : 'f16'),
        cacheTypeV: options?.cacheTypeV ?? (gpu.backend === 'metal' ? 'q8_0' : 'f16'),
        chatTemplate: options?.chatTemplate ?? '',
      });

      loaded = true;
    },

    async complete(messages: ChatMessage[], options?: CompletionOptions): Promise<CompletionResult> {
      assertLoaded();

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
    },

    async *stream(messages: ChatMessage[], options?: CompletionOptions): AsyncIterable<StreamChunk> {
      assertLoaded();

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

      const onChunk = (chunk: StreamChunk) => {
        pending.push(chunk);
        if (notify) { notify(); notify = null; }
      };

      const streamPromise = (nativeCtx.stream as Function)(
        formatMessages(messages),
        nativeOpts,
        onChunk,
      );

      try {
        while (!streamDone) {
          if (pending.length === 0) {
            await new Promise<void>(r => { notify = r; });
          }
          while (pending.length > 0) {
            const chunk = pending.shift()!;
            yield chunk;
            if (chunk.done) { streamDone = true; break; }
            if (options?.signal?.aborted) {
              (nativeCtx.abort as Function)?.();
              streamDone = true;
              break;
            }
          }
        }
      } finally {
        await streamPromise;
      }
    },

    async embed(text: string): Promise<Float64Array> {
      assertLoaded();
      return await (nativeCtx.embed as Function)(text) as Float64Array;
    },

    async embedBatch(texts: string[]): Promise<Float64Array[]> {
      assertLoaded();
      return await (nativeCtx.embedBatch as Function)(texts) as Float64Array[];
    },

    async unload() {
      if (!loaded || !nativeCtx) return;
      await (nativeCtx.unload as Function)();
      nativeCtx = null;
      loaded = false;
      metadata = null;
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
 * Format ChatMessage[] into the structure expected by the C++ binding.
 */
function formatMessages(messages: ChatMessage[]): Array<Record<string, unknown>> {
  return messages.map(m => ({
    role: m.role,
    content: m.content,
    ...(m.tool_call_id ? { tool_call_id: m.tool_call_id } : {}),
    ...(m.name ? { name: m.name } : {}),
    ...(m.tool_calls?.length ? { tool_calls: m.tool_calls } : {}),
  }));
}
