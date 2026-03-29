import { loadBinding } from '../binding-loader.ts';
import { detectGpu } from '../utils/gpu.ts';
import type {
  TtsModel,
  TtsModelStatus,
  TtsLoadOptions,
  SpeakOptions,
} from '../types.ts';

/**
 * Creates a TtsModel instance for text-to-speech via Qwen3-TTS.
 * modelPath should be the directory containing the Qwen3-TTS GGUF models.
 */
export function createTtsModel(modelPath: string): TtsModel {
  let nativeCtx: any = null;
  let loaded = false;
  let loading = false;
  let isBusy = false;

  // Mutex to serialize access — TTS contexts are not thread-safe.
  let busyPromise: Promise<void> = Promise.resolve();
  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const prev = busyPromise;
    let resolve: () => void;
    busyPromise = new Promise<void>(r => { resolve = r; });
    return prev.then(() => { isBusy = true; return fn(); }).finally(() => { isBusy = false; resolve!(); });
  }

  const model: TtsModel = {
    type: 'tts',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },
    get loading() { return loading; },
    get busy() { return isBusy; },

    async load(options?: TtsLoadOptions) {
      if (loaded || loading) return;
      loading = true;
      try {
        const binding = loadBinding();
        const gpu = detectGpu();
        const useGpu = options?.useGpu ?? (gpu.backend !== 'cpu');
        nativeCtx = await (binding['createTtsContext'] as Function)(modelPath, { useGpu });
        loaded = true;
      } finally {
        loading = false;
      }
    },

    async speak(text: string, options?: SpeakOptions): Promise<Buffer> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      const trimmed = text.trim();
      if (trimmed.length === 0) {
        throw new Error('Text must not be empty. Qwen3-TTS requires meaningful text input.');
      }

      // Convert maxDurationSeconds to audio frames.
      // Qwen3-TTS generates ~11.6 frames/second of audio (24kHz / 2048 hop).
      // Default 90s, hard cap at 180s to prevent runaway generation.
      const FRAMES_PER_SECOND = 11.6;
      const maxSeconds = Math.min(options?.maxDurationSeconds ?? 90, 180);
      const maxTokens = Math.ceil(maxSeconds * FRAMES_PER_SECOND);

      return serialize(async () =>
        await (nativeCtx.speak as Function)(text, {
          referenceAudioPath: options?.referenceAudioPath ?? '',
          temperature: options?.temperature ?? 0.9,
          topK: 50,
          maxTokens,
          repetitionPenalty: 1.05,
        }) as Buffer
      );
    },

    async unload() {
      if (!loaded || !nativeCtx) return;
      await serialize(async () => {
        if (!nativeCtx) return;
        await (nativeCtx.unload as Function)();
        nativeCtx = null;
        loaded = false;
      });
    },

    getStatus(): TtsModelStatus {
      return {
        type: 'tts',
        modelPath,
        loaded,
        loading,
        busy: isBusy,
      };
    },
  };

  return model;
}
