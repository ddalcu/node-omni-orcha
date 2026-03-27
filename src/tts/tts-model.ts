import { loadBinding } from '../binding-loader.ts';
import type {
  TtsModel,
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

  // Mutex to serialize access — TTS contexts are not thread-safe.
  let busy: Promise<void> = Promise.resolve();
  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const prev = busy;
    let resolve: () => void;
    busy = new Promise<void>(r => { resolve = r; });
    return prev.then(fn).finally(() => resolve!());
  }

  const model: TtsModel = {
    type: 'tts',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },

    async load(_options?: TtsLoadOptions) {
      if (loaded || loading) return;
      loading = true;
      try {
        const binding = loadBinding();
        nativeCtx = await (binding['createTtsContext'] as Function)(modelPath, {});
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
  };

  return model;
}
