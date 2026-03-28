import { loadBinding } from '../binding-loader.ts';
import type {
  SttModel,
  SttModelStatus,
  TranscribeOptions,
  TranscribeResult,
} from '../types.ts';

/**
 * Creates an SttModel instance for speech-to-text.
 * Expects a whisper GGUF model file.
 * Audio input should be 16-bit PCM at 16kHz mono.
 */
export function createSttModel(modelPath: string): SttModel {
  let nativeCtx: any = null;
  let loaded = false;
  let loading = false;
  let isBusy = false;

  // Mutex to serialize access — whisper contexts are not thread-safe.
  let busyPromise: Promise<void> = Promise.resolve();
  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const prev = busyPromise;
    let resolve: () => void;
    busyPromise = new Promise<void>(r => { resolve = r; });
    return prev.then(() => { isBusy = true; return fn(); }).finally(() => { isBusy = false; resolve!(); });
  }

  const model: SttModel = {
    type: 'stt',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },
    get loading() { return loading; },
    get busy() { return isBusy; },

    async load() {
      if (loaded || loading) return;
      loading = true;
      try {
        const binding = loadBinding();
        nativeCtx = await (binding['createSttContext'] as Function)(modelPath);
        loaded = true;
      } finally {
        loading = false;
      }
    },

    async transcribe(audio: Buffer, options?: TranscribeOptions): Promise<TranscribeResult> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return serialize(async () =>
        await (nativeCtx.transcribe as Function)(audio, {
          language: options?.language ?? 'auto',
        }) as TranscribeResult
      );
    },

    async detectLanguage(audio: Buffer): Promise<string> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return serialize(async () =>
        await (nativeCtx.detectLanguage as Function)(audio) as string
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

    getStatus(): SttModelStatus {
      return {
        type: 'stt',
        modelPath,
        loaded,
        loading,
        busy: isBusy,
      };
    },
  };

  return model;
}
