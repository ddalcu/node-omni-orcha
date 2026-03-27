import { loadBinding } from '../binding-loader.ts';
import type {
  SttModel,
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

  // Mutex to serialize access — whisper contexts are not thread-safe.
  let busy: Promise<void> = Promise.resolve();
  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const prev = busy;
    let resolve: () => void;
    busy = new Promise<void>(r => { resolve = r; });
    return prev.then(fn).finally(() => resolve!());
  }

  const model: SttModel = {
    type: 'stt',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },

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
  };

  return model;
}
