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

  const model: SttModel = {
    type: 'stt',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },

    async load() {
      if (loaded) return;

      const binding = loadBinding('stt');
      nativeCtx = await (binding['createContext'] as Function)(modelPath);
      loaded = true;
    },

    async transcribe(audio: Buffer, options?: TranscribeOptions): Promise<TranscribeResult> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return await (nativeCtx.transcribe as Function)(audio, {
        language: options?.language ?? 'auto',
      }) as TranscribeResult;
    },

    async detectLanguage(audio: Buffer): Promise<string> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return await (nativeCtx.detectLanguage as Function)(audio) as string;
    },

    async unload() {
      if (!loaded || !nativeCtx) return;
      await (nativeCtx.unload as Function)();
      nativeCtx = null;
      loaded = false;
    },
  };

  return model;
}
