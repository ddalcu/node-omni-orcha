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

  const model: TtsModel = {
    type: 'tts',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },

    async load(_options?: TtsLoadOptions) {
      if (loaded) return;

      const binding = loadBinding();
      nativeCtx = await (binding['createTtsContext'] as Function)(modelPath, {});
      loaded = true;
    },

    async speak(text: string, options?: SpeakOptions): Promise<Buffer> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return await (nativeCtx.speak as Function)(text, {
        referenceAudioPath: options?.referenceAudioPath ?? '',
        temperature: options?.temperature ?? 0.9,
        topK: 50,
        maxTokens: 4096,
        repetitionPenalty: 1.05,
      }) as Buffer;
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
