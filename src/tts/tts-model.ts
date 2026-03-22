import { loadBinding } from '../binding-loader.ts';
import type {
  TtsModel,
  TtsLoadOptions,
  SpeakOptions,
  TtsEngine,
} from '../types.ts';

/**
 * Creates a TtsModel instance for text-to-speech.
 *
 * Two engines supported:
 *   - 'kokoro' (default): Kokoro, Parler, Dia, Orpheus models via TTS.cpp
 *   - 'qwen3': Qwen3-TTS with voice cloning support
 *
 * For Qwen3: modelPath should be the directory containing the GGUF models.
 * For Kokoro: modelPath should be the .gguf model file path.
 */
export function createTtsModel(modelPath: string): TtsModel {
  let nativeCtx: any = null;
  let loaded = false;
  let engine: TtsEngine = 'kokoro';

  const model: TtsModel = {
    type: 'tts',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },
    get engine() { return engine; },

    async load(options?: TtsLoadOptions) {
      if (loaded) return;

      engine = options?.engine ?? 'kokoro';

      if (engine === 'qwen3') {
        const binding = loadBinding('tts_qwen3');
        nativeCtx = await (binding['createContext'] as Function)(modelPath, {});
      } else {
        const binding = loadBinding('tts');
        nativeCtx = await (binding['createContext'] as Function)(modelPath, {});
      }

      loaded = true;
    },

    async speak(text: string, options?: SpeakOptions): Promise<Buffer> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      if (engine === 'qwen3') {
        return await (nativeCtx.speak as Function)(text, {
          referenceAudioPath: options?.referenceAudioPath ?? '',
          temperature: options?.temperature ?? 0.9,
          topK: 50,
          maxTokens: 4096,
          repetitionPenalty: 1.05,
        }) as Buffer;
      }

      // Kokoro engine
      return await (nativeCtx.speak as Function)(text, {
        voice: options?.voice ?? '',
        speed: options?.speed ?? 1.0,
        temperature: options?.temperature ?? 1.0,
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
