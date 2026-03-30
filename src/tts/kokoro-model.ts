import { loadBinding } from '../binding-loader.ts';
import { detectGpu } from '../utils/gpu.ts';
import type {
  KokoroModel,
  KokoroModelStatus,
  KokoroLoadOptions,
  KokoroSpeakOptions,
} from '../types.ts';

/**
 * Creates a KokoroModel instance for fast TTS via Kokoro-82M (ONNX Runtime).
 * modelPath should be the directory containing:
 *   - kokoro-v1.0.fp16.onnx  (the ONNX model)
 *   - voices.bin              (converted voice embeddings)
 *   - phoneme_dict.bin        (CMU phoneme dictionary)
 */
export function createKokoroModel(modelPath: string): KokoroModel {
  let nativeCtx: any = null;
  let loaded = false;
  let loading = false;
  let isBusy = false;
  let voiceList: string[] = [];

  // Mutex to serialize access — not thread-safe.
  let busyPromise: Promise<void> = Promise.resolve();
  function serialize<T>(fn: () => Promise<T>): Promise<T> {
    const prev = busyPromise;
    let resolve: () => void;
    busyPromise = new Promise<void>(r => { resolve = r; });
    return prev.then(() => { isBusy = true; return fn(); }).finally(() => { isBusy = false; resolve!(); });
  }

  // Resolve file paths within the model directory
  function resolveModelFile(name: string): string {
    return modelPath.endsWith('/') ? modelPath + name : modelPath + '/' + name;
  }

  const model: KokoroModel = {
    type: 'kokoro',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },
    get loading() { return loading; },
    get busy() { return isBusy; },

    async load(options?: KokoroLoadOptions) {
      if (loaded || loading) return;
      loading = true;
      try {
        const binding = loadBinding();

        if (!binding['createKokoroContext']) {
          throw new Error(
            'Kokoro TTS not available in this build. ' +
            'Rebuild with ONNX Runtime (bash scripts/download-onnxruntime.sh) ' +
            'and espeak-ng (brew install espeak-ng).'
          );
        }

        const gpu = detectGpu();
        const useGpu = options?.useGpu ?? (gpu.backend !== 'cpu');

        const onnxPath = resolveModelFile('kokoro-v1.0.fp16.onnx');
        const voicesPath = resolveModelFile('voices.bin');
        const dictPath = resolveModelFile('phoneme_dict.bin');

        nativeCtx = await (binding['createKokoroContext'] as Function)(
          onnxPath, voicesPath, dictPath, { useGpu }
        );

        // Cache the voice list
        voiceList = (nativeCtx.listVoices as Function)() as string[];
        loaded = true;
      } finally {
        loading = false;
      }
    },

    async speak(text: string, options?: KokoroSpeakOptions): Promise<Buffer> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      const trimmed = text.trim();
      if (trimmed.length === 0) {
        throw new Error('Text must not be empty.');
      }

      return serialize(async () =>
        await (nativeCtx.speak as Function)(text, {
          voice: options?.voice ?? '',
          speed: options?.speed ?? 1.0,
        }) as Buffer
      );
    },

    listVoices(): string[] {
      return [...voiceList];
    },

    async unload() {
      if (!loaded || !nativeCtx) return;
      await serialize(async () => {
        if (!nativeCtx) return;
        await (nativeCtx.unload as Function)();
        nativeCtx = null;
        loaded = false;
        voiceList = [];
      });
    },

    getStatus(): KokoroModelStatus {
      return {
        type: 'kokoro',
        modelPath,
        loaded,
        loading,
        busy: isBusy,
        voices: [...voiceList],
      };
    },
  };

  return model;
}
