import { loadBinding } from '../binding-loader.ts';
import type {
  ImageModel,
  ImageLoadOptions,
  ImageOptions,
  VideoOptions,
} from '../types.ts';

/**
 * Creates an ImageModel instance.
 * Supports both standard SD models (single file) and FLUX (multi-file).
 *
 * For FLUX, provide the diffusion model path as modelPath, plus:
 *   clipLPath, t5xxlPath, vaePath in load options.
 */
export function createImageModel(modelPath: string): ImageModel {
  let nativeCtx: any = null;
  let loaded = false;

  const model: ImageModel = {
    type: 'image',
    get modelPath() { return modelPath; },
    get loaded() { return loaded; },

    async load(options?: ImageLoadOptions) {
      if (loaded) return;

      const binding = loadBinding('image');

      nativeCtx = await (binding['createContext'] as Function)(modelPath, {
        clipLPath: options?.clipLPath ?? '',
        t5xxlPath: options?.t5xxlPath ?? '',
        llmPath: options?.llmPath ?? '',
        vaePath: options?.vaePath ?? '',
        threads: options?.threads ?? -1,
        keepVaeOnCpu: options?.keepVaeOnCpu ?? false,
        offloadToCpu: options?.offloadToCpu ?? false,
        flashAttn: options?.flashAttn ?? true,
        vaeDecodeOnly: options?.vaeDecodeOnly ?? true,
      });

      loaded = true;
    },

    async generate(prompt: string, options?: ImageOptions): Promise<Buffer> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return await (nativeCtx.generate as Function)(prompt, {
        width: options?.width ?? 512,
        height: options?.height ?? 512,
        steps: options?.steps ?? 20,
        cfgScale: options?.cfgScale ?? 7.0,
        seed: options?.seed ?? -1,
        negativePrompt: options?.negativePrompt ?? '',
        sampleMethod: options?.sampleMethod ?? 'euler',
        scheduler: options?.scheduler,
        clipSkip: options?.clipSkip ?? -1,
      }) as Buffer;
    },

    async generateVideo(prompt: string, options?: VideoOptions): Promise<Buffer[]> {
      if (!loaded || !nativeCtx) {
        throw new Error('Model not loaded. Call load() first.');
      }

      return await (nativeCtx.generateVideo as Function)(prompt, {
        width: options?.width ?? 832,
        height: options?.height ?? 480,
        videoFrames: options?.videoFrames ?? 33,
        steps: options?.steps ?? 30,
        cfgScale: options?.cfgScale ?? 6.0,
        flowShift: options?.flowShift ?? 3.0,
        seed: options?.seed ?? -1,
        negativePrompt: options?.negativePrompt ?? '',
        sampleMethod: options?.sampleMethod ?? 'euler',
        scheduler: options?.scheduler,
        clipSkip: options?.clipSkip ?? -1,
      }) as Buffer[];
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
