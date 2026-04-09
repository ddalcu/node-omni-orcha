# node-omni-orcha

Unified native Node.js inference engine — **LLM**, **Vision**, **Image/Video Generation**, and **Text-to-Speech** in a single `omni.node` binary.

Built on a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork (synced to [`f49e917`](https://github.com/ggml-org/llama.cpp/commit/f49e917)) with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) compiled against a shared ggml backend.

## Features

- **LLM** — Chat completion, streaming, embeddings, tool calling, reasoning budget control
- **Vision** — Multimodal LLM with image input (Gemma 4, Qwen2-VL, Qwen3-VL, LLaVA, Gemma3, etc.)
- **Image Generation** — FLUX 2, Wan 2.2, SD/SDXL (text-to-image)
- **Video Generation** — Wan 2.2 (text-to-video, image-to-video)
- **Text-to-Speech** — Qwen3-TTS (voice cloning)
- **GPU accelerated** — Metal (macOS), CUDA (NVIDIA), CPU fallback
- **Single binary** — One `omni.node` for all engines, one shared ggml, no symbol conflicts
- **Native N-API** — In-process inference, no child processes or HTTP servers
- **Node 25** — Native TypeScript, ESM, no build step
- **Web UI** — Built-in server with chat, TTS, image/video generation (for testing only)

## Quick Start

```bash
npm install node-omni-orcha

npm run build:metal   # macOS
npm run build:cuda    # NVIDIA
npm run build:cpu     # CPU only
```

## API

```ts
import { loadModel, createModel, detectGpu, readGGUFMetadata } from 'node-omni-orcha'
```

### LLM

```ts
const llm = await loadModel('qwen3.5-4b.gguf', { type: 'llm', contextSize: 4096 })

// With reasoning (default)
const result = await llm.complete([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Hello!' },
], { temperature: 0.7, maxTokens: 512 })

console.log(result.content)    // response text
console.log(result.reasoning)  // thinking/reasoning (if model supports it)
console.log(result.usage)      // { inputTokens, outputTokens, totalTokens }

// Without reasoning (direct response, faster)
const fast = await llm.complete(messages, { thinkingBudget: 0, maxTokens: 256 })

// With capped reasoning (N tokens of thinking, then respond)
const capped = await llm.complete(messages, { thinkingBudget: 64, maxTokens: 512 })

// Streaming
for await (const chunk of llm.stream(messages, { thinkingBudget: 0 })) {
  process.stdout.write(chunk.content ?? '')
  if (chunk.done) break
}

// Embeddings
const embedding = await llm.embed('some text')

// Tool calling
const result = await llm.complete(messages, {
  tools: [{ name: 'get_weather', description: '...', parameters: { ... } }],
  toolChoice: 'auto',
})
```

### Vision LLM

```ts
const vlm = await loadModel('qwen2-vl-2b.gguf', {
  type: 'llm', contextSize: 4096, mmprojPath: 'mmproj.gguf'
})

const result = await vlm.complete([{
  role: 'user',
  content: [
    { type: 'image', data: pngBuffer },            // Buffer, or { path: 'photo.jpg' }
    { type: 'text', text: 'Describe this image.' },
  ],
}])
```

Supported architectures: Gemma 4, Qwen2-VL, Qwen3-VL, LLaVA, Gemma3, CogVLM, Chameleon, MiniCPM, InternVL, etc.

### Image Generation (FLUX 2)

```ts
const img = createModel('flux-2-klein-4b.gguf', 'image')
await img.load({
  llmPath: 'qwen3-4b.gguf',        // text encoder for FLUX 2
  vaePath: 'flux2-vae.safetensors', // VAE decoder
  keepVaeOnCpu: true,
})

const png = await img.generate('a sunset over mountains', {
  width: 512, height: 512, steps: 4, cfgScale: 1.0,
})
fs.writeFileSync('output.png', png)
```

### Video Generation (Wan 2.2)

```ts
const vid = createModel('wan2.2-5b.gguf', 'image')
await vid.load({ t5xxlPath: 'umt5-xxl.gguf', vaePath: 'wan-vae.safetensors' })

const frames = await vid.generateVideo('a dog playing fetch', {
  width: 832, height: 480, videoFrames: 33, steps: 30,
})
// frames is Buffer[] — one PNG per frame
```

### Qwen3-TTS (voice cloning)

```ts
const tts = createModel('/path/to/qwen3-tts-models/', 'tts')
await tts.load()

// Clone any voice from a short WAV reference
const wav = await tts.speak('Text to speak in cloned voice.', {
  referenceAudioPath: '/path/to/reference.wav', // 24kHz mono, 3-10s recommended
})

// Or generate with default voice (no cloning)
const wav2 = await tts.speak('Default voice synthesis.')
```

### Utilities

```ts
const gpu = detectGpu()
// { backend: 'metal' | 'cuda' | 'cpu' }

const meta = await readGGUFMetadata('model.gguf')
// { architecture, contextLength, blockCount, embeddingLength, ... }
```

## Web Server

```bash
npm run serve    # http://localhost:3333
```

Built-in web UI with tabs for Chat, TTS, Image, and Video. Models load on-demand when you click a tab.

## Architecture

All engines compile against a **single shared ggml** (tensor library + GPU backends) and link into one N-API addon:

```
engine/
  ggml/        ← shared tensor ops, Metal/CUDA/CPU backends
  src/         ← llama.cpp LLM core
  tts/         ← qwen3-tts.cpp (compiled against shared ggml)
  mtmd/        ← multimodal/vision (CLIP encoder, compiled against shared ggml)
  diffusion/   ← stable-diffusion.cpp (compiled against shared ggml)

→ build/Release/omni.node (single output, ~10-30MB depending on platform)
```

### Why not node-llama-cpp?

[node-llama-cpp](https://github.com/withcatai/node-llama-cpp) is a well-maintained binding — if you only need LLM inference, use it. This project exists because we need multiple engines (LLM, TTS, Vision, Image/Video) sharing **one ggml build** in **one process**:

- **No symbol conflicts** — llama.cpp, stable-diffusion.cpp, and qwen3-tts.cpp all depend on ggml. Separate binaries mean duplicate symbols and linker errors. One unified build solves this.
- **One GPU backend** — Metal/CUDA is initialized once and shared across all engines. Separate bindings would compete for VRAM and require independent backend negotiation.
- **Thin by design** — this library handles inference and chat templating (via llama.cpp's native Jinja engine) but not tool-call orchestration or model management. Those are handled by consumers (e.g. agent-orcha).

## Build

```bash
npm install

npm run build          # auto-detect GPU
npm run build:metal    # macOS Metal
npm run build:cuda     # NVIDIA CUDA
npm run build:cpu      # CPU only
```

## Test

```bash
# Unit tests (no models needed)
npm test

# Download test models
bash scripts/download-test-models.sh              # LLM (~745MB)
bash scripts/download-test-models.sh --tts         # + Qwen3-TTS (~1.2GB)
bash scripts/download-test-models.sh --image       # + FLUX 2 Klein (~5GB)
bash scripts/download-test-models.sh --video       # + Wan 2.2 5B (~5GB)
bash scripts/download-test-models.sh --vision      # + Qwen2-VL 2B (~1.8GB)
bash scripts/download-test-models.sh --all         # Everything

# Integration tests (requires models in ~/.orcha/workspace/.models/)
node scripts/full-integration-test.ts              # All engines
node scripts/vision-integration-test.ts            # Vision/Multimodal
```

## Platforms

| Platform | GPU | Status |
|----------|-----|--------|
| macOS arm64 | Metal | Tested |
| Linux x64 | CPU | CI |
| Linux x64 | CUDA | CI |
| Linux arm64 | CPU | CI |
| Windows x64 | CPU | CI |
| Windows x64 | CUDA | CI |

## Requirements

- **Node.js** >= 25.0.0
- **CMake** >= 3.15
- **C++17** compiler (Clang, GCC, MSVC)
- **macOS**: Xcode Command Line Tools (for Metal)
- **Linux/Windows**: CUDA Toolkit 13.1+ (for NVIDIA GPU support)

## Upstream Sync Log

| Date | llama.cpp Commit | Notable Additions |
|------|-----------------|-------------------|
| 2026-04-03 | [`f49e917`](https://github.com/ggml-org/llama.cpp/commit/f49e917) | Gemma 4 vision projector (`gemma4v`), Gemma 4 audio projector (`gemma4a`), Gemma 4 MoE LLM arch, Gemma 4 tokenizer, AMD ZenDNN label, various mtmd fixes |
| 2026-04-02 | [`f49e917`](https://github.com/ggml-org/llama.cpp/commit/f49e917) | Initial full sync — Gemma 4 LLM (text-only), all existing vision projectors |

**Local patches** (preserved across syncs):
- `engine/ggml/src/ggml-cuda/ggml-cuda.cu` — CUDA errors throw `std::runtime_error` instead of `GGML_ABORT` so N-API callers can catch OOM and retry with fewer GPU layers
- `engine/ggml/src/ggml-metal/ggml-metal-device.m` — Metal device customization

## License

MIT
