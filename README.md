# node-omni-orcha

Unified native Node.js inference engine — **LLM**, **Image/Video Generation**, **Speech-to-Text**, and **Text-to-Speech** in a single `omni.node` binary.

Built on a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), and [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) compiled against a shared ggml backend.

## Features

- **LLM** — Chat completion, streaming, embeddings, tool calling, reasoning budget control
- **Image Generation** — FLUX 2, Wan 2.2, SD/SDXL (text-to-image)
- **Video Generation** — Wan 2.2 (text-to-video)
- **Speech-to-Text** — Whisper (language detection, timestamps)
- **Text-to-Speech** — Qwen3-TTS with voice cloning from 3s reference audio
- **GPU accelerated** — Metal (macOS), CUDA (NVIDIA), CPU fallback
- **Single binary** — One `omni.node` for all engines, one shared ggml, no symbol conflicts
- **Native N-API** — In-process inference, no child processes or HTTP servers
- **Node 25** — Native TypeScript, ESM, no build step

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

### Speech-to-Text (Whisper)

```ts
const stt = await loadModel('whisper-tiny.bin', { type: 'stt' })

const result = await stt.transcribe(pcmBuffer, { language: 'en' })
// { text: "Hello world", language: "en", segments: [{ start: 0.0, end: 1.5, text: "..." }] }

const lang = await stt.detectLanguage(pcmBuffer)
// "en"
```

Audio format: **16-bit PCM, 16kHz, mono**.

### Text-to-Speech with Voice Cloning (Qwen3-TTS)

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

## Architecture

All engines compile against a **single shared ggml** (tensor library + GPU backends) and link into one N-API addon:

```
engine/
  ggml/        ← shared tensor ops, Metal/CUDA/CPU backends
  src/         ← llama.cpp LLM core
  stt/         ← whisper.cpp (compiled against shared ggml)
  tts/         ← qwen3-tts.cpp (compiled against shared ggml)
  diffusion/   ← stable-diffusion.cpp (compiled against shared ggml)

→ build/Release/omni.node (single output, ~10-30MB depending on platform)
```

### Why not node-llama-cpp?

[node-llama-cpp](https://github.com/withcatai/node-llama-cpp) is a well-maintained binding — if you only need LLM inference, use it. This project exists because we need four engines (LLM, STT, TTS, Image/Video) sharing **one ggml build** in **one process**:

- **No symbol conflicts** — llama.cpp, whisper.cpp, stable-diffusion.cpp, and qwen3-tts.cpp all depend on ggml. Separate binaries mean duplicate symbols and linker errors. One unified build solves this.
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
bash scripts/download-test-models.sh              # LLM + STT (~745MB)

# Integration tests (requires models in ~/.orcha/workspace/.models/)
node scripts/full-integration-test.ts
node scripts/samuel-jackson-test.ts
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
- **Linux/Windows**: CUDA Toolkit 12.6+ (for NVIDIA GPU support)

## License

MIT
