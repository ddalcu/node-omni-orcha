# node-omni-orcha

Unified native Node.js inference engine — single `omni.node` for LLM, STT, TTS, and Image/Video generation.

## Architecture

- **Single `omni.node`** — all engines share one ggml build, one binary, no symbol conflicts
- **`engine/`** — llama.cpp fork with whisper, qwen3-tts, mtmd (multimodal/vision), and stable-diffusion sources compiled against shared ggml
- **cmake-js** build system with Metal/CUDA/CPU backends
- **Node 25 native TypeScript** — no build step, type-stripping
- **node:test** for testing

## Project Structure

```
engine/        llama.cpp fork (ggml + LLM core)
  ggml/        Tensor library + backends (CPU, Metal, CUDA only)
  src/         llama.cpp model loading, KV cache, sampling
  common/      Chat templates, sampling, reasoning budget
  include/     llama.h public headers
  stt/         Whisper STT (compiled against shared ggml)
  tts/         Qwen3-TTS (compiled against shared ggml)
  mtmd/        Multimodal/Vision (CLIP encoder, image preprocessing, compiled against shared ggml)
  diffusion/   stable-diffusion.cpp (FLUX 2, Wan 2.2, compiled against shared ggml)
  vendor/      Third-party headers (cpp-httplib, stb, nlohmann)
cpp/           N-API binding (one unified omni.node)
  common/      Shared async/error helpers
  omni_binding.cpp   Module init (registers all contexts)
  llm_context.*      LLM N-API wrapper
  stt_context.*      STT N-API wrapper
  tts_context.*      TTS N-API wrapper
  image_context.*    Image/Video N-API wrapper
src/           TypeScript API layer (ESM, no build step)
test/          Tests using node:test
  fixtures/    Model files (gitignored)
scripts/       Build, download, and test helpers
```

## Build

```bash
npm install
npm run build          # auto-detect GPU
npm run build:metal    # macOS Metal
npm run build:cuda     # NVIDIA CUDA
npm run build:cpu      # CPU only
```

Produces: `build/Release/omni.node`

## Test

```bash
# Unit tests (no model files needed)
npm run test:unit

# Download models for integration tests
bash scripts/download-test-models.sh                # LLM + STT (~745MB)
bash scripts/download-test-models.sh --whisper      # STT only (~75MB)

# Full test suite
npm test

# Full integration test (all engines, requires models in ~/.orcha/workspace/.models/)
node scripts/full-integration-test.ts

# Samuel L. Jackson themed test (LLM + TTS + STT + Image)
node scripts/samuel-jackson-test.ts

# Vision/Multimodal tests (requires --vision models)
bash scripts/download-test-models.sh --vision
node scripts/vision-integration-test.ts   # basic vision inference
node scripts/vision-stress-test.ts        # stress test
node scripts/vision-crash-test.ts         # crash hardening
```

## API

```ts
import { loadModel, createModel, detectGpu, readGGUFMetadata } from 'node-omni-orcha'

// LLM (type is required)
const llm = await loadModel('model.gguf', { type: 'llm', contextSize: 4096 })
const result = await llm.complete([{ role: 'user', content: 'Hello' }])
const result2 = await llm.complete([{ role: 'user', content: 'Hello' }], { thinkingBudget: 0 }) // no reasoning
const embedding = await llm.embed('some text')

// Vision LLM (Qwen2-VL, LLaVA, Gemma3, etc.)
const vlm = await loadModel('qwen2-vl-2b.gguf', {
  type: 'llm', contextSize: 4096, mmprojPath: 'mmproj.gguf'
})
const desc = await vlm.complete([{
  role: 'user',
  content: [
    { type: 'image', data: pngBuffer },            // Buffer, or { path: 'photo.jpg' }
    { type: 'text', text: 'Describe this image.' },
  ],
}])

// Image (FLUX 2 / Wan 2.2)
const img = createModel('flux2.gguf', 'image')
await img.load({ llmPath: 'qwen3-4b.gguf', vaePath: 'ae.safetensors' })
const png = await img.generate('a sunset', { width: 1024, height: 1024, cfgScale: 1.0 })

// Speech-to-Text (Whisper)
const stt = await loadModel('whisper-tiny.bin', { type: 'stt' })
const transcript = await stt.transcribe(pcmBuffer, { language: 'en' })

// Text-to-Speech (Qwen3-TTS with voice cloning)
const tts = createModel('/path/to/qwen3-tts/', 'tts')
await tts.load()
const wav = await tts.speak('Hello world', { referenceAudioPath: 'voice.wav' })
```

## Design Decisions

- **Custom binding over node-llama-cpp** — node-llama-cpp only covers LLM. We need LLM + STT + TTS + Image/Video sharing one ggml build to avoid symbol conflicts and GPU backend contention. Separate binaries for each engine would mean duplicate ggml symbols, competing VRAM allocation, and more total complexity.
- **Thin inference layer** — this library handles inference and chat templating (via llama.cpp's native Jinja engine) but not tool-call orchestration or model management. Consumers (agent-orcha) own that.

## Key Conventions

- `loadModel()` requires explicit `type` — no auto-detection
- Tool calling is NOT handled by this library — consumers (agent-orcha) implement it
- Audio for STT: 16-bit PCM, 16kHz, mono
- Audio from TTS: WAV format (16-bit PCM, 24kHz)
- Image output: PNG buffer
- All inference runs off the event loop via AsyncWorker
- GPU auto-detected (Metal on macOS, CUDA on Linux/Windows, CPU fallback)
- `thinkingBudget`: -1 = reasoning enabled (default), 0 = disabled, N>0 = capped at N tokens
- Vision models require `mmprojPath` in load options — a separate GGUF file for the vision encoder
- `ChatMessage.content` accepts `string` or `ContentPart[]` for multimodal (text + image) input
- Image input: Buffer (PNG/JPEG/BMP/GIF), file path, or raw RGB pixels with width/height
- Vision architectures: Qwen2-VL, Qwen3-VL, LLaVA, Gemma3, CogVLM, Chameleon, MiniCPM, etc.

## Platforms

- macOS arm64 (Metal GPU)
- Linux x64 (CPU, CUDA)
- Linux arm64 (CPU)
- Windows x64 (CPU, CUDA)

## Known Issues

- sd.cpp context reuse crashes on second image generation — create a new context per image
- Qwen3.5-4B with unlimited reasoning (thinkingBudget=-1) may use all tokens on thinking — use thinkingBudget=0 or a positive budget for direct responses
