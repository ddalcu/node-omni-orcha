# node-omni-orcha

Unified native Node.js bindings for llama.cpp, whisper.cpp, TTS.cpp, and stable-diffusion.cpp.

## Architecture

- **Separate `.node` per engine** — `llm.node`, `stt.node`, `tts.node`, `image.node` to avoid ggml symbol conflicts
- **cmake-js** build system with Metal/CUDA/CPU backends
- **Node 25 native TypeScript** — no build step, type-stripping
- **node:test** for testing

## Project Structure

```
src/           TypeScript API layer (ESM, no build step)
cpp/           C++ N-API bindings per engine
  common/      Shared async/error helpers
  llm/         llama.cpp binding
  image/       stable-diffusion.cpp binding (FLUX support)
  stt/         whisper.cpp binding
  tts/         TTS.cpp binding (Kokoro, Parler, Dia, Orpheus)
deps/          Git submodules (llama.cpp, whisper.cpp, stable-diffusion.cpp, TTS.cpp)
test/          Tests using node:test
  fixtures/    Model files (gitignored)
scripts/       Build and download helpers
```

## Build

```bash
npm install
npm run build          # auto-detect GPU
npm run build:metal    # macOS Metal
npm run build:cuda     # NVIDIA CUDA
npm run build:cpu      # CPU only
```

Produces: `build/Release/{llm,image,stt,tts}.node`

## Test

```bash
# Unit tests (no model files needed)
npm run test:unit

# Download models for integration tests
bash scripts/download-test-models.sh                # LLM only (~670MB)
bash scripts/download-test-models.sh --whisper      # + Whisper tiny (~75MB)
bash scripts/download-test-models.sh --flux         # + FLUX.1-dev (~17GB)

# Full test suite
npm test
```

## API

```ts
import { loadModel, createModel, detectGpu, readGGUFMetadata } from 'node-omni-orcha'

// LLM
const llm = await loadModel('model.gguf', { type: 'llm', contextSize: 4096 })
const result = await llm.complete([{ role: 'user', content: 'Hello' }])
const embedding = await llm.embed('some text')

// Image (SD or FLUX)
const img = createModel('flux-dev.gguf', 'image')
await img.load({ clipLPath: 'clip_l.safetensors', t5xxlPath: 't5xxl.gguf', vaePath: 'ae.safetensors' })
const png = await img.generate('a sunset', { width: 1024, height: 1024, cfgScale: 1.0 })

// Speech-to-Text
const stt = await loadModel('whisper-tiny.bin', { type: 'stt' })
const transcript = await stt.transcribe(pcmBuffer, { language: 'en' })

// Text-to-Speech
const tts = await loadModel('kokoro.gguf', { type: 'tts' })
const wav = await tts.speak('Hello world', { voice: 'af_bella' })
```

## Key Conventions

- Tool calling is NOT handled by this library — consumers (agent-orcha) implement it
- Audio for STT: 16-bit PCM, 16kHz, mono
- Audio from TTS: WAV format (16-bit PCM)
- Image output: PNG buffer
- All inference runs off the event loop via AsyncWorker
- GPU auto-detected (Metal on macOS, CUDA on Linux/Windows, CPU fallback)

## Dependencies (submodules)

- `deps/llama.cpp` — pinned to b8467
- `deps/stable-diffusion.cpp` — latest
- `deps/whisper.cpp` — latest
- `deps/TTS.cpp` — mmwillet/TTS.cpp (Kokoro, Parler, Dia, Orpheus)
- `deps/qwen3-tts.cpp` — predict-woo/qwen3-tts.cpp (Qwen3-TTS with voice cloning)

## Known Issues

- TTS.cpp (Kokoro) runner destructor crashes on cleanup — we leak the runner intentionally (OS reclaims on exit)
- sd.cpp context reuse crashes on second image generation — create a new context per image
- Qwen3-TTS requires separate ggml build (ExternalProject) to avoid symbol conflicts
