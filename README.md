# node-omni-orcha

Unified native Node.js bindings for **LLM**, **Image Generation**, **Speech-to-Text**, and **Text-to-Speech** — all in one library.

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [TTS.cpp](https://github.com/mmwillet/TTS.cpp), and [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp).

## Features

- **LLM** — Chat completion, embeddings (llama.cpp, any GGUF model)
- **Image Generation** — Text-to-image with SD, SDXL, FLUX.1, FLUX.2 Klein/Dev
- **Speech-to-Text** — Audio transcription with Whisper (language detection, timestamps)
- **Text-to-Speech** — Two engines:
  - **Kokoro** — 28 built-in voices, fast, English-focused
  - **Qwen3-TTS** — Voice cloning from 3s reference audio, 10 languages
- **GPU accelerated** — Metal (macOS), CUDA (NVIDIA), CPU fallback
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

const result = await llm.complete([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Hello!' },
], { temperature: 0.7, maxTokens: 256 })

console.log(result.content)
console.log(result.usage) // { inputTokens, outputTokens, totalTokens }
```

### Image Generation (FLUX.2)

```ts
const img = createModel('flux-2-klein-4b.gguf', 'image')
await img.load({
  llmPath: 'qwen3-4b.gguf',       // text encoder for FLUX.2
  vaePath: 'flux2-ae.safetensors', // VAE decoder
  offloadToCpu: true,
})

const png = await img.generate('a sunset over mountains', {
  width: 512, height: 512, steps: 4, cfgScale: 1.0,
})
fs.writeFileSync('output.png', png)
```

Also supports FLUX.1 (with `clipLPath` + `t5xxlPath`) and standard SD/SDXL models.

### Speech-to-Text (Whisper)

```ts
const stt = await loadModel('whisper-tiny.bin', { type: 'stt' })

const result = await stt.transcribe(pcmBuffer, { language: 'en' })
// { text: "Hello world", language: "en", segments: [{ start: 0.0, end: 1.5, text: "..." }] }

const lang = await stt.detectLanguage(pcmBuffer)
// "en"
```

Audio format: **16-bit PCM, 16kHz, mono**.

### Text-to-Speech (Kokoro)

```ts
const tts = createModel('kokoro-q8.gguf', 'tts')
await tts.load() // defaults to Kokoro engine

const wav = await tts.speak('Hello world!', { voice: 'af_heart' })
fs.writeFileSync('speech.wav', wav)
```

28 built-in voices: `af_heart`, `am_adam`, `af_bella`, `bf_emma`, `bm_george`, etc.

### Text-to-Speech with Voice Cloning (Qwen3-TTS)

```ts
const tts = createModel('/path/to/qwen3-tts-models/', 'tts')
await tts.load({ engine: 'qwen3' })

// Clone any voice from a 3-second WAV reference
const wav = await tts.speak('Text to speak in cloned voice.', {
  referenceAudioPath: '/path/to/reference.wav', // 24kHz mono recommended
})

// Or generate with default voice (no cloning)
const wav2 = await tts.speak('Default voice synthesis.')
```

### Utilities

```ts
// Detect GPU
const gpu = detectGpu()
// { backend: 'metal' | 'cuda' | 'cpu', name?: string, vramBytes?: number }

// Read GGUF model metadata without loading
const meta = await readGGUFMetadata('model.gguf')
// { architecture, contextLength, blockCount, embeddingLength, ... }
```

## Architecture

Each C++ engine is compiled as a **separate `.node` addon** to avoid ggml symbol conflicts:

| Engine | File | Size | Backend |
|--------|------|------|---------|
| LLM | `llm.node` | ~4MB | llama.cpp |
| Image | `image.node` | ~23MB | stable-diffusion.cpp |
| STT | `stt.node` | ~2.4MB | whisper.cpp |
| TTS (Kokoro) | `tts.node` | ~1.4MB | TTS.cpp |
| TTS (Qwen3) | `tts_qwen3.node` | ~2.2MB | qwen3-tts.cpp |

## Build

```bash
git clone --recurse-submodules https://github.com/nickvdyck/node-omni-orcha.git
cd node-omni-orcha
npm install
npm run build          # auto-detect GPU
npm run build:metal    # macOS Metal
npm run build:cuda     # NVIDIA CUDA
npm run build:cpu      # CPU only
```

### Qwen3-TTS setup (optional)

Qwen3-TTS requires model conversion from HuggingFace:

```bash
cd deps/qwen3-tts.cpp
git submodule update --init --depth 1

# Build ggml + qwen3-tts
cmake -S ggml -B ggml/build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build ggml/build -j8
cmake -S . -B build -DQWEN3_TTS_COREML=OFF
cmake --build build -j8 --target qwen3_tts

# Convert models (requires Python)
python3 -m venv .venv && source .venv/bin/activate
pip install huggingface_hub torch safetensors numpy gguf
python scripts/setup_pipeline_models.py --coreml off
```

## Test

```bash
# Unit tests (no models needed)
npm run test:unit

# Download test models
bash scripts/download-test-models.sh              # LLM (~670MB)
bash scripts/download-test-models.sh --whisper     # + Whisper (~75MB)
bash scripts/download-test-models.sh --flux        # + FLUX.1-dev (~17GB)

# Run tests
npm test
```

## Requirements

- **Node.js** >= 25.0.0
- **CMake** >= 3.15
- **C++17** compiler (Clang, GCC, MSVC)
- **macOS**: Xcode Command Line Tools (for Metal)
- **Linux/Windows**: CUDA Toolkit (for NVIDIA GPU support)

## Dependencies (submodules)

| Library | Purpose | License |
|---------|---------|---------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference | MIT |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | Image generation | MIT |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Speech-to-text | MIT |
| [TTS.cpp](https://github.com/mmwillet/TTS.cpp) | TTS (Kokoro/Parler/Dia) | MIT |
| [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) | TTS with voice cloning | MIT |

## License

MIT
