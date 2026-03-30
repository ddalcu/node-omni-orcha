# node-omni-orcha

Unified native Node.js inference engine — single `omni.node` for LLM, STT, TTS (Kokoro + Qwen3), and Image/Video generation.

## Architecture

- **Single `omni.node`** — all engines share one ggml build, one binary, no symbol conflicts
- **`engine/`** — llama.cpp fork with whisper, qwen3-tts, kokoro-tts, mtmd (multimodal/vision), and stable-diffusion sources compiled against shared ggml
- **Kokoro TTS** — Kokoro-82M via ONNX Runtime, ~600-900ms per sentence on Apple M-series, 49 preset voices, zero external dependencies (embedded CMU phoneme dictionary)
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
  kokoro/      Kokoro-82M TTS (ONNX Runtime, optional — no external deps)
    kokoro.h/cpp       C API: create, speak, list voices, free
    phonemize.h/cpp    CMU dict G2P + weak forms + t-flapping
    vocab.h            187-symbol IPA → token ID mapping
  mtmd/        Multimodal/Vision (CLIP encoder, image preprocessing, compiled against shared ggml)
  diffusion/   stable-diffusion.cpp (FLUX 2, Wan 2.2, compiled against shared ggml)
  vendor/      Third-party (cpp-httplib, stb, nlohmann, onnxruntime/)
cpp/           N-API binding (one unified omni.node)
  common/      Shared async/error helpers
  omni_binding.cpp   Module init (registers all contexts)
  llm_context.*      LLM N-API wrapper
  stt_context.*      STT N-API wrapper
  tts_context.*      TTS (Qwen3) N-API wrapper
  kokoro_context.*   TTS (Kokoro) N-API wrapper
  image_context.*    Image/Video N-API wrapper
src/           TypeScript API layer (ESM, no build step)
  tts/kokoro-model.ts  Kokoro TypeScript wrapper
  tts/tts-model.ts     Qwen3-TTS TypeScript wrapper
scripts/       Build, download, and test helpers
  download-onnxruntime.sh     Downloads pre-built ONNX Runtime (macOS/Linux/Windows)
  download-kokoro-voices.ts   Downloads + packs 49 voice .bin files into voices.bin
  generate-phoneme-dict.ts    Downloads CMU dict → phoneme_dict.bin (126K words)
  kokoro-test.ts              Kokoro TTS benchmark + STT roundtrip test
  download-test-models.sh     Model downloader (LLM, STT, TTS, Image, Video, Vision)
packages/      Platform-specific npm packages (darwin-arm64, linux-x64, etc.)
server.ts      Web UI — chat, voice conversation, TTS, image/video generation
test/          Tests using node:test
  fixtures/    Test audio, model files (gitignored)
```

## Build

```bash
npm install

# Optional: Kokoro TTS support (only needs ONNX Runtime — no other dependencies)
bash scripts/download-onnxruntime.sh   # downloads pre-built ONNX Runtime (~50MB)

npm run build          # auto-detect GPU
npm run build:metal    # macOS Metal
npm run build:cuda     # NVIDIA CUDA
npm run build:cpu      # CPU only
```

Produces: `build/Release/omni.node`

Kokoro TTS is optional — if ONNX Runtime is not found, the build succeeds without it.

## Test

```bash
# Unit tests (no model files needed)
npm run test:unit

# Download models for integration tests
bash scripts/download-test-models.sh                # LLM + STT (~745MB)
bash scripts/download-test-models.sh --kokoro       # + Kokoro TTS (~190MB, fast)
bash scripts/download-test-models.sh --whisper      # STT only (~75MB)

# Full test suite
npm test

# Kokoro TTS benchmark + STT roundtrip
node scripts/kokoro-test.ts

# Full integration test (all engines)
node scripts/full-integration-test.ts

# Samuel L. Jackson themed test (LLM + TTS + STT + Image)
node scripts/samuel-jackson-test.ts

# Vision/Multimodal tests (requires --vision models)
bash scripts/download-test-models.sh --vision
node scripts/vision-integration-test.ts
node scripts/vision-stress-test.ts
node scripts/vision-crash-test.ts
```

## Server

```bash
npm run serve    # http://localhost:3333 (HTTPS if certs/ exists)
```

Web UI with tabs for Chat, STT, TTS, Image, Video, and Voice conversation. Models load on-demand when you click a tab.

**Voice pipeline**: continuous conversation with VAD (voice activity detection). Click "Start Conversation" → speak → auto-detects silence → sends to STT → LLM (with rolling 20-turn history) → Kokoro TTS → plays response → resumes listening. Configurable silence threshold and voice selection.

Supports HTTPS for mobile testing (microphone requires secure context):
```bash
# Generate self-signed certs (auto-detected by server)
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj "/CN=localhost"
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

// Kokoro TTS (fast, 49 preset voices — ~600-900ms per sentence)
const kokoro = await loadModel('/path/to/kokoro/', { type: 'kokoro' })
const wav = await kokoro.speak('Hello world', { voice: 'af_heart', speed: 1.0 })
const voices = kokoro.listVoices()  // ['af_alloy', 'af_heart', 'am_adam', ...]

// Qwen3-TTS (voice cloning — slower, ~20s per sentence)
const tts = createModel('/path/to/qwen3-tts/', 'tts')
await tts.load()
const wav2 = await tts.speak('Hello world', { referenceAudioPath: 'voice.wav' })
```

## Design Decisions

- **Custom binding over node-llama-cpp** — node-llama-cpp only covers LLM. We need LLM + STT + TTS + Image/Video sharing one ggml build to avoid symbol conflicts and GPU backend contention. Separate binaries for each engine would mean duplicate ggml symbols, competing VRAM allocation, and more total complexity.
- **Thin inference layer** — this library handles inference and chat templating (via llama.cpp's native Jinja engine) but not tool-call orchestration or model management. Consumers (agent-orcha) own that.
- **Kokoro as separate type (`'kokoro'`) not merged into TTS** — Kokoro has a fundamentally different API (preset voices, speed control, dictionary phonemization) vs Qwen3-TTS (voice cloning, temperature, reference audio). Keeping them as separate model types avoids a leaky abstraction where half the options are irrelevant. Both coexist: Kokoro for fast conversational use, Qwen3-TTS for voice cloning.
- **ONNX Runtime as only external dependency for Kokoro** — Downloaded as a pre-built shared library, no compilation needed. Builds without it succeed normally. Phonemization uses an embedded CMU dictionary (English) — no espeak-ng or other system libraries required. Multi-language support can be added later by upgrading to espeak-ng (see memory note).

## Kokoro Phonemization

The phonemization pipeline converts English text to Kokoro token IDs without any external dependencies:

**Pipeline**: Text → number expansion → tokenize → per-word lookup → post-processing → token IDs

**Three layers** (checked in order):
1. **Function word reductions** (~35 words) — "to"→tə, "the"→ðə, "a"→ə, "for"→fəɹ, etc. Hardcoded in `engine/kokoro/phonemize.cpp` (`g_weak_forms` map). Prevents robotic over-pronunciation of common unstressed words.
2. **CMU dictionary** (126K words) — `phoneme_dict.bin`, generated from Carnegie Mellon Pronouncing Dictionary. ARPABET converted to IPA with espeak-ng-style vowel length markers (iː, uː, ɑː, ɔː) and syllable-onset stress placement.
3. **Letter-to-sound fallback** — Simple rules for unknown words (names, neologisms). Handles digraphs (th, sh, ch, ph, ng) then single-letter mappings.

**Post-processing rules** (applied after dictionary lookup):
- **T-flapping**: /t/ or /d/ between vowels → ɾ, but ONLY when not at a stressed syllable onset. "water"→wɔːɾəɹ but "hotel"→hoʊˈtɛl. Rule checks backward from the t/d for stress markers.

**Tuning pronunciation**:
- Add/modify entries in `g_weak_forms` (phonemize.cpp) for function word reductions
- Regenerate `phoneme_dict.bin` after changing ARPABET→IPA mappings in `scripts/generate-phoneme-dict.ts`
- Add post-processing rules in `apply_phonological_rules()` (phonemize.cpp) for context-dependent allophonic changes
- The vocabulary mapping (vocab.h) must match Kokoro's training vocabulary exactly — do not modify

**Future improvements**:
- Upgrade to espeak-ng for multi-language support and higher-quality phonemization (see project memory)
- Add more allophonic rules (vowel reduction in unstressed syllables, glottalization, etc.)
- Context-dependent pronunciation (e.g., "read" present vs past tense)

## Key Conventions

- `loadModel()` requires explicit `type` — no auto-detection
- Tool calling is NOT handled by this library — consumers (agent-orcha) implement it
- Audio for STT: 16-bit PCM, 16kHz, mono
- Audio from TTS: WAV format (16-bit PCM, 24kHz) — both Kokoro and Qwen3-TTS
- Audio normalization: Kokoro output is peak-normalized to 90% headroom with outlier clamping (±10.0) to prevent clipping on loud voices
- Kokoro TTS is optional — requires ONNX Runtime at build time (no other dependencies)
- Kokoro phonemization: CMU dictionary (126K English words) + letter-to-sound fallback. English only for now.
- Kokoro voices: 49 presets. Naming: `{lang}{gender}_{name}` — af=American Female, am=American Male, bf=British Female, bm=British Male, etc. Use `listVoices()` to enumerate.
- Kokoro style indexing: voice frame is selected by phoneme count WITHOUT pad tokens (matches Python reference `voice[len(tokens)]` before `[[0, *tokens, 0]]`)
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

CI/CD (.github/workflows/build-and-publish.yml) builds all platforms, auto-downloads ONNX Runtime per-platform, and bundles libonnxruntime with platform packages.

## Known Issues

- sd.cpp context reuse crashes on second image generation — create a new context per image
- Qwen3.5-4B with unlimited reasoning (thinkingBudget=-1) may use all tokens on thinking — use thinkingBudget=0 or a positive budget for direct responses
- Kokoro: some voices produce outlier samples (1e30+) which are clamped to ±10.0 before normalization — this is a model quirk, not a bug
- Kokoro: word-initial schwa (ə) in words like "about" can sound slightly "o"-colored — model rendering limitation, not phonemization
- Kokoro: the `af` base voice (524KB, different size) is excluded from the voice pack — use named variants like `af_heart` instead
