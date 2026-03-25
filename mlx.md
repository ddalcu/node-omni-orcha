# MLX Backend for Apple Silicon

## Problem

ggml's Metal backend is significantly slower than CUDA on equivalent hardware. The M4 Max (27 TFLOPS, 546 GB/s) underperforms an RTX 2080 Super (14 TFLOPS, 448 GB/s) by ~3-5x on video diffusion due to immature Metal compute kernels in ggml — particularly for quantized matmul, attention, and fused operations.

## Solution

Add MLX as an alternative compute backend on Apple Silicon, keeping ggml-cuda for Windows/Linux. MLX is Apple's open-source ML framework (C++ / Metal) designed specifically for Apple Silicon, with efficient quantized inference and GGUF support.

```
                    ┌─── ggml-cuda ──── NVIDIA GPU (Win/Linux)
omni.node ──────────┤
                    ├─── ggml-cpu ───── CPU fallback (all platforms)
                    │
                    └─── mlx ────────── Apple Silicon (macOS)
```

## Why MLX over MPSGraph

- C++ API — integrates with N-API binding naturally
- Loads GGUF directly — same model files, no conversion
- Quantized matmul on Metal (2/4/6/8-bit, grouped) — what ggml lacks
- Lazy evaluation + graph fusion — automatic kernel optimization
- Open source, active community, Apple Silicon-specific tuning
- KV cache, sampling built in for LLM

## Architecture

### Current (all engines use ggml)

```
image_context.cpp → stable-diffusion.cpp → ggml → ggml-metal (slow)
llm_context.cpp   → llama.cpp            → ggml → ggml-metal (slow)
stt_context.cpp   → whisper.cpp          → ggml → ggml-metal (ok)
tts_context.cpp   → qwen3-tts.cpp        → ggml → ggml-metal (slow)
```

### Proposed (MLX on Mac, ggml-cuda on Win/Linux)

```
image_context.cpp → [Mac] mlx_diffusion.cpp → MLX → Metal (fast)
                  → [Win] stable-diffusion.cpp → ggml → CUDA (fast)

llm_context.cpp   → [Mac] mlx_llm.cpp → MLX → Metal (fast)
                  → [Win] llama.cpp    → ggml → CUDA (fast)

stt_context.cpp   → whisper.cpp → ggml (acceptable on all platforms)

tts_context.cpp   → [Mac] mlx_tts.cpp → MLX → Metal (fast)
                  → [Win] qwen3-tts.cpp → ggml → CUDA (fast)
```

### Runtime backend selection

```cpp
// image_context.cpp
#if defined(__APPLE__) && defined(MLX_BACKEND)
  #include "mlx_diffusion.h"  // MLX path
#else
  #include "stable-diffusion.h"  // ggml path
#endif
```

## Implementation Plan

### Phase 1: Diffusion only (biggest impact) — 2-3 weeks

Video/image generation is the bottleneck. LLM and TTS are slower but less critical.

**1.1 MLX diffusion wrapper** (`cpp/mlx/mlx_diffusion.h/.cpp`)
- Load GGUF diffusion model weights via MLX C++ API
- Implement WAN 2.2 forward pass (DiT blocks, attention, feedforward)
- Expose same interface as stable-diffusion.cpp: `generate_video()`, `generate_image()`
- Support: text conditioning (from T5 embeddings), init image (I2V), cfg guidance

**1.2 T5 text encoder on MLX** (`cpp/mlx/mlx_t5.h/.cpp`)
- Load UMT5-XXL GGUF via MLX
- Run text encoding, output embeddings
- Or: keep T5 on ggml (runs once, not perf-critical)

**1.3 VAE on MLX** (`cpp/mlx/mlx_vae.h/.cpp`)
- Load VAE safetensors via MLX
- Encode init images, decode latents to pixels
- Or: keep VAE on ggml (runs once per generation)

**1.4 CMake integration**
- Detect macOS + Apple Silicon at configure time
- Fetch MLX via FetchContent or find_package
- Compile MLX backend only on Mac
- `GPU_BACKEND=mlx` option alongside metal/cuda/cpu

**1.5 Test & benchmark**
- Run same video-benchmark.ts on Mac
- Compare: ggml-metal vs MLX for WAN 2.2 5B
- Target: 2-4x speedup on M4 Max

### Phase 2: LLM on MLX — 1-2 weeks

**2.1 MLX LLM wrapper** (`cpp/mlx/mlx_llm.h/.cpp`)
- Load GGUF LLM (Qwen, Llama, etc.) via MLX
- Implement chat completion with KV cache
- Streaming support
- Tool calling (parse from output, same as current)

**2.2 Expose same interface as llm_context**
- `complete()`, `stream()`, `embed()`
- Drop-in replacement, same TypeScript API

### Phase 3: TTS on MLX — 1-2 weeks

**3.1 MLX TTS wrapper** (`cpp/mlx/mlx_tts.h/.cpp`)
- Port Qwen3-TTS transformer forward pass to MLX
- Vocoder (AudioTokenizerDecoder) on MLX
- Speaker encoder on MLX

### Phase 4: Optimization — 1 week

- Profile with Instruments (Metal System Trace)
- Tune MLX graph compilation options
- Memory: ensure mmap is used for large models
- Quantization: test q4 vs q8 vs f16 speed on MLX
- Batch frame generation where possible

## Files to Add

```
cpp/
  mlx/
    mlx_diffusion.h      # C API matching stable-diffusion.h interface
    mlx_diffusion.cpp     # WAN 2.2 DiT implementation on MLX
    mlx_llm.h             # LLM inference on MLX
    mlx_llm.cpp
    mlx_tts.h             # TTS inference on MLX
    mlx_tts.cpp
    mlx_common.h          # GGUF loading, tensor utils
```

## CMake Changes

```cmake
if(GPU_BACKEND STREQUAL "mlx")
  # Fetch MLX
  FetchContent_Declare(mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG v0.25.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(mlx)

  # Build MLX diffusion/llm/tts wrappers
  add_library(omni-mlx STATIC
    cpp/mlx/mlx_diffusion.cpp
    cpp/mlx/mlx_llm.cpp
    cpp/mlx/mlx_tts.cpp
  )
  target_link_libraries(omni-mlx PRIVATE mlx)
  target_link_libraries(omni PRIVATE omni-mlx)
  target_compile_definitions(omni PRIVATE MLX_BACKEND)
endif()
```

## Build Commands

```bash
npm run build:mlx     # macOS Apple Silicon — MLX backend
npm run build:metal   # macOS Apple Silicon — ggml Metal (current)
npm run build:cuda    # Windows/Linux — CUDA
npm run build:cpu     # CPU fallback
npm run build         # Auto-detect (mlx on Mac, cuda on Win/Linux)
```

## Model Compatibility

Same GGUF files work on all platforms. MLX loads GGUF natively.

| Model | GGUF file | ggml-cuda | ggml-metal | MLX |
|-------|-----------|-----------|------------|-----|
| WAN 2.2 TI2V-5B Q4_K_M | 3.2 GB | fast | slow | fast (target) |
| Qwen3.5-4B IQ4_NL | 2.5 GB | fast | slow | fast (target) |
| Qwen3-TTS 0.6B F16 | 1.8 GB | fast | slow | fast (target) |
| Whisper Tiny | 75 MB | fast | ok | ok (keep ggml) |

## Expected Performance (M4 Max)

| Engine | Current (ggml-metal) | Target (MLX) | Speedup |
|--------|---------------------|--------------|---------|
| WAN 2.2 video (20 steps) | ~40s/step? | ~5-8s/step | 5-8x |
| LLM Qwen3.5-4B (tok/s) | ~30 tok/s? | ~80-120 tok/s | 3-4x |
| TTS Qwen3 (RTF) | ~0.15x | ~0.5-1.0x | 3-6x |

## Dependencies

- [MLX C++ API](https://github.com/ml-explore/mlx) (MIT license)
- macOS 14.0+ / Xcode 15+
- Apple Silicon (M1 or later)

## References

- [MLX documentation](https://ml-explore.github.io/mlx/)
- [MLX GGUF support](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [MLX C++ examples](https://github.com/ml-explore/mlx/tree/main/examples/cpp)
- [WAN 2.2 model architecture](https://github.com/Wan-Video/Wan2.2)
