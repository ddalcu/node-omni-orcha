#!/bin/bash
# Downloads models for testing.
# All models go to ~/.orcha/workspace/.models/
#
# Usage:
#   bash scripts/download-test-models.sh           # LLM + STT (basics, ~1GB)
#   bash scripts/download-test-models.sh --tts     # + Qwen3-TTS (~1.2GB)
#   bash scripts/download-test-models.sh --image   # + FLUX 2 Klein (~5GB)
#   bash scripts/download-test-models.sh --video   # + WAN 2.2 5B (~5GB)
#   bash scripts/download-test-models.sh --vision  # + Qwen2-VL 2B (~1.8GB)
#   bash scripts/download-test-models.sh --all     # Everything

set -e

MODELS_DIR="${MODELS_DIR:-$HOME/.orcha/workspace/.models}"
FIXTURES_DIR="$(dirname "$0")/../test/fixtures"
mkdir -p "$MODELS_DIR" "$FIXTURES_DIR"

download_if_missing() {
  local url="$1"
  local dest="$2"
  local desc="$3"
  if [ ! -f "$dest" ]; then
    echo "Downloading $desc..."
    mkdir -p "$(dirname "$dest")"
    curl -L --progress-bar -o "$dest" "$url"
    echo "Downloaded: $dest"
  else
    echo "Already exists: $dest"
  fi
}

has_flag() {
  for arg in "$@"; do
    if [ "$arg" = "$1" ]; then return 0; fi
  done
  return 1
}

ALL=false
if has_flag "--all" "$@"; then ALL=true; fi

# ─── Always: LLM + STT (basics) ───

echo "=== LLM: TinyLlama 1.1B (~250MB) ==="
download_if_missing \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  "$MODELS_DIR/tinyllama/tinyllama.gguf" \
  "TinyLlama 1.1B Q4_K_M"

echo ""
echo "=== LLM: Qwen3.5-4B (~2.5GB + 641MB mmproj) ==="
download_if_missing \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-IQ4_NL.gguf" \
  "$MODELS_DIR/qwen3-5-4b/Qwen3.5-4B-IQ4_NL.gguf" \
  "Qwen3.5-4B IQ4_NL"

download_if_missing \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf" \
  "$MODELS_DIR/qwen3-5-4b/mmproj-F16.gguf" \
  "Qwen3.5-4B mmproj F16 (vision encoder)"

echo ""
echo "=== STT: Whisper Tiny (~75MB) ==="
download_if_missing \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" \
  "$MODELS_DIR/whisper-tiny/whisper-tiny.bin" \
  "Whisper tiny"

# Generate test audio (silence) for STT tests
if [ ! -f "$FIXTURES_DIR/test-audio.pcm" ]; then
  echo "Generating silent test PCM (16kHz, 16-bit, mono, 1 second)..."
  node -e "
const fs = require('fs');
const buf = Buffer.alloc(32000);
fs.writeFileSync('$FIXTURES_DIR/test-audio.pcm', buf);
" 2>/dev/null || python3 -c "
import struct, sys
sys.stdout.buffer.write(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
" > "$FIXTURES_DIR/test-audio.pcm"
  echo "Generated: $FIXTURES_DIR/test-audio.pcm"
fi

# ─── TTS: Qwen3-TTS 0.6B (~1.2GB) ───

if $ALL || has_flag "--tts" "$@"; then
  echo ""
  echo "=== TTS: Qwen3-TTS 0.6B F16 (~1.2GB) ==="
  mkdir -p "$MODELS_DIR/qwen3-tts"

  download_if_missing \
    "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-0.6b-f16.gguf" \
    "$MODELS_DIR/qwen3-tts/qwen3-tts-0.6b-f16.gguf" \
    "Qwen3-TTS 0.6B F16 (~1GB)"

  download_if_missing \
    "https://huggingface.co/offbeatengineer/qwen3-tts-gguf/resolve/main/qwen3-tts-tokenizer-f16.gguf" \
    "$MODELS_DIR/qwen3-tts/qwen3-tts-tokenizer-f16.gguf" \
    "Qwen3-TTS tokenizer F16 (~200MB)"
fi

# ─── Image: FLUX 2 Klein 4B (~5GB total) ───

if $ALL || has_flag "--image" "$@"; then
  echo ""
  echo "=== Image: FLUX 2 Klein 4B (~5GB total) ==="
  mkdir -p "$MODELS_DIR/flux2-klein"

  download_if_missing \
    "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf" \
    "$MODELS_DIR/flux2-klein/flux-2-klein-4b-Q4_K_M.gguf" \
    "FLUX 2 Klein 4B Q4_K_M (~2.5GB)"

  download_if_missing \
    "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf" \
    "$MODELS_DIR/flux2-klein/Qwen3-4B-Q4_K_M.gguf" \
    "Qwen3-4B Q4_K_M (FLUX LLM, ~2.2GB)"

  download_if_missing \
    "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "$MODELS_DIR/flux2-klein/flux2-vae.safetensors" \
    "FLUX 2 VAE (~0.3GB)"
fi

# ─── Video: WAN 2.2 TI2V 5B (~5GB total) ───

if $ALL || has_flag "--video" "$@"; then
  echo ""
  echo "=== Video: WAN 2.2 TI2V 5B (~5GB total) ==="
  mkdir -p "$MODELS_DIR/wan22-5b"

  download_if_missing \
    "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q4_K_M.gguf" \
    "$MODELS_DIR/wan22-5b/Wan2.2-TI2V-5B-Q4_K_M.gguf" \
    "WAN 2.2 TI2V 5B Q4_K_M (~3.5GB)"

  download_if_missing \
    "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors" \
    "$MODELS_DIR/wan22-5b/Wan2.2_VAE.safetensors" \
    "WAN 2.2 VAE (~0.3GB)"

  download_if_missing \
    "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q8_0.gguf" \
    "$MODELS_DIR/wan22-5b/umt5-xxl-encoder-Q8_0.gguf" \
    "UMT5-XXL encoder Q8_0 (~6GB)"
fi

# ─── Vision: Qwen2-VL 2B Instruct (~1.8GB total) ───

if $ALL || has_flag "--vision" "$@"; then
  echo ""
  echo "=== Vision: Qwen2-VL 2B Instruct (~1.8GB total) ==="
  mkdir -p "$MODELS_DIR/qwen2-vl-2b"

  download_if_missing \
    "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf" \
    "$MODELS_DIR/qwen2-vl-2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf" \
    "Qwen2-VL 2B Instruct Q4_K_M (~0.9GB)"

  download_if_missing \
    "https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-2B-Instruct-f16.gguf" \
    "$MODELS_DIR/qwen2-vl-2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf" \
    "Qwen2-VL 2B mmproj F16 (~1.2GB)"

  # Download a test image
  download_if_missing \
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png" \
    "$MODELS_DIR/qwen2-vl-2b/test-image.png" \
    "Test image for vision"
fi

# ─── Summary ───

echo ""
echo "Done. Models directory: $MODELS_DIR"
echo ""
for subdir in tinyllama qwen3-5-4b whisper-tiny qwen3-tts flux2-klein wan22-5b qwen2-vl-2b; do
  if [ -d "$MODELS_DIR/$subdir" ]; then
    echo "$subdir/:"
    ls -lhS "$MODELS_DIR/$subdir/" 2>/dev/null || true
    echo ""
  fi
done

echo ""
echo "Run tests:"
echo "  npm test                                  # unit + integration"
echo "  node scripts/full-integration-test.ts     # all engines"
echo "  node scripts/samuel-jackson-test.ts       # LLM + TTS + STT + Image"
echo ""
if ! $ALL; then
  echo "Download more:"
  has_flag "--tts"    "$@" || echo "  bash scripts/download-test-models.sh --tts    # Qwen3-TTS (~1.2GB)"
  has_flag "--image"  "$@" || echo "  bash scripts/download-test-models.sh --image  # FLUX 2 Klein (~5GB)"
  has_flag "--video"  "$@" || echo "  bash scripts/download-test-models.sh --video  # WAN 2.2 5B (~5GB)"
  has_flag "--vision" "$@" || echo "  bash scripts/download-test-models.sh --vision # Qwen2-VL 2B (~1.8GB)"
  echo "  bash scripts/download-test-models.sh --all    # Everything"
fi
