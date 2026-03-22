#!/bin/bash
# Downloads test models for integration tests.
# Run: bash scripts/download-test-models.sh [--flux]

set -e

FIXTURES_DIR="$(dirname "$0")/../test/fixtures"
mkdir -p "$FIXTURES_DIR"

download_if_missing() {
  local url="$1"
  local dest="$2"
  local desc="$3"
  if [ ! -f "$dest" ]; then
    echo "Downloading $desc..."
    curl -L --progress-bar -o "$dest" "$url"
    echo "Downloaded: $dest"
  else
    echo "Already exists: $dest"
  fi
}

# --- LLM: TinyLlama 1.1B Q4_K_M (~670MB) ---
download_if_missing \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  "$FIXTURES_DIR/tinyllama.gguf" \
  "TinyLlama 1.1B Q4_K_M"

# --- FLUX models (only if --flux flag is passed) ---
if [ "$1" = "--flux" ]; then
  echo ""
  echo "=== Downloading FLUX.1-dev models (total ~17GB) ==="
  echo "  FLUX.1-dev Q8_0:  ~11.8GB"
  echo "  T5-XXL Q8_0:      ~4.7GB"
  echo "  CLIP-L:           ~0.2GB"
  echo "  VAE:              ~0.3GB"
  echo ""

  # FLUX.1-dev Q8_0 diffusion model
  download_if_missing \
    "https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf" \
    "$FIXTURES_DIR/flux-dev-Q8_0.gguf" \
    "FLUX.1-dev Q8_0 (~11.8GB)"

  # T5-XXL Q8_0 text encoder
  download_if_missing \
    "https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/t5xxl_q8_0.gguf" \
    "$FIXTURES_DIR/t5xxl-Q8_0.gguf" \
    "T5-XXL Q8_0 (~4.7GB)"

  # CLIP-L text encoder
  download_if_missing \
    "https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/clip_l.safetensors" \
    "$FIXTURES_DIR/clip_l.safetensors" \
    "CLIP-L (~0.2GB)"

  # VAE (autoencoder)
  download_if_missing \
    "https://huggingface.co/leejet/FLUX.1-dev-gguf/resolve/main/ae.safetensors" \
    "$FIXTURES_DIR/ae.safetensors" \
    "VAE ae.safetensors (~0.3GB)"
fi

# --- Whisper: tiny model (~75MB) + test audio ---
if [ "$1" = "--whisper" ] || [ "$1" = "--all" ]; then
  echo ""
  echo "=== Downloading Whisper test model ==="

  download_if_missing \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" \
    "$FIXTURES_DIR/whisper-tiny.bin" \
    "Whisper tiny (~75MB)"

  # Generate a short test audio file (1 second of silence as PCM)
  # For real testing, replace with actual speech audio
  if [ ! -f "$FIXTURES_DIR/test-audio.pcm" ]; then
    echo "Generating silent test PCM (16kHz, 16-bit, mono, 1 second)..."
    python3 -c "
import struct, sys
# 1 second of silence at 16kHz
sys.stdout.buffer.write(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
" > "$FIXTURES_DIR/test-audio.pcm" 2>/dev/null || \
    node -e "
const fs = require('fs');
const buf = Buffer.alloc(32000); // 16000 samples * 2 bytes = 1 second of silence
fs.writeFileSync('$FIXTURES_DIR/test-audio.pcm', buf);
"
    echo "Generated: $FIXTURES_DIR/test-audio.pcm"
  fi
fi

echo ""
echo "Done. Test models ready."
echo "  LLM tests:   node --test test/llm.test.ts"
echo "  Image tests: node --test test/image.test.ts"
echo "  STT tests:   node --test test/stt.test.ts"
echo ""
echo "Optional downloads:"
if [[ "$*" != *"--flux"* ]]; then
  echo "  bash scripts/download-test-models.sh --flux     # FLUX models (~17GB)"
fi
if [[ "$*" != *"--whisper"* ]]; then
  echo "  bash scripts/download-test-models.sh --whisper  # Whisper tiny (~75MB)"
fi
