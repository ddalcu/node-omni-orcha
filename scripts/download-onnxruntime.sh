#!/bin/bash
# Downloads pre-built ONNX Runtime for the current platform.
# Extracts to engine/vendor/onnxruntime/ (include/ + lib/).
#
# Supports: macOS (arm64/x64), Linux (x64/arm64), Windows (x64) via Git Bash/MSYS2.
#
# Usage:
#   bash scripts/download-onnxruntime.sh
#   ORT_VERSION=1.22.0 bash scripts/download-onnxruntime.sh

set -e

ORT_VERSION="${ORT_VERSION:-1.22.0}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/engine/vendor/onnxruntime"

if [ -d "$DEST_DIR/include" ] && [ -d "$DEST_DIR/lib" ]; then
  echo "ONNX Runtime already exists at $DEST_DIR"
  exit 0
fi

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"
EXT="tgz"

case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64)  PLATFORM="osx-arm64" ;;
      x86_64) PLATFORM="osx-x86_64" ;;
      *)      echo "Unsupported macOS arch: $ARCH"; exit 1 ;;
    esac
    ;;
  Linux)
    case "$ARCH" in
      x86_64)  PLATFORM="linux-x64" ;;
      aarch64) PLATFORM="linux-aarch64" ;;
      *)       echo "Unsupported Linux arch: $ARCH"; exit 1 ;;
    esac
    ;;
  MINGW*|MSYS*|CYGWIN*)
    PLATFORM="win-x64"
    EXT="zip"
    ;;
  *)
    echo "Unsupported OS: $OS"; exit 1
    ;;
esac

# Use CUDA variant if GPU_BACKEND=cuda (set by CI or user)
# Exception: Windows — the GPU variant adds onnxruntime_providers_cuda.dll (322MB)
# which pushes the npm package over the 200MB limit. Kokoro (82M params) runs fine
# on CPU; the heavy CUDA work (LLM/STT/image) goes through ggml, not ONNX Runtime.
GPU_SUFFIX=""
if [ "${GPU_BACKEND}" = "cuda" ] && [ "$OS" != "Darwin" ]; then
  case "$OS" in
    MINGW*|MSYS*|CYGWIN*)
      echo "Skipping ONNX Runtime GPU variant on Windows (CPU-only for Kokoro TTS)"
      ;;
    *)
      GPU_SUFFIX="-gpu"
      ;;
  esac
fi

ARCHIVE="onnxruntime-${PLATFORM}${GPU_SUFFIX}-${ORT_VERSION}.${EXT}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE}"

echo "Downloading ONNX Runtime ${ORT_VERSION} for ${PLATFORM}..."
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

curl -L --progress-bar -o "$TMPDIR/$ARCHIVE" "$URL"

echo "Extracting..."
if [ "$EXT" = "zip" ]; then
  unzip -q "$TMPDIR/$ARCHIVE" -d "$TMPDIR"
else
  tar xzf "$TMPDIR/$ARCHIVE" -C "$TMPDIR"
fi

# The archive extracts to onnxruntime-{platform}[-gpu]-{version}/
EXTRACTED="$TMPDIR/onnxruntime-${PLATFORM}${GPU_SUFFIX}-${ORT_VERSION}"

mkdir -p "$DEST_DIR"
cp -r "$EXTRACTED/include" "$DEST_DIR/"
cp -r "$EXTRACTED/lib" "$DEST_DIR/"

echo "ONNX Runtime ${ORT_VERSION} installed to $DEST_DIR"
ls -lh "$DEST_DIR/lib/"
