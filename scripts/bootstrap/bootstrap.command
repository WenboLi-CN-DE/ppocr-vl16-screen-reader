#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

USE_CHINA_SOURCES=1
if [ "${1:-}" = "--global" ] || [ -f ".ppocr_use_global" ]; then
  USE_CHINA_SOURCES=0
fi

export UV_HTTP_TIMEOUT="300"
export PADDLE_PDX_CACHE_HOME="$PWD/models"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"

if [ "$USE_CHINA_SOURCES" = "1" ]; then
  export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  export UV_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  export UV_INDEX_STRATEGY="unsafe-best-match"
  export PADDLE_PDX_MODEL_SOURCE="modelscope"
  echo "china" > .ppocr_use_tuna
  echo "Using China mirrors: Tsinghua PyPI, PaddlePaddle index, ModelScope models."
else
  export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-huggingface}"
  echo "Using global package/model sources."
fi

echo "Ensuring Python 3.11 is available..."
uv python install 3.11

if [ -f ".ppocr_no_sync" ]; then
  echo "Acceleration environment marker found; checking runtime packages without uv sync..."
  if ! uv run --no-sync python -c "import PIL, paddleocr, paddle" >/dev/null 2>&1; then
    echo "Installing missing runtime packages into existing environment..."
    uv pip install "pillow>=10,<13" "paddleocr[doc-parser]>=3.6.0,<4"
    if ! uv run --no-sync python -c "import paddle" >/dev/null 2>&1; then
      uv pip install "paddlepaddle>=3.2.1,<4"
    fi
  fi
else
  echo "Syncing base dependencies..."
  uv sync
fi

machine="$(uname -m 2>/dev/null || true)"
system="$(uname -s 2>/dev/null || true)"
if [ "$system" = "Darwin" ] && { [ "$machine" = "arm64" ] || [ "$machine" = "aarch64" ]; }; then
  if ! command -v mlx_vlm.server >/dev/null 2>&1; then
    echo "Installing Apple Silicon MLX-VLM acceleration..."
    uv pip install "mlx-vlm>=0.3.11"
    echo "mlx" > .ppocr_no_sync
  fi
fi

echo "Ensuring local OCR models are downloaded..."
if [ -f ".ppocr_no_sync" ]; then
  uv run --no-sync python scripts/models/download_models.py
else
  uv run python scripts/models/download_models.py
fi

echo "Bootstrap complete."
