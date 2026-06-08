#!/bin/bash
set -e
cd "$(dirname "$0")"
export UV_HTTP_TIMEOUT="300"

if [ "${1:-}" = "tuna" ] || [ "${1:-}" = "--tuna" ] || [ -f ".ppocr_use_tuna" ]; then
  export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  export UV_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  export UV_INDEX_STRATEGY="unsafe-best-match"
  echo "tuna" > .ppocr_use_tuna
  echo "Using Tsinghua PyPI mirror for base dependencies."
fi

echo "Syncing base dependencies first. This prevents missing packages such as PIL/Pillow."
uv sync

uv pip install "mlx-vlm>=0.3.11"
echo mlx > .ppocr_no_sync
echo "macOS Apple Silicon acceleration installed."
echo "Run start.command again; it will start mlx_vlm.server automatically."
