#!/bin/bash
set -e
cd "$(dirname "$0")"
export UV_HTTP_TIMEOUT="300"

if [ -f ".ppocr_use_tuna" ]; then
  export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  export UV_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  export UV_INDEX_STRATEGY="unsafe-best-match"
  echo "Using Tsinghua PyPI mirror and PaddlePaddle CPU index."
fi

if [ -f ".ppocr_no_sync" ]; then
  if ! uv run --no-sync python -c "import PIL, paddleocr, paddle" >/dev/null 2>&1; then
    echo "Missing OCR runtime dependencies in acceleration environment. Installing them now..."
    uv pip install "pillow>=10,<13" "paddleocr[doc-parser]>=3.6.0,<4"

    if ! uv run --no-sync python -c "import paddle" >/dev/null 2>&1; then
      if grep -qi "gpu" ".ppocr_no_sync"; then
        uv pip install paddlepaddle-gpu==3.2.1 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
      else
        uv pip install "paddlepaddle>=3.2.1,<4"
      fi
    fi

    uv run --no-sync python -c "import PIL, paddleocr, paddle"
  fi
  uv run --no-sync python launcher.py
else
  uv run python launcher.py
fi
