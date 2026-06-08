@echo off
cd /d "%~dp0"
set "UV_HTTP_TIMEOUT=300"

if exist ".ppocr_use_tuna" (
  set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
  set "UV_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  set "UV_INDEX_STRATEGY=unsafe-best-match"
  echo Using Tsinghua PyPI mirror and PaddlePaddle CPU index.
)

if exist ".ppocr_no_sync" (
  findstr /I "gpu" ".ppocr_no_sync" >nul 2>nul
  if not errorlevel 1 (
    nvidia-smi 2>nul | findstr /C:"CUDA Version: 13" >nul
    if not errorlevel 1 (
      uv run --no-sync python -c "import pathlib, sys; root=pathlib.Path(sys.prefix)/'Lib'/'site-packages'/'nvidia'; raise SystemExit(0 if root.exists() and any(root.rglob('cublasLt64_13.dll')) else 1)" >nul 2>nul
      if errorlevel 1 (
        echo Missing CUDA 13 cuBLAS runtime. Installing nvidia-cublas...
        uv pip install nvidia-cublas -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
        if errorlevel 1 goto :error
      )
    )
  )

  uv run --no-sync python -c "import PIL, paddleocr, paddle" >nul 2>nul
  if errorlevel 1 (
    echo Missing OCR runtime dependencies in acceleration environment. Installing them now...
    uv pip install "pillow>=10,<13" "paddleocr[doc-parser]>=3.6.0,<4"
    if errorlevel 1 goto :error

    uv run --no-sync python -c "import paddle" >nul 2>nul
    if errorlevel 1 (
      findstr /I "gpu" ".ppocr_no_sync" >nul 2>nul
      if errorlevel 1 (
        uv pip install "paddlepaddle>=3.2.1,<4"
      ) else (
        set "PADDLE_GPU_PACKAGE=paddlepaddle-gpu==3.3.0"
        set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/"
        set "PADDLE_GPU_RUNTIME_PACKAGE="
        nvidia-smi 2>nul | findstr /C:"CUDA Version: 13" >nul
        if not errorlevel 1 (
          set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu130/"
          set "PADDLE_GPU_RUNTIME_PACKAGE=nvidia-cublas"
        )
        uv pip install %PADDLE_GPU_PACKAGE% -i %PADDLE_GPU_INDEX%
        if defined PADDLE_GPU_RUNTIME_PACKAGE (
          uv pip install %PADDLE_GPU_RUNTIME_PACKAGE% -i %PADDLE_GPU_INDEX%
        )
      )
      if errorlevel 1 goto :error
    )

    uv run --no-sync python -c "import PIL, paddleocr, paddle" >nul 2>nul
    if errorlevel 1 goto :error
  )
  uv run --no-sync python launcher.py
) else (
  uv run python launcher.py
)
pause
exit /b 0

:error
echo Failed to install required OCR runtime dependencies. Check the error above, then run start.bat again.
pause
exit /b 1
