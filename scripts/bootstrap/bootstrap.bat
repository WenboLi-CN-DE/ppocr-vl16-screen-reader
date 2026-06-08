@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\..\.."

set "USE_CHINA_SOURCES=1"
if /I "%~1"=="--global" set "USE_CHINA_SOURCES="
if exist ".ppocr_use_global" set "USE_CHINA_SOURCES="

set "UV_HTTP_TIMEOUT=300"
set "PADDLE_PDX_CACHE_HOME=%CD%\models"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"

if defined USE_CHINA_SOURCES (
  set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
  set "UV_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  set "UV_INDEX_STRATEGY=unsafe-best-match"
  set "PADDLE_PDX_MODEL_SOURCE=modelscope"
  echo china > .ppocr_use_tuna
  echo Using China mirrors: Tsinghua PyPI, PaddlePaddle index, ModelScope models.
) else (
  if not defined PADDLE_PDX_MODEL_SOURCE set "PADDLE_PDX_MODEL_SOURCE=huggingface"
  echo Using global package/model sources.
)

echo Ensuring Python 3.11 is available...
uv python install 3.11
if errorlevel 1 goto :error

if exist ".ppocr_no_sync" (
  echo Acceleration environment marker found; checking runtime packages without uv sync...
  uv run --no-sync python -c "import PIL, paddleocr, paddle" >nul 2>nul
  if errorlevel 1 (
    echo Installing missing runtime packages into existing environment...
    uv pip install "pillow>=10,<13" "paddleocr[doc-parser]>=3.6.0,<4"
    if errorlevel 1 goto :error
    uv run --no-sync python -c "import paddle" >nul 2>nul
    if errorlevel 1 (
      uv pip install "paddlepaddle>=3.2.1,<4"
      if errorlevel 1 goto :error
    )
  )
) else (
  echo Syncing base dependencies...
  uv sync
  if errorlevel 1 goto :error
)

nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo NVIDIA GPU detected; ensuring PaddlePaddle GPU package is installed...
  set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/"
  set "PADDLE_GPU_WHEEL_BASE=https://paddle-whl.bj.bcebos.com/stable/cu126/"
  set "PADDLE_GPU_WHEEL_NAME=paddlepaddle_gpu-3.3.1-cp311-cp311-win_amd64.whl"
  set "PADDLE_GPU_WHEEL_URL=!PADDLE_GPU_WHEEL_BASE!paddlepaddle-gpu/!PADDLE_GPU_WHEEL_NAME!"
  set "PADDLE_GPU_WHEEL_FILE=%CD%\temp\!PADDLE_GPU_WHEEL_NAME!"
  set "PADDLE_GPU_RUNTIME_PACKAGE="
  nvidia-smi 2>nul | findstr /C:"CUDA Version: 13" >nul
  if not errorlevel 1 (
    set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu130/"
    set "PADDLE_GPU_WHEEL_BASE=https://paddle-whl.bj.bcebos.com/stable/cu130/"
    set "PADDLE_GPU_WHEEL_URL=!PADDLE_GPU_WHEEL_BASE!paddlepaddle-gpu/!PADDLE_GPU_WHEEL_NAME!"
    set "PADDLE_GPU_RUNTIME_PACKAGE=nvidia-cublas"
  )

  uv run python -c "import paddle, sys; sys.exit(0 if paddle.is_compiled_with_cuda() else 1)" >nul 2>nul
  if errorlevel 1 (
    uv pip uninstall paddlepaddle
    call :download_paddle_gpu_wheel
    if errorlevel 1 goto :error
    uv pip install !PADDLE_GPU_WHEEL_FILE!
    if errorlevel 1 goto :error
    if defined PADDLE_GPU_RUNTIME_PACKAGE (
      uv pip install !PADDLE_GPU_RUNTIME_PACKAGE! -i !PADDLE_GPU_INDEX!
      if errorlevel 1 goto :error
    )
    echo gpu > .ppocr_no_sync
  )

  nvidia-smi 2>nul | findstr /C:"CUDA Version: 13" >nul
  if not errorlevel 1 (
    uv run --no-sync python -c "import pathlib, sys; root=pathlib.Path(sys.prefix)/'Lib'/'site-packages'/'nvidia'; raise SystemExit(0 if root.exists() and any(root.rglob('cublasLt64_13.dll')) else 1)" >nul 2>nul
    if errorlevel 1 (
      uv pip install nvidia-cublas -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
      if errorlevel 1 goto :error
    )
  )
)

echo Ensuring local OCR models are downloaded...
if exist ".ppocr_no_sync" (
  uv run --no-sync python scripts\models\download_models.py
) else (
  uv run python scripts\models\download_models.py
)
if errorlevel 1 goto :error

echo Bootstrap complete.
exit /b 0

:download_paddle_gpu_wheel
if not exist "temp" mkdir "temp"
for /L %%A in (1,1,5) do (
  echo Downloading PaddlePaddle GPU wheel attempt %%A/5...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '!PADDLE_GPU_WHEEL_URL!' -OutFile '!PADDLE_GPU_WHEEL_FILE!'"
  if not errorlevel 1 (
    uv run python -c "import sys, zipfile; p=sys.argv[1]; exec('try:\n z=zipfile.ZipFile(p)\n bad=z.testzip()\n z.close()\n raise SystemExit(1 if bad else 0)\nexcept Exception:\n raise SystemExit(1)')" "!PADDLE_GPU_WHEEL_FILE!"
    if not errorlevel 1 exit /b 0
  )
  echo PaddlePaddle GPU wheel download was incomplete; retrying...
  del /q "!PADDLE_GPU_WHEEL_FILE!" >nul 2>nul
  timeout /t 3 /nobreak >nul
)
exit /b 1

:error
echo Bootstrap failed. Check the error above, then run start.bat again.
pause
exit /b 1
