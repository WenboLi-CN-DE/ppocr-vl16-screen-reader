@echo off
setlocal
cd /d "%~dp0"
set "UV_HTTP_TIMEOUT=300"

set "USE_TUNA="
if /I "%~1"=="tuna" set "USE_TUNA=1"
if /I "%~1"=="--tuna" set "USE_TUNA=1"
if exist ".ppocr_use_tuna" set "USE_TUNA=1"

if defined USE_TUNA (
  set "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
  set "UV_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/"
  set "UV_INDEX_STRATEGY=unsafe-best-match"
  echo tuna > .ppocr_use_tuna
  echo Using Tsinghua PyPI mirror for base dependencies.
)

echo This installs the PaddlePaddle GPU package for NVIDIA Windows machines.
echo If nvidia-smi reports CUDA Version 13.x, the CUDA 13.0 Paddle package will be used.
pause

set "PADDLE_GPU_PACKAGE=paddlepaddle-gpu==3.3.0"
set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/"
set "PADDLE_GPU_RUNTIME_PACKAGE="
nvidia-smi 2>nul | findstr /C:"CUDA Version: 13" >nul
if not errorlevel 1 (
  set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu130/"
  set "PADDLE_GPU_RUNTIME_PACKAGE=nvidia-cublas"
)

echo Using %PADDLE_GPU_PACKAGE% from %PADDLE_GPU_INDEX%

echo Syncing base dependencies first. This prevents missing packages such as PIL/Pillow.
uv sync
if errorlevel 1 goto :error

uv pip uninstall -y paddlepaddle
if errorlevel 1 goto :error

if defined USE_TUNA (
  uv pip install %PADDLE_GPU_PACKAGE% -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url %PADDLE_GPU_INDEX%
) else (
  uv pip install %PADDLE_GPU_PACKAGE% -i %PADDLE_GPU_INDEX%
)
if errorlevel 1 goto :error

if defined PADDLE_GPU_RUNTIME_PACKAGE (
  uv pip install %PADDLE_GPU_RUNTIME_PACKAGE% -i %PADDLE_GPU_INDEX%
  if errorlevel 1 goto :error
)

echo gpu > .ppocr_no_sync
echo Windows GPU acceleration package installed.
echo Run start.bat again; it will use GPU automatically when nvidia-smi is available.
pause
exit /b 0

:error
echo Installation failed. Check the error above, then run this script again.
pause
exit /b 1
