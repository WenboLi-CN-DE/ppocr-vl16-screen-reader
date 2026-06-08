@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\..\.."
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

set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/"
set "PADDLE_GPU_WHEEL_NAME=paddlepaddle_gpu-3.3.1-cp311-cp311-win_amd64.whl"
set "PADDLE_GPU_WHEEL_URL=!PADDLE_GPU_INDEX!paddlepaddle-gpu/!PADDLE_GPU_WHEEL_NAME!"
set "PADDLE_GPU_WHEEL_FILE=%CD%\temp\!PADDLE_GPU_WHEEL_NAME!"
set "PADDLE_GPU_RUNTIME_PACKAGE="
nvidia-smi 2>nul | findstr /C:"CUDA Version: 13" >nul
if not errorlevel 1 (
  set "PADDLE_GPU_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu130/"
  set "PADDLE_GPU_WHEEL_URL=!PADDLE_GPU_INDEX!paddlepaddle-gpu/!PADDLE_GPU_WHEEL_NAME!"
  set "PADDLE_GPU_RUNTIME_PACKAGE=nvidia-cublas"
)

echo Using !PADDLE_GPU_WHEEL_URL!

echo Syncing base dependencies first. This prevents missing packages such as PIL/Pillow.
uv sync
if errorlevel 1 goto :error

uv pip uninstall paddlepaddle
if errorlevel 1 goto :error

if defined USE_TUNA (
  call :download_paddle_gpu_wheel
  if errorlevel 1 goto :error
  uv pip install !PADDLE_GPU_WHEEL_FILE!
) else (
  call :download_paddle_gpu_wheel
  if errorlevel 1 goto :error
  uv pip install !PADDLE_GPU_WHEEL_FILE!
)
if errorlevel 1 goto :error

if defined PADDLE_GPU_RUNTIME_PACKAGE (
  uv pip install !PADDLE_GPU_RUNTIME_PACKAGE! -i !PADDLE_GPU_INDEX!
  if errorlevel 1 goto :error
)

echo gpu > .ppocr_no_sync
echo Windows GPU acceleration package installed.
echo Run start.bat again; it will use GPU automatically when nvidia-smi is available.
pause
exit /b 0

:download_paddle_gpu_wheel
if not exist "temp" mkdir "temp"
for /L %%A in (1,1,5) do (
  echo Downloading PaddlePaddle GPU wheel attempt %%A/5...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '!PADDLE_GPU_WHEEL_URL!' -OutFile '!PADDLE_GPU_WHEEL_FILE!'"
  if not errorlevel 1 (
    uv run python -c "import sys, zipfile; z=zipfile.ZipFile(sys.argv[1]); bad=z.testzip(); z.close(); raise SystemExit(1 if bad else 0)" "!PADDLE_GPU_WHEEL_FILE!"
    if not errorlevel 1 exit /b 0
  )
  echo PaddlePaddle GPU wheel download was incomplete; retrying...
  del /q "!PADDLE_GPU_WHEEL_FILE!" >nul 2>nul
  timeout /t 3 /nobreak >nul
)
exit /b 1

:error
echo Installation failed. Check the error above, then run this script again.
pause
exit /b 1
