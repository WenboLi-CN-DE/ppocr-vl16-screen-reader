@echo off
cd /d "%~dp0"
set "UV_HTTP_TIMEOUT=300"

call scripts\bootstrap\bootstrap.bat %*
if errorlevel 1 goto :error
uv run --no-sync python launcher.py
pause
exit /b 0

:error
echo Failed to start PaddleOCR-VL screen reader. Check the error above, then run start.bat again.
pause
exit /b 1
