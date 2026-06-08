from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_start_syncs_dependencies_unless_acceleration_marker_exists():
    script = (ROOT / "start.bat").read_text(encoding="utf-8")

    assert 'if exist ".ppocr_no_sync"' in script
    assert 'if exist ".ppocr_use_tuna"' in script
    assert "UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple" in script
    assert "UV_INDEX_STRATEGY" in script
    assert "unsafe-best-match" in script
    assert "UV_HTTP_TIMEOUT=300" in script
    assert "import PIL, paddleocr, paddle" in script
    assert 'uv pip install "pillow>=10,<13" "paddleocr[doc-parser]>=3.6.0,<4"' in script
    assert "PADDLE_GPU_PACKAGE" in script
    assert "paddlepaddle-gpu==3.3.0" in script
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu130/" in script
    assert "nvidia-cublas" in script
    assert "cublasLt64_13.dll" in script
    assert 'uv pip install "paddlepaddle>=3.2.1,<4"' in script
    assert "uv run python launcher.py" in script
    assert "uv run --no-sync python launcher.py" in script
    assert 'if exist ".venv"' not in script


def test_macos_start_supports_tuna_marker_and_no_sync_marker():
    script = (ROOT / "start.command").read_text(encoding="utf-8")

    assert 'if [ -f ".ppocr_use_tuna" ]; then' in script
    assert "UV_DEFAULT_INDEX" in script
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in script
    assert "UV_INDEX_STRATEGY" in script
    assert "unsafe-best-match" in script
    assert "UV_HTTP_TIMEOUT" in script
    assert "300" in script
    assert "import PIL, paddleocr, paddle" in script
    assert 'uv pip install "pillow>=10,<13" "paddleocr[doc-parser]>=3.6.0,<4"' in script
    assert 'uv pip install "paddlepaddle>=3.2.1,<4"' in script
    assert 'if [ -f ".ppocr_no_sync" ]; then' in script
    assert "uv run python launcher.py" in script
    assert "uv run --no-sync python launcher.py" in script


def test_acceleration_installers_sync_base_dependencies_before_no_sync_marker():
    windows_script = (ROOT / "install_windows_gpu.bat").read_text(encoding="utf-8")
    mac_script = (ROOT / "install_macos_accel.command").read_text(encoding="utf-8")

    assert "uv sync" in windows_script
    assert "UV_HTTP_TIMEOUT=300" in windows_script
    assert ".ppocr_no_sync" in windows_script
    assert ".ppocr_use_tuna" in windows_script
    assert "PADDLE_GPU_PACKAGE" in windows_script
    assert "paddlepaddle-gpu==3.3.0" in windows_script
    assert "CUDA Version: 13" in windows_script
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu130/" in windows_script
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu126/" in windows_script
    assert "nvidia-cublas" in windows_script

    assert "uv sync" in mac_script
    assert "UV_HTTP_TIMEOUT" in mac_script
    assert "300" in mac_script
    assert ".ppocr_no_sync" in mac_script
    assert ".ppocr_use_tuna" in mac_script
    assert "mlx-vlm>=0.3.11" in mac_script
