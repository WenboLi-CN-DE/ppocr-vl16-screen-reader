from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_start_bootstraps_then_launches():
    script = (ROOT / "start.bat").read_text(encoding="utf-8")

    assert r"call scripts\bootstrap\bootstrap.bat" in script
    assert "uv run --no-sync python launcher.py" in script
    assert "UV_HTTP_TIMEOUT=300" in script
    assert 'if exist ".venv"' not in script


def test_macos_start_bootstraps_then_launches():
    script = (ROOT / "start.command").read_text(encoding="utf-8")

    assert './scripts/bootstrap/bootstrap.command "$@"' in script
    assert "uv run --no-sync python launcher.py" in script
    assert "UV_HTTP_TIMEOUT" in script
    assert "300" in script


def test_bootstrap_scripts_default_to_china_sources_and_download_models():
    windows_script = (ROOT / "scripts" / "bootstrap" / "bootstrap.bat").read_text(encoding="utf-8")
    mac_script = (ROOT / "scripts" / "bootstrap" / "bootstrap.command").read_text(encoding="utf-8")

    for script in (windows_script, mac_script):
        assert "UV_DEFAULT_INDEX" in script
        assert "https://pypi.tuna.tsinghua.edu.cn/simple" in script
        assert "UV_INDEX_STRATEGY" in script
        assert "unsafe-best-match" in script
        assert "UV_HTTP_TIMEOUT" in script
        assert "300" in script
        assert "PADDLE_PDX_MODEL_SOURCE" in script
        assert "modelscope" in script
        assert "PADDLE_PDX_CACHE_HOME" in script
        assert "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" in script
        assert "True" in script
        assert "uv python install 3.11" in script
        assert "uv sync" in script
        assert "scripts/models/download_models.py" in script or r"scripts\models\download_models.py" in script

    assert "https://www.paddlepaddle.org.cn/packages/stable/cpu/" in windows_script
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu130/" in windows_script
    assert "https://www.paddlepaddle.org.cn/packages/stable/cu126/" in windows_script
    assert "paddlepaddle-gpu==3.3.0" in windows_script
    assert "nvidia-cublas" in windows_script
    assert "nvidia-smi" in windows_script
    assert "mlx-vlm>=0.3.11" in mac_script


def test_acceleration_installers_sync_base_dependencies_before_no_sync_marker():
    windows_script = (ROOT / "scripts" / "install" / "install_windows_gpu.bat").read_text(encoding="utf-8")
    mac_script = (ROOT / "scripts" / "install" / "install_macos_accel.command").read_text(encoding="utf-8")

    assert "uv sync" in windows_script
    assert "UV_HTTP_TIMEOUT=300" in windows_script
    assert ".ppocr_no_sync" in windows_script
    assert ".ppocr_use_tuna" in windows_script
    assert "PADDLE_GPU_PACKAGE" in windows_script
    assert "paddlepaddle-gpu==3.3.0" in windows_script
    assert "uv pip uninstall paddlepaddle" in windows_script
    assert "uv pip uninstall -y" not in windows_script
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


def test_model_downloader_defaults_to_project_cache_and_modelscope():
    script = (ROOT / "scripts" / "models" / "download_models.py").read_text(encoding="utf-8")

    assert 'PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE_HOME)' in script
    assert 'PADDLE_PDX_MODEL_SOURCE", "modelscope"' in script
    assert 'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True"' in script
    assert '"PaddleOCR-VL-1.6"' in script
