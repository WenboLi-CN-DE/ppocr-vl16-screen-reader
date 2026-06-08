from pathlib import Path

import launcher


def test_launcher_configures_mlx_on_apple_silicon_when_server_command_exists(tmp_path):
    started = []
    port_checks = []
    env = {}
    model_dir = tmp_path / "models" / "official_models" / "PaddleOCR-VL-1.6"
    model_dir.mkdir(parents=True)

    def fake_port_open(host, port):
        port_checks.append((host, port))
        return bool(started)

    launcher.configure_acceleration(
        base_dir=tmp_path,
        env=env,
        system_name="Darwin",
        machine="arm64",
        command_exists=lambda name: name == "mlx_vlm.server",
        port_open=fake_port_open,
        start_server=lambda command, log_path: started.append((command, log_path)),
    )

    assert env["PPOCR_ACCEL"] == "mlx"
    assert env["PPOCR_MLX_SERVER_URL"] == "http://127.0.0.1:8111/"
    assert env["PPOCR_MLX_MODEL_NAME"] == str(model_dir)
    assert started == [(["mlx_vlm.server", "--port", "8111"], tmp_path / "temp" / "mlx_vlm_server.log")]


def test_launcher_configures_windows_gpu_when_nvidia_smi_exists(tmp_path):
    env = {}

    launcher.configure_acceleration(
        base_dir=tmp_path,
        env=env,
        system_name="Windows",
        machine="AMD64",
        command_exists=lambda name: name == "nvidia-smi",
        port_open=lambda host, port: False,
        start_server=lambda command, log_path: None,
    )

    assert env["PPOCR_DEVICE"] == "gpu"
    assert env["PPOCR_GPU_MEMORY_MODE"] == "low"


def test_launcher_keeps_existing_gpu_memory_mode(tmp_path):
    env = {"PPOCR_GPU_MEMORY_MODE": "normal"}

    launcher.configure_acceleration(
        base_dir=tmp_path,
        env=env,
        system_name="Windows",
        machine="AMD64",
        command_exists=lambda name: name == "nvidia-smi",
        port_open=lambda host, port: False,
        start_server=lambda command, log_path: None,
    )

    assert env["PPOCR_DEVICE"] == "gpu"
    assert env["PPOCR_GPU_MEMORY_MODE"] == "normal"


def test_launcher_keeps_cpu_fallback_when_no_accelerator_exists(tmp_path):
    env = {}

    launcher.configure_acceleration(
        base_dir=tmp_path,
        env=env,
        system_name="Darwin",
        machine="x86_64",
        command_exists=lambda name: False,
        port_open=lambda host, port: False,
        start_server=lambda command, log_path: None,
    )

    assert "PPOCR_ACCEL" not in env
    assert "PPOCR_DEVICE" not in env


def test_launcher_filters_noisy_third_party_output():
    noisy_lines = [
        "Creating model: ('PP-DocLayoutV3', 'path', None)",
        "信息: 用提供的模式无法找到文件。",
        "WARNING: Logging before InitGoogleLogging() is written to STDERR",
        "W0608 20:13:11.612800 gpu_resources.cc:116] Please NOTE: device: 0",
        "use GQA - num_heads: 16- num_key_value_heads: 2",
        "Loaded weights file from disk, setting weights to model.",
        "E:\\path\\creation.py:1152: UserWarning: To copy construct from a tensor",
        "Bucketed engine_config has no entry for resolved engine 'paddle_dynamic'; using an empty config for that engine.",
    ]

    assert [launcher.should_show_ocr_output(line) for line in noisy_lines] == [False] * len(noisy_lines)


def test_launcher_keeps_app_status_and_errors_visible():
    visible_lines = [
        "PaddleOCR-VL 1.6 初始化成功",
        "当前工作目录: E:\\personal_project\\ppocr_v1.6",
        "选择区域: (1, 2, 3, 4)",
        "OCR推理完成, 结果条目数: 1",
        "OCR识别文本行数: 3",
        "GPU 显存不足,自动切换 CPU 重试本次识别",
        "OCR识别失败: CUDA error",
        "Traceback (most recent call last):",
        "Could not locate cublasLt64_13.dll",
    ]

    assert [launcher.should_show_ocr_output(line) for line in visible_lines] == [True] * len(visible_lines)
