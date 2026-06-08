import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


MLX_HOST = "127.0.0.1"
MLX_PORT = 8111
MLX_URL = f"http://{MLX_HOST}:{MLX_PORT}/"

QUIET_OUTPUT_PATTERNS = [
    re.compile(r"^Creating model:"),
    re.compile(r"^信息: 用提供的模式无法找到文件。"),
    re.compile(r"^WARNING: Logging before InitGoogleLogging"),
    re.compile(r"^[WEI]\d{4}\s+.*\.cc:\d+\]"),
    re.compile(r"^use GQA - "),
    re.compile(r"^Loaded weights file from disk"),
    re.compile(r"^All model checkpoint weights were used"),
    re.compile(r"^All the weights of "),
    re.compile(r"^If your task is similar to "),
    re.compile(r"^Loading configuration file "),
    re.compile(r"^Loading weights file "),
    re.compile(r"^Bucketed engine_config has no entry "),
    re.compile(r".*\\creation\.py:\d+: UserWarning:"),
    re.compile(r"^\s*return tensor\("),
    re.compile(r"^\s*warnings\.warn\(warning_message\)"),
    re.compile(r".*\\extension_utils\.py:\d+: UserWarning:"),
]

VISIBLE_OUTPUT_KEYWORDS = [
    "PaddleOCR-VL 1.6 初始化成功",
    "当前工作目录:",
    "选择区域:",
    "截图尺寸:",
    "保留原文分段截图尺寸:",
    "快速模式截图尺寸:",
    "OCR推理完成",
    "OCR识别文本行数:",
    "GPU 显存不足",
    "GPU 初始化失败",
    "未检测到 mlx_vlm.server",
    "OCR识别失败",
    "OCR识别完整错误",
    "结果处理失败",
    "完整错误信息",
    "Traceback",
    "CUDA error",
    "out of memory",
    "Could not locate",
    "Invalid handle",
    "运行依赖缺失",
]


def command_exists(command):
    return shutil.which(command) is not None


def port_open(host, port, timeout=0.25):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_background_server(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def should_show_ocr_output(line):
    stripped = line.strip()
    if not stripped:
        return False
    if any(keyword in stripped for keyword in VISIBLE_OUTPUT_KEYWORDS):
        return True
    return not any(pattern.search(stripped) for pattern in QUIET_OUTPUT_PATTERNS)


def run_ocr_process(command, env):
    if env.get("PPOCR_VERBOSE", "").strip().lower() in {"1", "true", "yes"}:
        return subprocess.run(command, check=True).returncode

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if should_show_ocr_output(line):
            print(line, end="")
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def local_vl_model_name(base_dir):
    model_dir = Path(base_dir) / "models" / "official_models" / "PaddleOCR-VL-1.6"
    if model_dir.exists():
        return str(model_dir)
    return "PaddlePaddle/PaddleOCR-VL-1.6"


def configure_acceleration(
    *,
    base_dir,
    env,
    system_name,
    machine,
    command_exists,
    port_open,
    start_server,
):
    if env.get("PPOCR_ACCEL") or env.get("PPOCR_DEVICE"):
        return

    system_name = system_name.lower()
    machine = machine.lower()

    if system_name == "darwin" and machine in {"arm64", "aarch64"}:
        if command_exists("mlx_vlm.server"):
            if not port_open(MLX_HOST, MLX_PORT):
                start_server(
                    ["mlx_vlm.server", "--port", str(MLX_PORT)],
                    Path(base_dir) / "temp" / "mlx_vlm_server.log",
                )
                for _ in range(30):
                    if port_open(MLX_HOST, MLX_PORT):
                        break
                    time.sleep(0.5)
            env["PPOCR_ACCEL"] = "mlx"
            env["PPOCR_MLX_SERVER_URL"] = MLX_URL
            env["PPOCR_MLX_MODEL_NAME"] = local_vl_model_name(base_dir)
        else:
            print("未检测到 mlx_vlm.server,macOS 将使用 Paddle 本地推理。")
            print('如需 Apple Silicon 加速: uv pip install "mlx-vlm>=0.3.11"')
        return

    if system_name == "windows" and command_exists("nvidia-smi"):
        env["PPOCR_DEVICE"] = "gpu"
        env.setdefault("PPOCR_GPU_MEMORY_MODE", "low")
        return


def main():
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    configure_acceleration(
        base_dir=base_dir,
        env=os.environ,
        system_name=platform.system(),
        machine=platform.machine(),
        command_exists=command_exists,
        port_open=port_open,
        start_server=start_background_server,
    )
    run_ocr_process([sys.executable, "ocr.py"], os.environ)


if __name__ == "__main__":
    main()
