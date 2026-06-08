import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


MLX_HOST = "127.0.0.1"
MLX_PORT = 8111
MLX_URL = f"http://{MLX_HOST}:{MLX_PORT}/"


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
    subprocess.run([sys.executable, "ocr.py"], check=True)


if __name__ == "__main__":
    main()
