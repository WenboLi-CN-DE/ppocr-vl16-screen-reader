import os
from pathlib import Path

from .platform import is_url_port_open


def require_runtime_dependencies(runtime_import_error):
    """Ensure OCR runtime dependencies are installed."""
    if runtime_import_error:
        raise RuntimeError(f"运行依赖缺失: {runtime_import_error}. 请先执行: uv sync")


def use_gpu_low_memory_mode(environ=None):
    if environ is None:
        environ = os.environ
    return (
        environ.get("PPOCR_DEVICE", "").strip().lower() == "gpu"
        and environ.get("PPOCR_GPU_MEMORY_MODE", "").strip().lower() in {"1", "true", "low"}
    )


def is_cuda_out_of_memory_error(error):
    text = str(error).lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cuda error(2)" in text
    )


def local_model_kwargs(base_dir):
    model_root = Path(base_dir) / "models" / "official_models"
    layout_dir = model_root / "PP-DocLayoutV3"
    vl_dir = model_root / "PaddleOCR-VL-1.6"
    legacy_vl_dir = model_root / "PaddleOCR-VL-1.6-0.9B"
    kwargs = {}
    if layout_dir.exists():
        kwargs["layout_detection_model_dir"] = str(layout_dir)
    if vl_dir.exists():
        kwargs["vl_rec_model_dir"] = str(vl_dir)
    elif legacy_vl_dir.exists():
        kwargs["vl_rec_model_dir"] = str(legacy_vl_dir)
    return kwargs


def vl_model_api_name(base_dir):
    model_dir = Path(base_dir) / "models" / "official_models" / "PaddleOCR-VL-1.6"
    if model_dir.exists():
        return str(model_dir)
    return "PaddlePaddle/PaddleOCR-VL-1.6"


def acceleration_kwargs(base_dir, environ=None, port_open=is_url_port_open):
    if environ is None:
        environ = os.environ
    kwargs = {}

    device = environ.get("PPOCR_DEVICE")
    if device:
        kwargs["device"] = device

    engine = environ.get("PPOCR_ENGINE")
    if engine:
        kwargs["engine"] = engine

    precision = environ.get("PPOCR_PRECISION")
    if precision:
        kwargs["precision"] = precision

    accel = environ.get("PPOCR_ACCEL", "auto").strip().lower()
    if accel in {"mlx", "mlx-vlm", "mlx-vlm-server"}:
        server_url = environ.get("PPOCR_MLX_SERVER_URL", "http://127.0.0.1:8111/")
        if port_open(server_url):
            kwargs.update(
                {
                    "vl_rec_backend": "mlx-vlm-server",
                    "vl_rec_server_url": server_url,
                    "vl_rec_api_model_name": environ.get(
                        "PPOCR_MLX_MODEL_NAME", vl_model_api_name(base_dir)
                    ),
                }
            )
        else:
            print(f"MLX-VLM 服务不可用,回退本地推理: {server_url}")

    return kwargs
