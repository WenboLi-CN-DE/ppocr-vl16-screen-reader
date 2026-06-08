import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_CACHE_HOME = BASE_DIR / "models"
MODELS = ("PP-DocLayoutV3", "PaddleOCR-VL-1.6")


def main():
    MODEL_CACHE_HOME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODEL_CACHE_HOME))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")

    from paddlex.inference.utils.official_models import official_models

    for model_name in MODELS:
        print(f"Downloading {model_name} ...")
        model_path = official_models.get_model_path(model_name)
        print(f"{model_name}: {model_path}")

    print(f"Local models are stored under: {MODEL_CACHE_HOME / 'official_models'}")


if __name__ == "__main__":
    main()
