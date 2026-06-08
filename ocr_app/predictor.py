import gc

from .profiles import emergency_predict_kwargs
from .runtime import is_cuda_out_of_memory_error


def _release_gpu_memory():
    """Release cached GPU memory after inference to prevent VRAM accumulation."""
    gc.collect()
    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


class OcrPredictor:
    def __init__(self, get_engine, get_engine_kwargs, create_engine, set_engine):
        self.get_engine = get_engine
        self.get_engine_kwargs = get_engine_kwargs
        self.create_engine = create_engine
        self.set_engine = set_engine
        self._transient_engines = []

    def predict(self, image_path, predict_kwargs):
        try:
            result = self.get_engine().predict(image_path, **predict_kwargs)
            _release_gpu_memory()
            return result
        except Exception as error:
            if not self.can_retry_after_gpu_oom(error):
                raise

        emergency_kwargs = emergency_predict_kwargs(predict_kwargs)
        try:
            print("GPU 显存不足,使用更保守 GPU 参数重试本次识别")
            _release_gpu_memory()
            result = self.get_engine().predict(image_path, **emergency_kwargs)
            _release_gpu_memory()
            return result
        except Exception as error:
            if not self.can_retry_after_gpu_oom(error):
                raise

        print("GPU 显存仍不足,仅本次识别切换 CPU 重试")
        _release_gpu_memory()
        cpu_kwargs = self.get_engine_kwargs().copy()
        cpu_kwargs["device"] = "cpu"
        cpu_engine = self.create_engine(cpu_kwargs)
        self._transient_engines.append(cpu_engine)
        result = cpu_engine.predict(image_path, **emergency_kwargs)
        _release_gpu_memory()
        return result

    def can_retry_after_gpu_oom(self, error):
        kwargs = self.get_engine_kwargs()
        return kwargs.get("device") == "gpu" and is_cuda_out_of_memory_error(error)
