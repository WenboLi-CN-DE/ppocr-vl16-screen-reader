import sys
import types
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


pil_module = types.ModuleType("PIL")
pil_module.Image = object()
pil_module.ImageGrab = types.SimpleNamespace(grab=lambda **kwargs: None)
pil_module.ImageTk = types.SimpleNamespace(PhotoImage=lambda image: image)
sys.modules.setdefault("PIL", pil_module)

paddleocr_module = types.ModuleType("paddleocr")
paddleocr_module.PaddleOCRVL = object
sys.modules.setdefault("paddleocr", paddleocr_module)
