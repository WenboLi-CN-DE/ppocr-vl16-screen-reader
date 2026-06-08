import ctypes
import html
import os
import re
import socket
import queue
import site
import sys
import sysconfig
import tempfile
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import messagebox
from urllib.parse import urlparse


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_minloglevel", "2")
warnings.filterwarnings("ignore", message=r"No ccache found.*")
warnings.filterwarnings("ignore", message=r"To copy construct from a tensor.*")

WINDOWS_NVIDIA_DLL_HANDLES = []


def windows_site_package_roots():
    roots = {
        Path(sys.prefix) / "Lib" / "site-packages",
        Path(sysconfig.get_paths().get("purelib", "")),
        Path(sysconfig.get_paths().get("platlib", "")),
    }
    try:
        roots.update(Path(path) for path in site.getsitepackages())
    except AttributeError:
        pass
    try:
        roots.add(Path(site.getusersitepackages()))
    except AttributeError:
        pass
    return [root for root in roots if str(root) and root.exists()]


def configure_windows_nvidia_dll_paths(
    *,
    site_package_roots=None,
    add_dll_directory=None,
    load_library=None,
    environ=None,
    os_name=None,
):
    """Expose CUDA DLLs installed by NVIDIA Python wheels on Windows."""
    if os_name is None:
        os_name = os.name
    if os_name != "nt":
        return []
    if site_package_roots is None:
        site_package_roots = windows_site_package_roots()
    if add_dll_directory is None:
        add_dll_directory = getattr(os, "add_dll_directory", None)
    if load_library is None:
        load_library = ctypes.WinDLL
    if environ is None:
        environ = os.environ

    dll_dirs = []
    preload_dlls = []
    for root in site_package_roots:
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.exists():
            continue
        for dll_path in sorted(nvidia_root.rglob("*.dll")):
            dll_dir = str(dll_path.parent)
            if dll_dir not in dll_dirs:
                dll_dirs.append(dll_dir)
            if dll_path.name.lower() == "cublaslt64_13.dll":
                preload_dlls.append(str(dll_path))

    path_separator = ";" if os_name == "nt" else os.pathsep
    existing_path = environ.get("PATH", "")
    existing_parts = {part.lower() for part in existing_path.split(path_separator) if part}
    new_parts = [path for path in dll_dirs if path.lower() not in existing_parts]
    if new_parts:
        environ["PATH"] = path_separator.join(new_parts + ([existing_path] if existing_path else []))

    if add_dll_directory is not None:
        for path in new_parts:
            handle = add_dll_directory(path)
            if handle is not None:
                WINDOWS_NVIDIA_DLL_HANDLES.append(handle)

    for dll_path in preload_dlls:
        try:
            WINDOWS_NVIDIA_DLL_HANDLES.append(load_library(dll_path))
        except OSError:
            pass

    return dll_dirs


configure_windows_nvidia_dll_paths()

RUNTIME_IMPORT_ERROR = None

try:
    from PIL import Image, ImageGrab, ImageTk
except ImportError as error:
    Image = None
    ImageGrab = None
    ImageTk = None
    RUNTIME_IMPORT_ERROR = error

try:
    from paddleocr import PaddleOCRVL
except ImportError as error:
    PaddleOCRVL = None
    RUNTIME_IMPORT_ERROR = RUNTIME_IMPORT_ERROR or error


FAST_OCR_PREDICT_KWARGS = {
    "use_layout_detection": False,
    "prompt_label": "ocr",
    "max_new_tokens": 1536,
    "max_pixels": 1280 * 28 * 28,
}
PRESERVE_LAYOUT_PREDICT_KWARGS = {
    "use_layout_detection": True,
    "merge_layout_blocks": True,
    "format_block_content": False,
    "max_new_tokens": 1536,
    "max_pixels": 1280 * 28 * 28,
}
MAX_FAST_IMAGE_EDGE = 1536
LOW_MEMORY_FAST_OCR_PREDICT_KWARGS = {
    **FAST_OCR_PREDICT_KWARGS,
    "max_new_tokens": 768,
    "max_pixels": 768 * 28 * 28,
}
LOW_MEMORY_PRESERVE_LAYOUT_PREDICT_KWARGS = {
    **PRESERVE_LAYOUT_PREDICT_KWARGS,
    "max_new_tokens": 768,
    "max_pixels": 768 * 28 * 28,
}
MAX_LOW_MEMORY_IMAGE_EDGE = 1024

if Image is not None:
    RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", None), "LANCZOS", None)
    if RESAMPLE_LANCZOS is None:
        RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", None)
else:
    RESAMPLE_LANCZOS = None


def require_runtime_dependencies():
    """Ensure OCR runtime dependencies are installed."""
    if RUNTIME_IMPORT_ERROR:
        raise RuntimeError(f"运行依赖缺失: {RUNTIME_IMPORT_ERROR}. 请先执行: uv sync")


def html_to_plain_text(text):
    if not isinstance(text, str):
        return text
    if "<" not in text or ">" not in text:
        return html.unescape(text)
    cleaned = text
    cleaned = re.sub(r"(?i)</tr\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</p\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)</div\s*>", "\n\n", cleaned)
    cleaned = re.sub(r"(?i)</td\s*>\s*<td[^>]*>", " ", cleaned)
    cleaned = re.sub(r"(?i)<[^>]+>", "", cleaned)
    cleaned = html.unescape(cleaned)
    return cleaned.strip()


def normalize_text_lines(text):
    if not isinstance(text, str):
        return []
    text = html_to_plain_text(text)
    normalized_lines = []
    paragraph_lines = []

    def flush_paragraph():
        if not paragraph_lines:
            return
        normalized_lines.append(" ".join(paragraph_lines))
        paragraph_lines.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            paragraph_lines.append(line)
        else:
            flush_paragraph()
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")

    flush_paragraph()

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()
    return normalized_lines


def resize_for_fast_ocr(image, max_edge=MAX_FAST_IMAGE_EDGE):
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_edge:
        return image
    scale = max_edge / longest_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, RESAMPLE_LANCZOS)


def use_gpu_low_memory_mode():
    return (
        os.environ.get("PPOCR_DEVICE", "").strip().lower() == "gpu"
        and os.environ.get("PPOCR_GPU_MEMORY_MODE", "").strip().lower() in {"1", "true", "low"}
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


def is_url_port_open(url, timeout=0.25):
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def vl_model_api_name(base_dir):
    model_dir = Path(base_dir) / "models" / "official_models" / "PaddleOCR-VL-1.6"
    if model_dir.exists():
        return str(model_dir)
    return "PaddlePaddle/PaddleOCR-VL-1.6"


def acceleration_kwargs(base_dir):
    kwargs = {}

    device = os.environ.get("PPOCR_DEVICE")
    if device:
        kwargs["device"] = device

    engine = os.environ.get("PPOCR_ENGINE")
    if engine:
        kwargs["engine"] = engine

    precision = os.environ.get("PPOCR_PRECISION")
    if precision:
        kwargs["precision"] = precision

    accel = os.environ.get("PPOCR_ACCEL", "auto").strip().lower()
    if accel in {"mlx", "mlx-vlm", "mlx-vlm-server"}:
        server_url = os.environ.get("PPOCR_MLX_SERVER_URL", "http://127.0.0.1:8111/")
        if is_url_port_open(server_url):
            kwargs.update(
                {
                    "vl_rec_backend": "mlx-vlm-server",
                    "vl_rec_server_url": server_url,
                    "vl_rec_api_model_name": os.environ.get(
                        "PPOCR_MLX_MODEL_NAME", vl_model_api_name(base_dir)
                    ),
                }
            )
        else:
            print(f"MLX-VLM 服务不可用,回退本地推理: {server_url}")

    return kwargs


class DPIHandler:
    """DPI handling for Windows screen coordinates."""

    def __init__(self):
        self.setup_dpi_awareness()

    def setup_dpi_awareness(self):
        if sys.platform != "win32":
            return

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass


class ScreenSelector:
    """Full-screen region selector."""

    def __init__(self, callback, cancel_callback=None):
        self.callback = callback
        self.cancel_callback = cancel_callback
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.selecting = False
        self.selection_window = None
        self.canvas = None
        self.rect_id = None
        self.screen_snapshot = None
        self.screen_preview = None

    def start_selection(self):
        if sys.platform != "darwin":
            try:
                self.screen_snapshot = ImageGrab.grab()
            except Exception as error:
                print(f"屏幕快照生成失败: {error}")
                self.screen_snapshot = None
        self.create_selection_window()

    def build_screen_preview(self, screen_width, screen_height):
        screenshot = self.screen_snapshot or ImageGrab.grab()
        target_size = (screen_width, screen_height)
        if screenshot.size != target_size:
            screenshot = screenshot.resize(target_size)
        return ImageTk.PhotoImage(screenshot)

    def create_selection_window(self):
        self.selection_window = tk.Toplevel()
        self.selection_window.withdraw()
        self.selection_window.overrideredirect(True)
        self.selection_window.attributes("-topmost", True)
        if sys.platform == "darwin":
            self.selection_window.attributes("-alpha", 0.3)
        self.selection_window.configure(bg="black")

        screen_width = self.selection_window.winfo_screenwidth()
        screen_height = self.selection_window.winfo_screenheight()
        self.selection_window.geometry(f"{screen_width}x{screen_height}+0+0")
        if sys.platform != "darwin" and not self.screen_preview:
            try:
                self.screen_preview = self.build_screen_preview(screen_width, screen_height)
            except Exception as error:
                print(f"屏幕预览生成失败: {error}")
                self.screen_preview = None

        self.canvas = tk.Canvas(self.selection_window, highlightthickness=0, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        if self.screen_preview:
            self.canvas.create_image(0, 0, image=self.screen_preview, anchor=tk.NW)
            self.canvas.create_rectangle(
                0,
                0,
                screen_width,
                screen_height,
                fill="black",
                stipple="gray25",
                outline="",
            )

        self.canvas.create_text(
            screen_width // 2,
            50,
            text="拖拽鼠标选择区域,ESC键取消",
            fill="white",
            font=("Arial", 16),
        )
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.selection_window.bind("<Escape>", self.cancel_selection)

        self.selection_window.deiconify()
        self.canvas.focus_set()

    def on_mouse_down(self, event):
        self.start_x = self.selection_window.winfo_pointerx()
        self.start_y = self.selection_window.winfo_pointery()
        self.selecting = True

        if self.rect_id:
            self.canvas.delete(self.rect_id)

    def on_mouse_drag(self, event):
        if not self.selecting:
            return

        current_x = self.selection_window.winfo_pointerx()
        current_y = self.selection_window.winfo_pointery()
        canvas_x1 = self.start_x - self.selection_window.winfo_rootx()
        canvas_y1 = self.start_y - self.selection_window.winfo_rooty()
        canvas_x2 = current_x - self.selection_window.winfo_rootx()
        canvas_y2 = current_y - self.selection_window.winfo_rooty()

        if self.rect_id:
            self.canvas.delete(self.rect_id)

        self.rect_id = self.canvas.create_rectangle(
            canvas_x1,
            canvas_y1,
            canvas_x2,
            canvas_y2,
            outline="red",
            width=2,
        )

    def on_mouse_up(self, event):
        if not self.selecting:
            return

        self.end_x = self.selection_window.winfo_pointerx()
        self.end_y = self.selection_window.winfo_pointery()
        self.selecting = False

        width = abs(self.end_x - self.start_x)
        height = abs(self.end_y - self.start_y)
        if width < 10 or height < 10:
            messagebox.showwarning("警告", "选择区域太小,请重新选择")
            self.cancel_selection()
            return

        left = min(self.start_x, self.end_x)
        top = min(self.start_y, self.end_y)
        right = max(self.start_x, self.end_x)
        bottom = max(self.start_y, self.end_y)

        print(f"选择区域: ({left}, {top}, {right}, {bottom})")
        self.selection_window.destroy()
        self.callback(left, top, right, bottom)

    def cancel_selection(self, event=None):
        if self.selection_window:
            self.selection_window.destroy()
        if self.cancel_callback:
            self.cancel_callback()


class OCRApp:
    """Desktop screenshot OCR application."""

    def __init__(self):
        self.dpi_handler = DPIHandler()
        self.base_dir = Path(__file__).resolve().parent
        self.temp_dir = self.base_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.ui_queue = queue.Queue()

        self.create_main_window()
        self.setup_ocr()
        self.root.after(100, self.process_ui_queue)

    def setup_ocr(self):
        try:
            require_runtime_dependencies()
            kwargs = {"pipeline_version": "v1.6"}
            base_dir = getattr(self, "base_dir", Path.cwd())
            if hasattr(self, "base_dir"):
                kwargs.update(local_model_kwargs(self.base_dir))
            accel_kwargs = acceleration_kwargs(base_dir)
            if "vl_rec_backend" in accel_kwargs:
                kwargs.pop("vl_rec_model_dir", None)
            kwargs.update(accel_kwargs)
            self.ocr_kwargs = kwargs
            self.ocr = self.create_ocr_engine(kwargs)
            print("PaddleOCR-VL 1.6 初始化成功")
            if hasattr(self, "base_dir"):
                print(f"当前工作目录: {self.base_dir}")
        except Exception as error:
            messagebox.showerror("错误", f"OCR初始化失败: {str(error)}")
            self.ocr = None

    def create_ocr_engine(self, kwargs):
        try:
            return PaddleOCRVL(**kwargs)
        except Exception:
            if kwargs.get("device") != "gpu":
                raise
            fallback_kwargs = kwargs.copy()
            fallback_kwargs["device"] = "cpu"
            print("GPU 初始化失败,自动回退 CPU")
            engine = PaddleOCRVL(**fallback_kwargs)
            self.ocr_kwargs = fallback_kwargs
            return engine

    def create_main_window(self):
        self.root = tk.Tk()
        self.root.title("PaddleOCR-VL 1.6 屏幕文字识别工具")
        self.root.geometry("760x460")
        self.root.minsize(640, 360)
        self.root.resizable(True, True)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        action_frame = tk.Frame(toolbar_frame)
        action_frame.pack(fill=tk.X, pady=(0, 4))

        option_frame = tk.Frame(toolbar_frame)
        option_frame.pack(fill=tk.X)

        self.select_button = tk.Button(
            action_frame,
            text="选择屏幕区域",
            font=("Arial", 12),
            width=12,
            padx=8,
            pady=4,
            command=self.start_screen_selection,
        )
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_button = tk.Button(
            action_frame,
            text="清除文本",
            font=("Arial", 12),
            width=10,
            padx=8,
            pady=4,
            command=self.clear_text,
        )
        self.clear_button.pack(side=tk.LEFT)

        self.copy_button = tk.Button(
            action_frame,
            text="复制",
            font=("Arial", 12),
            width=8,
            padx=8,
            pady=4,
            command=self.copy_text_to_clipboard,
        )
        self.copy_button.pack(side=tk.LEFT, padx=(10, 0))

        self.fast_mode_var = tk.BooleanVar(value=True)
        self.fast_mode_check = tk.Checkbutton(
            option_frame,
            text="快速模式",
            variable=self.fast_mode_var,
        )
        self.fast_mode_check.grid(row=0, column=0, sticky="w", padx=(0, 16))

        self.preserve_layout_var = tk.BooleanVar(value=True)
        self.preserve_layout_check = tk.Checkbutton(
            option_frame,
            text="保留原文分段",
            variable=self.preserve_layout_var,
        )
        self.preserve_layout_check.grid(row=0, column=1, sticky="w", padx=(0, 16))

        self.auto_copy_var = tk.BooleanVar(value=False)
        self.auto_copy_check = tk.Checkbutton(
            option_frame,
            text="自动复制",
            variable=self.auto_copy_var,
        )
        self.auto_copy_check.grid(row=0, column=2, sticky="w", padx=(0, 16))

        self.always_on_top_var = tk.BooleanVar(value=False)
        self.always_on_top_check = tk.Checkbutton(
            option_frame,
            text="窗口置顶",
            variable=self.always_on_top_var,
            command=self.apply_always_on_top,
        )
        self.always_on_top_check.grid(row=0, column=3, sticky="w", padx=(0, 16))

        text_frame = tk.Frame(main_frame)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        # Reflow text to the current window width while preserving paragraph breaks.
        self.text_area = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            state=tk.NORMAL,
        )
        self.text_area.grid(row=0, column=0, sticky="nsew")

        vertical_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")

        self.text_area.configure(
            yscrollcommand=vertical_scrollbar.set,
        )

        self.status_label = tk.Label(
            main_frame,
            text="就绪",
            font=("Arial", 9),
            fg="green",
            anchor="w",
        )
        self.status_label.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def start_screen_selection(self):
        if not self.ocr:
            messagebox.showerror("错误", "OCR引擎未初始化")
            return

        self.root.withdraw()
        self.root.update()
        selector = ScreenSelector(self.on_area_selected, cancel_callback=self.root.deiconify)
        selector.start_selection()

    def on_area_selected(self, left, top, right, bottom):
        self.root.deiconify()
        self.status_label.config(text="正在截图和识别...", fg="orange")
        self.root.update()

        thread = threading.Thread(target=self.perform_ocr, args=(left, top, right, bottom))
        thread.daemon = True
        thread.start()

    def perform_ocr(self, left, top, right, bottom):
        temp_image_path = None
        try:
            screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
            print(f"截图尺寸: {screenshot.size}")
            predict_kwargs = {}
            low_memory_mode = use_gpu_low_memory_mode()
            max_edge = MAX_LOW_MEMORY_IMAGE_EDGE if low_memory_mode else MAX_FAST_IMAGE_EDGE
            if self.use_preserve_layout():
                screenshot = resize_for_fast_ocr(screenshot, max_edge=max_edge)
                print(f"保留原文分段截图尺寸: {screenshot.size}")
                if low_memory_mode:
                    predict_kwargs = LOW_MEMORY_PRESERVE_LAYOUT_PREDICT_KWARGS.copy()
                else:
                    predict_kwargs = PRESERVE_LAYOUT_PREDICT_KWARGS.copy()
            elif self.use_fast_mode():
                screenshot = resize_for_fast_ocr(screenshot, max_edge=max_edge)
                print(f"快速模式截图尺寸: {screenshot.size}")
                if low_memory_mode:
                    predict_kwargs = LOW_MEMORY_FAST_OCR_PREDICT_KWARGS.copy()
                else:
                    predict_kwargs = FAST_OCR_PREDICT_KWARGS.copy()

            self.temp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.temp_dir,
                prefix="ocr_",
                suffix=".png",
                delete=False,
            ) as temp_image:
                temp_image_path = Path(temp_image.name)
            screenshot.save(str(temp_image_path))

            results = self.predict_with_gpu_oom_fallback(str(temp_image_path), predict_kwargs)
            count = len(results) if hasattr(results, "__len__") else "未知"
            print(f"OCR推理完成, 结果条目数: {count}")
            self.ui_queue.put(self.process_ocr_results(results))
        except Exception as error:
            error_msg = f"OCR识别失败: {str(error)}"
            print(f"OCR识别完整错误: {error}")
            self.ui_queue.put(("error", error_msg))
        finally:
            if temp_image_path and temp_image_path.exists():
                temp_image_path.unlink()

    def predict_with_gpu_oom_fallback(self, image_path, predict_kwargs):
        try:
            return self.ocr.predict(image_path, **predict_kwargs)
        except Exception as error:
            if not self.can_retry_cpu_after_gpu_oom(error):
                raise
            print("GPU 显存不足,自动切换 CPU 重试本次识别")
            cpu_kwargs = self.ocr_kwargs.copy()
            cpu_kwargs["device"] = "cpu"
            self.ocr = self.create_ocr_engine(cpu_kwargs)
            self.ocr_kwargs = cpu_kwargs
            os.environ["PPOCR_DEVICE"] = "cpu"
            return self.ocr.predict(image_path, **predict_kwargs)

    def can_retry_cpu_after_gpu_oom(self, error):
        kwargs = getattr(self, "ocr_kwargs", {})
        return kwargs.get("device") == "gpu" and is_cuda_out_of_memory_error(error)

    def process_ocr_results(self, results):
        try:
            text_lines = self.extract_layout_block_lines(results) or self.extract_text_lines(results)
            if text_lines:
                print(f"OCR识别文本行数: {len(text_lines)}")
                return ("text", "\n".join(text_lines))

            print("OCR识别文本行数: 0")
            return ("status", "未识别到有效文字", "orange")
        except Exception as error:
            error_msg = f"结果处理失败: {str(error)}"
            print(f"完整错误信息: {error}")
            return ("error", error_msg)

    def extract_layout_block_lines(self, value):
        if value is None:
            return []

        if isinstance(value, (list, tuple)):
            lines = []
            for item in value:
                item_lines = self.extract_layout_block_lines(item)
                if item_lines:
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.extend(item_lines)
            return lines

        if isinstance(value, dict):
            blocks = value.get("parsing_res_list")
            if isinstance(blocks, list):
                return self.extract_lines_from_layout_blocks(blocks)
            return []

        blocks = getattr(value, "parsing_res_list", None)
        if isinstance(blocks, list):
            return self.extract_lines_from_layout_blocks(blocks)

        return []

    def extract_lines_from_layout_blocks(self, blocks):
        lines = []
        for block in blocks:
            content = None
            if isinstance(block, dict):
                content = block.get("block_content") or block.get("content")
            else:
                content = getattr(block, "content", None)
            block_lines = normalize_text_lines(content)
            if not block_lines:
                continue
            if lines and lines[-1] != "":
                lines.append("")
            lines.extend(block_lines)

        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def extract_text_lines(self, value):
        lines = []

        if value is None:
            return lines

        if isinstance(value, str):
            return normalize_text_lines(value)

        if isinstance(value, (list, tuple)):
            for item in value:
                lines.extend(self.extract_text_lines(item))
            return lines

        if isinstance(value, dict):
            for key in (
                "markdown",
                "text",
                "content",
                "rec_text",
                "rec_texts",
                "parsing_res_list",
                "layout_parsing_result",
            ):
                if key in value:
                    extracted = self.extract_text_lines(value[key])
                    if extracted:
                        return extracted
            return []

        for attr in ("markdown", "text", "content", "rec_text", "rec_texts", "json", "res"):
            if hasattr(value, attr):
                attr_value = getattr(value, attr)
                if callable(attr_value):
                    continue
                lines.extend(self.extract_text_lines(attr_value))

        return lines

    def process_ui_queue(self):
        while True:
            try:
                message = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            message_type = message[0]
            if message_type == "text":
                self.display_results(message[1])
            elif message_type == "status":
                self.update_status(message[1], message[2])
            elif message_type == "error":
                self.show_error(message[1])

        self.root.after(100, self.process_ui_queue)

    def use_fast_mode(self):
        fast_mode_var = getattr(self, "fast_mode_var", None)
        if fast_mode_var is None:
            return True
        return bool(fast_mode_var.get())

    def use_preserve_layout(self):
        preserve_layout_var = getattr(self, "preserve_layout_var", None)
        if preserve_layout_var is None:
            return False
        return bool(preserve_layout_var.get())

    def use_auto_copy(self):
        auto_copy_var = getattr(self, "auto_copy_var", None)
        if auto_copy_var is None:
            return False
        return bool(auto_copy_var.get())

    def apply_always_on_top(self):
        always_on_top_var = getattr(self, "always_on_top_var", None)
        enabled = bool(always_on_top_var.get()) if always_on_top_var is not None else False
        self.root.attributes("-topmost", enabled)

    def display_results(self, text):
        self.last_result_text = text
        self.text_area.config(wrap=tk.WORD)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, text)
        line_count = len([line for line in text.split("\n") if line.strip()])
        self.update_status(f"识别完成,共识别到 {line_count} 行文字", "green")
        if self.use_auto_copy():
            self.copy_text_to_clipboard(auto=True)

    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.last_result_text = ""
        self.update_status("文本已清除", "blue")

    def copy_text_to_clipboard(self, auto=False):
        text = getattr(self, "last_result_text", "")
        if not text:
            self.update_status("没有可复制的文本", "orange")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        if auto:
            self.update_status("已自动复制到剪贴板", "blue")
        else:
            self.update_status("已复制到剪贴板", "blue")

    def update_status(self, message, color="black"):
        self.status_label.config(text=message, fg=color)

    def show_error(self, message):
        messagebox.showerror("错误", message)
        self.update_status("发生错误", "red")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        require_runtime_dependencies()
        print("OCR 运行依赖已安装")
    except RuntimeError as error:
        print(error)
        raise SystemExit(1)

    app = OCRApp()
    app.run()
