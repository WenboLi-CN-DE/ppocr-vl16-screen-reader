import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import messagebox

from .platform import (
    DPIHandler,
    configure_windows_nvidia_dll_paths,
    is_url_port_open,
)
from .predictor import OcrPredictor
from .profiles import (
    FAST_OCR_PREDICT_KWARGS,
    LOW_MEMORY_FAST_OCR_PREDICT_KWARGS,
    LOW_MEMORY_PRESERVE_LAYOUT_PREDICT_KWARGS,
    MAX_FAST_IMAGE_EDGE,
    MAX_LOW_MEMORY_IMAGE_EDGE,
    PRESERVE_LAYOUT_PREDICT_KWARGS,
    resize_for_ocr,
    select_max_edge,
    select_predict_kwargs,
)
from .runtime import (
    acceleration_kwargs as runtime_acceleration_kwargs,
    is_cuda_out_of_memory_error,
    local_model_kwargs,
    require_runtime_dependencies as require_runtime_dependencies_for_error,
    use_gpu_low_memory_mode,
)
from .text import (
    extract_layout_block_lines,
    extract_lines_from_layout_blocks,
    extract_text_lines,
)


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_minloglevel", "2")
warnings.filterwarnings("ignore", message=r"No ccache found.*")
warnings.filterwarnings("ignore", message=r"To copy construct from a tensor.*")

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


if Image is not None:
    RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", None), "LANCZOS", None)
    if RESAMPLE_LANCZOS is None:
        RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", None)
else:
    RESAMPLE_LANCZOS = None


def require_runtime_dependencies():
    require_runtime_dependencies_for_error(RUNTIME_IMPORT_ERROR)


def acceleration_kwargs(base_dir):
    return runtime_acceleration_kwargs(base_dir, port_open=is_url_port_open)


def resize_for_fast_ocr(image, max_edge=MAX_FAST_IMAGE_EDGE):
    return resize_for_ocr(image, max_edge, RESAMPLE_LANCZOS)


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
            self.ocr_predictor = self.create_ocr_predictor()
            print("PaddleOCR-VL 1.6 初始化成功")
            if hasattr(self, "base_dir"):
                print(f"当前工作目录: {self.base_dir}")
        except Exception as error:
            messagebox.showerror("错误", f"OCR初始化失败: {str(error)}")
            self.ocr = None

    def create_ocr_engine(self, kwargs, persist=True):
        try:
            return PaddleOCRVL(**kwargs)
        except Exception:
            if kwargs.get("device") != "gpu":
                raise
            fallback_kwargs = kwargs.copy()
            fallback_kwargs["device"] = "cpu"
            print("GPU 初始化失败,自动回退 CPU")
            engine = PaddleOCRVL(**fallback_kwargs)
            if persist:
                self.ocr_kwargs = fallback_kwargs
            return engine

    def create_ocr_predictor(self):
        return OcrPredictor(
            get_engine=lambda: self.ocr,
            get_engine_kwargs=lambda: self.ocr_kwargs,
            create_engine=self.create_transient_ocr_engine,
            set_engine=self.set_ocr_engine,
        )

    def create_transient_ocr_engine(self, kwargs):
        try:
            return self.create_ocr_engine(kwargs, persist=False)
        except TypeError as error:
            if "persist" not in str(error):
                raise
            return self.create_ocr_engine(kwargs)

    def set_ocr_engine(self, engine, kwargs):
        self.ocr = engine
        self.ocr_kwargs = kwargs

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

        self.preserve_layout_var = tk.BooleanVar(value=False)
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
            low_memory_mode = use_gpu_low_memory_mode()
            fast_mode = self.use_fast_mode()
            preserve_layout = self.use_preserve_layout()
            max_edge = select_max_edge(
                fast_mode=fast_mode,
                preserve_layout=preserve_layout,
                low_memory_mode=low_memory_mode,
            )
            if max_edge:
                screenshot = resize_for_fast_ocr(screenshot, max_edge=max_edge)
            predict_kwargs = select_predict_kwargs(
                preserve_layout=preserve_layout,
                fast_mode=fast_mode,
                low_memory_mode=low_memory_mode,
            )
            if preserve_layout:
                print(f"保留原文分段截图尺寸: {screenshot.size}")
            elif fast_mode:
                print(f"快速模式截图尺寸: {screenshot.size}")

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
        predictor = getattr(self, "ocr_predictor", None)
        if predictor is None:
            predictor = self.create_ocr_predictor()
            self.ocr_predictor = predictor
        return predictor.predict(image_path, predict_kwargs)

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
        return extract_layout_block_lines(value)

    def extract_lines_from_layout_blocks(self, blocks):
        return extract_lines_from_layout_blocks(blocks)

    def extract_text_lines(self, value):
        return extract_text_lines(value)

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
