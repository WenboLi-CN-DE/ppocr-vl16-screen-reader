from pathlib import Path
import sys

import pytest

import ocr


class FakeWindow:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class FakeRoot:
    def __init__(self):
        self.deiconify_calls = 0
        self.after_calls = []
        self.attributes_calls = []
        self.clipboard = ""
        self.geometry_calls = []
        self.min_size_calls = []

    def deiconify(self):
        self.deiconify_calls += 1

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))

    def attributes(self, *args):
        self.attributes_calls.append(args)

    def geometry(self, value):
        self.geometry_calls.append(value)

    def minsize(self, width, height):
        self.min_size_calls.append((width, height))

    def title(self, value):
        pass

    def resizable(self, width, height):
        pass

    def clipboard_clear(self):
        self.clipboard = ""

    def clipboard_append(self, text):
        self.clipboard += text

    def update_idletasks(self):
        pass

    def update(self):
        pass


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeTextArea:
    def __init__(self):
        self.operations = []

    def delete(self, *args):
        self.operations.append(("delete", args))

    def insert(self, *args):
        self.operations.append(("insert", args))

    def config(self, **kwargs):
        self.operations.append(("config", kwargs))


class FakeWidget:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pack_calls = []
        self.grid_calls = []
        self.config_calls = []
        self.rowconfigure_calls = []
        self.columnconfigure_calls = []

    def pack(self, **kwargs):
        self.pack_calls.append(kwargs)

    def grid(self, **kwargs):
        self.grid_calls.append(kwargs)

    def configure(self, **kwargs):
        self.config_calls.append(kwargs)

    def config(self, **kwargs):
        self.config_calls.append(kwargs)

    def rowconfigure(self, index, **kwargs):
        self.rowconfigure_calls.append((index, kwargs))

    def columnconfigure(self, index, **kwargs):
        self.columnconfigure_calls.append((index, kwargs))

    def yview(self, *args):
        pass

    def set(self, *args):
        pass

    def delete(self, *args):
        pass

    def insert(self, *args):
        pass


class FakeScreenshot:
    size = (20, 20)

    def save(self, path):
        Path(path).write_bytes(b"image")


class ResizableScreenshot:
    def __init__(self, size):
        self.size = size
        self.resize_calls = []

    def resize(self, size, resample=None):
        self.resize_calls.append((size, resample))
        self.size = size
        return self

    def save(self, path):
        Path(path).write_bytes(b"image")


def test_configure_windows_nvidia_dll_paths_adds_nvidia_dll_dirs(tmp_path):
    cublas_bin = tmp_path / "nvidia" / "cublas" / "bin"
    cudnn_lib = tmp_path / "nvidia" / "cudnn" / "lib"
    cublas_bin.mkdir(parents=True)
    cudnn_lib.mkdir(parents=True)
    (cublas_bin / "cublasLt64_13.dll").write_bytes(b"dll")
    (cudnn_lib / "cudnn64_9.dll").write_bytes(b"dll")
    added = []
    loaded = []
    environ = {"PATH": "C:\\Windows\\System32"}

    configured = ocr.configure_windows_nvidia_dll_paths(
        site_package_roots=[tmp_path],
        add_dll_directory=lambda path: added.append(path),
        load_library=lambda path: loaded.append(path),
        environ=environ,
        os_name="nt",
    )

    assert configured == [str(cublas_bin), str(cudnn_lib)]
    assert added == [str(cublas_bin), str(cudnn_lib)]
    assert loaded == [str(cublas_bin / "cublasLt64_13.dll")]
    assert environ["PATH"].startswith(f"{cublas_bin};{cudnn_lib};")


def test_setup_ocr_uses_paddleocr_vl_16(monkeypatch):
    created = []
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(ocr, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setattr(ocr.messagebox, "showerror", lambda *args: None)

    app.setup_ocr()

    assert app.ocr is not None
    assert created == [{"pipeline_version": "v1.6"}]


def test_setup_ocr_uses_project_local_models_when_available(tmp_path, monkeypatch):
    created = []
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.base_dir = tmp_path
    layout_dir = tmp_path / "models" / "official_models" / "PP-DocLayoutV3"
    vl_dir = tmp_path / "models" / "official_models" / "PaddleOCR-VL-1.6"
    layout_dir.mkdir(parents=True)
    vl_dir.mkdir(parents=True)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(ocr, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setattr(ocr.messagebox, "showerror", lambda *args: None)

    app.setup_ocr()

    assert created == [
        {
            "pipeline_version": "v1.6",
            "layout_detection_model_dir": str(layout_dir),
            "vl_rec_model_dir": str(vl_dir),
        }
    ]


def test_setup_ocr_uses_mlx_server_and_skips_local_vl_model_on_apple_silicon(tmp_path, monkeypatch):
    created = []
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.base_dir = tmp_path
    layout_dir = tmp_path / "models" / "official_models" / "PP-DocLayoutV3"
    vl_dir = tmp_path / "models" / "official_models" / "PaddleOCR-VL-1.6"
    layout_dir.mkdir(parents=True)
    vl_dir.mkdir(parents=True)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(ocr, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setattr(ocr.messagebox, "showerror", lambda *args: None)
    monkeypatch.setenv("PPOCR_ACCEL", "mlx")
    monkeypatch.setenv("PPOCR_MLX_SERVER_URL", "http://127.0.0.1:8111/")
    monkeypatch.setattr(ocr, "is_url_port_open", lambda url: True)

    app.setup_ocr()

    assert created == [
        {
            "pipeline_version": "v1.6",
            "layout_detection_model_dir": str(layout_dir),
            "vl_rec_backend": "mlx-vlm-server",
            "vl_rec_server_url": "http://127.0.0.1:8111/",
            "vl_rec_api_model_name": str(vl_dir),
        }
    ]


def test_setup_ocr_passes_windows_gpu_device_from_environment(monkeypatch):
    created = []
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(ocr, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setattr(ocr.messagebox, "showerror", lambda *args: None)
    monkeypatch.setenv("PPOCR_DEVICE", "gpu")

    app.setup_ocr()

    assert created == [{"pipeline_version": "v1.6", "device": "gpu"}]


def test_setup_ocr_retries_cpu_when_gpu_initialization_fails(monkeypatch):
    created = []
    errors = []
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    class FakePaddleOCRVL:
        def __init__(self, **kwargs):
            created.append(kwargs)
            if kwargs.get("device") == "gpu":
                raise RuntimeError("gpu unavailable")

    monkeypatch.setattr(ocr, "PaddleOCRVL", FakePaddleOCRVL)
    monkeypatch.setattr(ocr.messagebox, "showerror", lambda title, message: errors.append((title, message)))
    monkeypatch.setenv("PPOCR_DEVICE", "gpu")

    app.setup_ocr()

    assert app.ocr is not None
    assert created == [
        {"pipeline_version": "v1.6", "device": "gpu"},
        {"pipeline_version": "v1.6", "device": "cpu"},
    ]
    assert errors == []


def test_main_window_uses_two_row_toolbar_and_visible_status(monkeypatch):
    root = FakeRoot()
    frames = []
    labels = []
    checkbuttons = []
    buttons = []

    def fake_frame(*args, **kwargs):
        widget = FakeWidget(*args, **kwargs)
        frames.append(widget)
        return widget

    def fake_label(*args, **kwargs):
        widget = FakeWidget(*args, **kwargs)
        labels.append(widget)
        return widget

    def fake_checkbutton(*args, **kwargs):
        widget = FakeWidget(*args, **kwargs)
        checkbuttons.append(widget)
        return widget

    monkeypatch.setattr(ocr.tk, "Tk", lambda: root)
    monkeypatch.setattr(ocr.tk, "Frame", fake_frame)
    monkeypatch.setattr(ocr.tk, "Button", lambda *args, **kwargs: buttons.append(FakeWidget(*args, **kwargs)) or buttons[-1])
    monkeypatch.setattr(ocr.tk, "Checkbutton", fake_checkbutton)
    monkeypatch.setattr(ocr.tk, "Text", lambda *args, **kwargs: FakeWidget(*args, **kwargs))
    monkeypatch.setattr(ocr.tk, "Scrollbar", lambda *args, **kwargs: FakeWidget(*args, **kwargs))
    monkeypatch.setattr(ocr.tk, "Label", fake_label)
    monkeypatch.setattr(ocr.tk, "BooleanVar", lambda value=False: FakeVar(value))

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.create_main_window()

    assert root.geometry_calls == ["760x460"]
    assert root.min_size_calls == [(640, 360)]
    assert len(checkbuttons) == 4
    assert all(widget.grid_calls for widget in checkbuttons)
    main_frame = frames[0]
    assert main_frame.rowconfigure_calls == [(1, {"weight": 1})]
    assert main_frame.columnconfigure_calls == [(0, {"weight": 1})]
    assert labels[-1].grid_calls == [{"row": 2, "column": 0, "sticky": "ew", "pady": (6, 0)}]
    assert labels[-1].kwargs["anchor"] == "w"
    for button in buttons[:3]:
        assert "bg" not in button.kwargs
        assert "fg" not in button.kwargs
        assert button.kwargs["width"] >= 8


def test_process_ocr_results_extracts_vl_markdown_content():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {"markdown": {"text": " 第一行\n\nsecond line "}, "rec_text": "ignored fallback"},
    ])

    assert result == ("text", "第一行\n\nsecond line")


def test_process_ocr_results_reflows_lines_inside_paragraphs():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {
            "markdown": (
                "튀르키예 주재 이란 대사관은 미국이 비자 발급을 거부했다며,\n"
                "이를 정치적 편향에 따른 개입이라고 비판했다.\n"
                "\n"
                "이란 국영 연계 매체들은 행정 관계자 15명이 미국 입국을\n"
                "거부당한 이들에 포함됐다고 보도했다."
            )
        }
    ])

    assert result == (
        "text",
        "튀르키예 주재 이란 대사관은 미국이 비자 발급을 거부했다며, "
        "이를 정치적 편향에 따른 개입이라고 비판했다.\n\n"
        "이란 국영 연계 매체들은 행정 관계자 15명이 미국 입국을 "
        "거부당한 이들에 포함됐다고 보도했다.",
    )


def test_process_ocr_results_does_not_guess_paragraph_breaks_from_sentence_endings():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {
            "markdown": (
                "튀르키예 주재 이란 대사관은 미국이 비자 발급을 거부했다며,\n"
                "이를 정치적 편향에 따른 개입이라고 비판했다.\n"
                "이란 국영 연계 매체들은 행정 관계자 15명이 미국 입국을\n"
                "거부당한 이들에 포함됐다고 보도했다.\n"
                "이란 대표팀은 멕시코로 향했다."
            )
        }
    ])

    assert result == (
        "text",
        "튀르키예 주재 이란 대사관은 미국이 비자 발급을 거부했다며, "
        "이를 정치적 편향에 따른 개입이라고 비판했다. "
        "이란 국영 연계 매체들은 행정 관계자 15명이 미국 입국을 "
        "거부당한 이들에 포함됐다고 보도했다. "
        "이란 대표팀은 멕시코로 향했다.",
    )


def test_process_ocr_results_uses_layout_blocks_for_original_paragraphs():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {
            "parsing_res_list": [
                {
                    "block_label": "text",
                    "block_content": (
                        "튀르키예 주재 이란 대사관은 미국이 비자 발급을 거부했다며,\n"
                        "이를 정치적 편향에 따른 개입이라고 비판했다."
                    ),
                    "block_bbox": [52, 42, 1172, 120],
                },
                {
                    "block_label": "text",
                    "block_content": (
                        "이란 국영 연계 매체들은 행정 관계자 15명이 미국 입국을\n"
                        "거부당한 이들에 포함됐다고 보도했다."
                    ),
                    "block_bbox": [52, 200, 1160, 280],
                },
            ],
            "markdown": "fallback should not be used",
        }
    ])

    assert result == (
        "text",
        "튀르키예 주재 이란 대사관은 미국이 비자 발급을 거부했다며, "
        "이를 정치적 편향에 따른 개입이라고 비판했다.\n\n"
        "이란 국영 연계 매체들은 행정 관계자 15명이 미국 입국을 "
        "거부당한 이들에 포함됐다고 보도했다.",
    )


def test_process_ocr_results_converts_html_table_blocks_to_plain_text():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {
            "parsing_res_list": [
                {
                    "block_label": "table",
                    "block_content": (
                        "<table><tr><td>첫 번째 문장입니다.</td></tr>"
                        "<tr><td>두 번째 문장입니다.</td></tr></table>"
                    ),
                    "block_bbox": [10, 10, 200, 100],
                }
            ]
        }
    ])

    assert result == ("text", "첫 번째 문장입니다.\n\n두 번째 문장입니다.")


def test_process_ocr_results_unescapes_html_entities():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {"markdown": "미국이 &quot;필수 지원 인력&quot;에게 비자를 발급했다."}
    ])

    assert result == ("text", '미국이 "필수 지원 인력"에게 비자를 발급했다.')


def test_process_ocr_results_preserves_paragraph_breaks():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {
            "markdown": (
                "튀르키예 주재 이란 대사관은 비자 발급을 거부했다고 밝혔다.\n"
                "\n"
                "이란 국영 연계 매체들은 행정 관계자 15명이 포함됐다고 보도했다.\n"
                "\n\n"
                "이란 대표팀은 멕시코로 향했다."
            )
        }
    ])

    assert result == (
        "text",
        "튀르키예 주재 이란 대사관은 비자 발급을 거부했다고 밝혔다.\n\n"
        "이란 국영 연계 매체들은 행정 관계자 15명이 포함됐다고 보도했다.\n\n"
        "이란 대표팀은 멕시코로 향했다.",
    )


def test_process_ocr_results_extracts_vl_plain_text_content():
    app = ocr.OCRApp.__new__(ocr.OCRApp)

    result = app.process_ocr_results([
        {"text": " 第一行 "},
        {"content": "second line"},
    ])

    assert result == ("text", "第一行\nsecond line")


def test_perform_ocr_puts_text_result_on_ui_queue_and_removes_temp_file(tmp_path, monkeypatch):
    class SuccessfulOCR:
        def __init__(self):
            self.calls = []

        def predict(self, path, **kwargs):
            assert Path(path).exists()
            self.calls.append(kwargs)
            return [{"markdown": "第一行\nsecond line"}]

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = SuccessfulOCR()
    app.use_fast_mode = lambda: True
    app.use_preserve_layout = lambda: False
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert list(tmp_path.iterdir()) == []
    assert app.ui_queue.get_nowait() == ("text", "第一行 second line")
    assert app.ocr.calls == [ocr.FAST_OCR_PREDICT_KWARGS]
    assert ocr.FAST_OCR_PREDICT_KWARGS["max_new_tokens"] >= 1024


def test_perform_ocr_resizes_large_screenshot_in_fast_mode(tmp_path, monkeypatch):
    screenshot = ResizableScreenshot((3000, 1500))

    class SuccessfulOCR:
        def predict(self, path, **kwargs):
            return [{"text": "ok"}]

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = SuccessfulOCR()
    app.use_fast_mode = lambda: True
    app.use_preserve_layout = lambda: False
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: screenshot)

    app.perform_ocr(0, 0, 3000, 1500)

    assert screenshot.resize_calls == [((1536, 768), ocr.RESAMPLE_LANCZOS)]


def test_perform_ocr_keeps_layout_detection_in_full_mode(tmp_path, monkeypatch):
    calls = []

    class SuccessfulOCR:
        def predict(self, path, **kwargs):
            calls.append(kwargs)
            return [{"text": "ok"}]

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = SuccessfulOCR()
    app.use_fast_mode = lambda: False
    app.use_preserve_layout = lambda: False
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert calls == [{}]


def test_perform_ocr_uses_layout_detection_when_preserving_original_paragraphs(tmp_path, monkeypatch):
    calls = []

    class SuccessfulOCR:
        def predict(self, path, **kwargs):
            calls.append(kwargs)
            return [{"parsing_res_list": [{"block_content": "第一段"}]}]

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = SuccessfulOCR()
    app.use_fast_mode = lambda: True
    app.use_preserve_layout = lambda: True
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert calls == [ocr.PRESERVE_LAYOUT_PREDICT_KWARGS]
    assert app.ui_queue.get_nowait() == ("text", "第一段")


def test_perform_ocr_uses_low_memory_predict_kwargs_on_gpu(tmp_path, monkeypatch):
    calls = []

    class SuccessfulOCR:
        def predict(self, path, **kwargs):
            calls.append(kwargs)
            return [{"text": "ok"}]

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = SuccessfulOCR()
    app.use_fast_mode = lambda: True
    app.use_preserve_layout = lambda: True
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setenv("PPOCR_DEVICE", "gpu")
    monkeypatch.setenv("PPOCR_GPU_MEMORY_MODE", "low")
    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert calls == [ocr.LOW_MEMORY_PRESERVE_LAYOUT_PREDICT_KWARGS]


def test_perform_ocr_retries_cpu_once_when_gpu_predict_oom(tmp_path, monkeypatch):
    created = []

    class FailingGpuOCR:
        def predict(self, path, **kwargs):
            raise RuntimeError("CUDA error(2), out of memory")

    class SuccessfulCpuOCR:
        def predict(self, path, **kwargs):
            return [{"text": "cpu ok"}]

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = FailingGpuOCR()
    app.ocr_kwargs = {"pipeline_version": "v1.6", "device": "gpu"}
    app.use_fast_mode = lambda: True
    app.use_preserve_layout = lambda: True
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    def fake_create_engine(kwargs):
        created.append(kwargs)
        return SuccessfulCpuOCR()

    app.create_ocr_engine = fake_create_engine
    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert created == [{"pipeline_version": "v1.6", "device": "cpu"}]
    assert app.ui_queue.get_nowait() == ("text", "cpu ok")


def test_perform_ocr_does_not_overwrite_existing_temporary_screenshot(tmp_path, monkeypatch):
    existing_path = tmp_path / "temp_screenshot.png"
    existing_path.write_bytes(b"existing")

    class SuccessfulOCR:
        def predict(self, path):
            return []

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = SuccessfulOCR()
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert existing_path.read_bytes() == b"existing"


def test_perform_ocr_reports_error_after_failure_and_removes_temp_file(tmp_path, monkeypatch):
    class FailingOCR:
        def predict(self, path):
            raise RuntimeError("predict failed")

    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.temp_dir = tmp_path
    app.ocr = FailingOCR()
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()

    monkeypatch.setattr(ocr.ImageGrab, "grab", lambda bbox: FakeScreenshot())

    app.perform_ocr(0, 0, 20, 20)

    assert list(tmp_path.iterdir()) == []
    assert app.ui_queue.get_nowait()[0] == "error"


def test_process_ui_queue_displays_text_result():
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.root = FakeRoot()
    app.ui_queue = ocr.queue.Queue()
    displayed = []
    app.display_results = lambda text: displayed.append(text)

    app.ui_queue.put(("text", "第一行"))
    app.process_ui_queue()

    assert displayed == ["第一行"]


def test_display_results_wraps_text_to_window_width_without_losing_paragraphs():
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.text_area = FakeTextArea()
    statuses = []
    app.update_status = lambda message, color="black": statuses.append((message, color))

    app.display_results("第一段\n\n第二段")

    assert ("config", {"wrap": ocr.tk.WORD}) in app.text_area.operations
    assert ("insert", (ocr.tk.END, "第一段\n\n第二段")) in app.text_area.operations
    assert statuses == [("识别完成,共识别到 2 行文字", "green")]


def test_apply_always_on_top_updates_window_attribute():
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.root = FakeRoot()
    app.always_on_top_var = FakeVar(True)

    app.apply_always_on_top()

    assert app.root.attributes_calls == [("-topmost", True)]


def test_copy_text_to_clipboard_uses_latest_result_text():
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.root = FakeRoot()
    app.last_result_text = "第一段\n\n第二段"
    statuses = []
    app.update_status = lambda message, color="black": statuses.append((message, color))

    app.copy_text_to_clipboard()

    assert app.root.clipboard == "第一段\n\n第二段"
    assert statuses == [("已复制到剪贴板", "blue")]


def test_display_results_auto_copies_when_enabled():
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.root = FakeRoot()
    app.text_area = FakeTextArea()
    app.auto_copy_var = FakeVar(True)
    statuses = []
    app.update_status = lambda message, color="black": statuses.append((message, color))

    app.display_results("第一段")

    assert app.root.clipboard == "第一段"
    assert statuses == [
        ("识别完成,共识别到 1 行文字", "green"),
        ("已自动复制到剪贴板", "blue"),
    ]


def test_start_screen_selection_restores_main_window_when_cancelled(monkeypatch):
    app = ocr.OCRApp.__new__(ocr.OCRApp)
    app.ocr = object()
    app.root = FakeRoot()
    captured = {}

    class FakeSelector:
        def __init__(self, callback, cancel_callback=None):
            captured["cancel_callback"] = cancel_callback

        def start_selection(self):
            captured["cancel_callback"]()

    monkeypatch.setattr(ocr, "ScreenSelector", FakeSelector)
    app.root.withdraw = lambda: None

    app.start_screen_selection()

    assert app.root.deiconify_calls == 1


def test_cancel_selection_notifies_caller():
    cancel_calls = []
    selector = ocr.ScreenSelector(callback=lambda *args: None, cancel_callback=lambda: cancel_calls.append(True))
    selector.selection_window = FakeWindow()

    selector.cancel_selection()

    assert selector.selection_window.destroyed is True
    assert cancel_calls == [True]


def test_dpi_handler_skips_windows_api_on_non_windows(monkeypatch):
    class UnexpectedWindll:
        access_count = 0

        def __getattr__(self, name):
            self.access_count += 1
            raise AttributeError(name)

    windll = UnexpectedWindll()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ocr.ctypes, "windll", windll, raising=False)

    ocr.DPIHandler()

    assert windll.access_count == 0


def test_require_runtime_dependencies_reports_uv_sync_command(monkeypatch):
    monkeypatch.setattr(ocr, "RUNTIME_IMPORT_ERROR", ImportError("No module named 'paddleocr'"))

    with pytest.raises(RuntimeError, match=r"uv sync"):
        ocr.require_runtime_dependencies()
