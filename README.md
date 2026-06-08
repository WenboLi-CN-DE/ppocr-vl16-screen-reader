# PaddleOCR-VL 1.6 Screen Reader

一个基于 PaddleOCR-VL 1.6 的桌面截图文字识别工具。点击“选择屏幕区域”，拖拽框选屏幕内容后，程序会使用 PaddleOCR-VL 1.6 识别内容，并在窗口中显示纯文本结果。

## 功能

- 使用 PaddleOCR-VL 1.6 作为 OCR 引擎。
- 保持和旧版工具一致的桌面区域截图工作流。
- 支持 Windows、macOS 和 Linux 桌面环境。
- 默认使用快速 OCR，输出纯文本，适合截图翻译、复制和快速阅读。

PaddleOCR-VL 1.6 是文档解析模型，官方说明支持文本、表格、公式、图表、印章等能力。本项目第一版只保留纯文本显示，后续可以扩展 Markdown/JSON 导出。

## 环境要求

- [uv](https://docs.astral.sh/uv/)
- Python 3.11
- 64 位操作系统
- Linux 需要带桌面环境并安装 Tkinter。
- macOS 首次截图时需要在系统设置中授予终端或 Python 屏幕录制权限。

PaddleOCR-VL 1.6 依赖比传统 PP-OCRv5 更重。官方建议安装 `paddleocr[doc-parser]>=3.6.0` 和 PaddlePaddle 3.2.1 或更高版本。

## 一键启动

新环境部署不需要分多步执行。安装 [uv](https://docs.astral.sh/uv/) 后，直接运行启动脚本即可自动完成 Python 3.11、依赖、平台加速包和模型下载。

macOS / Linux：

```bash
./start.command
```

Windows：

```text
start.bat
```

启动脚本默认使用国内源：清华 PyPI、PaddlePaddle 官方国内 wheel 源、ModelScope 模型源。模型会下载到项目内 `models/official_models/`，后续启动会复用本地模型。

如需使用国际源，可传入 `--global`，或创建 `.ppocr_use_global` 标记文件后再启动。

```bash
./start.command --global
```

国内或离线新机器部署见 [DEPLOY_CN.md](DEPLOY_CN.md)。

Linux 如果缺少 Tkinter，请先使用系统包管理器安装。例如 Ubuntu：

```bash
sudo apt-get install python3-tk
```

## 手动开发命令

```bash
uv python install 3.11
uv sync
uv run python scripts/models/download_models.py
uv run python launcher.py
```

`launcher.py` 会自动检测系统和可用加速通道，然后启动 `ocr_app` 包内的桌面 OCR 应用：

- macOS Apple Silicon：如果已安装 `mlx_vlm.server`，自动启动/连接 MLX-VLM 服务，并让 PaddleOCR-VL 使用 `mlx-vlm-server` 后端。
- Windows：如果检测到 `nvidia-smi`，自动设置 `PPOCR_DEVICE=gpu`。如果 GPU 初始化失败，应用会自动回退 CPU。
- 其他环境：使用 PaddleOCR 默认本地推理。

操作步骤：

1. 点击“选择屏幕区域”。
2. 拖拽鼠标框选需要识别的文字。
3. 等待识别结果显示在主窗口中。
4. 框选时按 `Esc` 可取消操作。

窗口工具栏：

- “复制”：把当前识别结果复制到剪贴板。
- “自动复制”：每次识别完成后自动复制结果。
- “窗口置顶”：开关主窗口置顶，方便反复截图和复制。
- “快速模式”：默认开启，跳过布局检测，加快普通截图 OCR。
- “保留原文分段”：按 PaddleOCR-VL 返回的版面 block 重建段落，需要原文段落结构时手动开启。

## 速度优化

默认勾选“快速模式”并关闭“保留原文分段”，普通截图会跳过布局检测，优先保证速度和显存占用。

快速模式会：

- 跳过文档布局检测，直接执行 OCR。
- 将大截图最长边压到 1536 像素以内。
- 限制生成长度，减少视觉语言模型输出等待时间。

纯快速模式适合菜单、按钮、短句、代码片段等不关心段落结构的截图。长文章、新闻正文、复杂版面、表格、公式或多块文档建议手动勾选“保留原文分段”；该模式会开启布局检测，阅读效果更稳定，但比纯快速 OCR 慢，也更占显存。

识别会在后台线程执行，主窗口不会被推理阻塞。OCR 推理任务本身保持串行执行；不要并发启动多个 PaddleOCR-VL 推理任务，否则通常不会更快，还会增加 GPU 显存峰值和 OOM 概率。

这套优化对 Windows 和 macOS 都生效，不依赖平台专属后端。

## 平台专属加速

### macOS Apple Silicon

直接运行 `start.command` 会自动安装 MLX-VLM 加速依赖并写入 `.ppocr_no_sync`。如果只想单独安装加速包，也可以运行 `./scripts/install/install_macos_accel.command`。

之后使用 `start.command` 或：

```bash
uv run python launcher.py
```

启动器会自动启动 `mlx_vlm.server --port 8111`，日志写入 `temp/mlx_vlm_server.log`。

### Windows NVIDIA GPU

先确认机器安装了 NVIDIA 驱动，并且 `nvidia-smi` 可用。直接运行 `start.bat` 会自动安装 GPU 版 `paddlepaddle-gpu` 并写入 `.ppocr_no_sync`。如果 `nvidia-smi` 显示 `CUDA Version: 13.x`，脚本会安装 Paddle 官方 CUDA 13.0 包；否则使用 CUDA 12.6 包。

之后使用 `start.bat`。启动器会自动设置 GPU 设备；如果 GPU 版 PaddlePaddle 或驱动不可用，会自动回退 CPU。

## 测试

```bash
uv run pytest
```

测试使用 stub 隔离 PaddleOCR-VL 和桌面 GUI，不需要实际加载模型，也不会弹出窗口。

## 常见问题

### 启动时提示运行依赖缺失

确认当前目录是项目根目录，然后重新运行一键启动脚本：

```bash
./start.command
```

Windows 直接重新运行 `start.bat`。如果存在 `.ppocr_no_sync`，新版 bootstrap 会一次性检查 `PIL`、`paddleocr`、`paddle` 并补齐缺失运行依赖，不会执行完整 `uv sync` 覆盖 Windows GPU 加速包。

### uv 下载 numpy 超时

如果 Windows 上看到类似：

```text
Failed to download `numpy`
Failed to download distribution due to network timeout
Try increasing UV_HTTP_TIMEOUT
```

请使用新版 `start.bat` 或 `start.command`，它们已经自动设置 `UV_HTTP_TIMEOUT=300` 并默认使用国内源。也可以在命令行临时执行：

```bat
set UV_HTTP_TIMEOUT=300
uv sync
```

### 首次 OCR 很慢

PaddleOCR-VL 1.6 是视觉语言模型，首次运行会初始化较大的模型。新版启动脚本会自动执行 `scripts/models/download_models.py`，把模型下载到项目内。

### Windows 或 macOS 仍然很慢

首次识别会加载模型，通常比后续识别慢。框选区域越大，识别越慢；尽量只框选需要识别的文字区域。如果你更重视速度而不是段落结构，保持默认“快速模式”即可，不要勾选“保留原文分段”。

macOS Apple Silicon 可安装 MLX-VLM 加速；Windows NVIDIA 机器可安装 GPU 版 PaddlePaddle。安装方式见“平台专属加速”。

如果 Windows 日志出现 `Mismatched GPU Architecture`，例如当前 GPU 是 `120`，说明已安装的 Paddle GPU wheel 不支持你的显卡架构。NVIDIA 50 系显卡通常需要 CUDA 13.0 对应的 Paddle GPU 包，请重新运行新版 `start.bat`，或手动运行 `scripts\install\install_windows_gpu.bat`。

如果 Windows 日志出现 `Could not locate cublasLt64_13.dll`，说明 CUDA 13 运行库 DLL 没有被 Paddle 找到。新版程序启动时会递归扫描 `.venv\Lib\site-packages\nvidia\` 下所有 DLL 目录并加入搜索路径，同时预加载 `cublasLt64_13.dll`；新版 `start.bat` 也会在 CUDA 13 GPU 环境下检查这个 DLL 是否存在，缺失时自动补装 `nvidia-cublas`。

如果 Windows 日志出现 `CUDA error(2), out of memory` 或 `cudaErrorMemoryAllocation`，说明 GPU 显存不足。新版启动器在 Windows GPU 环境会默认设置 `PPOCR_GPU_MEMORY_MODE=low`，降低截图输入尺寸和 VLM 生成预算；如果预测阶段仍然 OOM，程序会先用更保守的 GPU 参数重试，本次仍失败时才临时创建 CPU OCR 引擎兜底。CPU 兜底不会永久替换主 GPU 引擎，下一次识别仍会优先尝试 GPU。显存充足时可以在启动前设置 `PPOCR_GPU_MEMORY_MODE=normal` 关闭低显存模式。

### macOS 无法截图

在“系统设置 -> 隐私与安全性 -> 屏幕录制”中允许终端或 Python 截取屏幕，然后重新启动程序。

### Linux 无法打开窗口

确认当前系统带有桌面环境，并已安装 Tkinter。远程服务器通常需要配置图形会话后才能使用本工具。

## 参考

- [PaddleOCR-VL 1.6 Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [PaddleOCR-VL 使用文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR-VL Apple Silicon 使用文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
