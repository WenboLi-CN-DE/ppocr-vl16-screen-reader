# PaddleOCR-VL 1.6 Screen Reader

一个基于 PaddleOCR-VL 1.6 的桌面截图文字识别工具。点击“选择屏幕区域”，拖拽框选屏幕内容后，程序会使用 PaddleOCR-VL 1.6 识别内容，并在窗口中显示纯文本结果。

## 功能

- 使用 PaddleOCR-VL 1.6 作为 OCR 引擎。
- 保持和旧版工具一致的桌面区域截图工作流。
- 支持 Windows、macOS 和 Linux 桌面环境。
- 默认保留原文分段，输出纯文本，适合截图翻译、复制和快速阅读。

PaddleOCR-VL 1.6 是文档解析模型，官方说明支持文本、表格、公式、图表、印章等能力。本项目第一版只保留纯文本显示，后续可以扩展 Markdown/JSON 导出。

## 环境要求

- [uv](https://docs.astral.sh/uv/)
- Python 3.11
- 64 位操作系统
- Linux 需要带桌面环境并安装 Tkinter。
- macOS 首次截图时需要在系统设置中授予终端或 Python 屏幕录制权限。

PaddleOCR-VL 1.6 依赖比传统 PP-OCRv5 更重。官方建议安装 `paddleocr[doc-parser]>=3.6.0` 和 PaddlePaddle 3.2.1 或更高版本。

## 创建虚拟环境

```bash
uv python install 3.11
uv sync
```

`uv sync` 会根据 `.python-version` 创建 `.venv` 并安装应用及测试依赖。

## 下载本地模型

```bash
uv run python download_models.py
```

模型会下载到：

```text
models/official_models/PP-DocLayoutV3/
models/official_models/PaddleOCR-VL-1.6/
```

应用启动时会优先使用这两个项目内模型目录。如果目录不存在，PaddleOCR 仍可能回退到自己的用户缓存下载机制。

国内或离线新机器部署见 [DEPLOY_CN.md](DEPLOY_CN.md)。

Linux 如果缺少 Tkinter，请先使用系统包管理器安装。例如 Ubuntu：

```bash
sudo apt-get install python3-tk
```

## 启动

```bash
uv run python launcher.py
```

macOS 也可以直接双击：

```text
start.command
```

Windows 可以直接双击：

```text
start.bat
```

`launcher.py` 会自动检测系统和可用加速通道。默认启动会执行 `uv run`，缺少 `pillow/PIL` 等依赖时会自动同步安装。
启动脚本会把 `UV_HTTP_TIMEOUT` 设置为 300 秒，减少 Windows 或国内网络下载大 wheel 时的超时失败。

如果你希望使用清华 PyPI 源，可以先创建 `.ppocr_use_tuna` 标记，之后 `start.command` / `start.bat` 会自动使用清华源和 PaddlePaddle CPU 包源：

```bash
echo tuna > .ppocr_use_tuna
```

如果你运行了 macOS MLX 或 Windows GPU 加速安装脚本，项目会创建 `.ppocr_no_sync` 标记；之后双击启动脚本会使用 `uv run --no-sync`，避免覆盖你手动安装的加速包：

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
- “快速模式”：在不保留原文分段时跳过布局检测，加快普通截图 OCR。
- “保留原文分段”：按 PaddleOCR-VL 返回的版面 block 重建段落，默认开启。

## 速度优化

默认勾选“保留原文分段”。该模式会开启布局检测，按原图里的文本块重建段落空行，阅读效果更稳定，但比纯快速 OCR 慢。

如果取消勾选“保留原文分段”并勾选“快速模式”，快速模式会：

- 跳过文档布局检测，直接执行 OCR。
- 将大截图最长边压到 1536 像素以内。
- 限制生成长度，减少视觉语言模型输出等待时间。

纯快速模式适合菜单、按钮、短句等不关心段落结构的截图。长文章、新闻正文、复杂版面、表格、公式或多块文档建议保留“保留原文分段”。

这套优化对 Windows 和 macOS 都生效，不依赖平台专属后端。

## 平台专属加速

### macOS Apple Silicon

先安装 MLX-VLM：

```bash
./install_macos_accel.command
```

国内网络环境可以使用清华源：

```bash
./install_macos_accel.command tuna
```

安装脚本会先执行 `uv sync` 补齐基础依赖，再安装 MLX-VLM，并写入 `.ppocr_no_sync`。
脚本会把 `UV_HTTP_TIMEOUT` 设置为 300 秒，避免下载大依赖时 30 秒默认超时。

之后使用 `start.command` 或：

```bash
uv run python launcher.py
```

启动器会自动启动 `mlx_vlm.server --port 8111`，日志写入 `temp/mlx_vlm_server.log`。

### Windows NVIDIA GPU

先确认机器安装了 NVIDIA 驱动，并且 `nvidia-smi` 可用。然后双击：

```text
install_windows_gpu.bat
```

国内网络环境可以在命令行里执行：

```bat
install_windows_gpu.bat tuna
```

安装脚本会先执行 `uv sync` 补齐基础依赖，再把 CPU 版 `paddlepaddle` 换成 GPU 版 `paddlepaddle-gpu`，并写入 `.ppocr_no_sync`。如果 `nvidia-smi` 显示 `CUDA Version: 13.x`，脚本会安装 Paddle 官方 CUDA 13.0 包；否则使用 CUDA 12.6 包。
脚本会把 `UV_HTTP_TIMEOUT` 设置为 300 秒，避免下载 `numpy`、`paddlepaddle-gpu` 等大 wheel 时 30 秒默认超时。

之后使用 `start.bat`。启动器会自动设置 GPU 设备；如果 GPU 版 PaddlePaddle 或驱动不可用，会自动回退 CPU。

## 测试

```bash
uv run pytest
```

测试使用 stub 隔离 PaddleOCR-VL 和桌面 GUI，不需要实际加载模型，也不会弹出窗口。

## 常见问题

### 启动时提示运行依赖缺失

确认当前目录是项目根目录，然后重新同步环境：

```bash
uv sync
```

Windows 如果看到 `No module named 'PIL'`、`No module named 'paddleocr'` 或类似运行依赖缺失，直接重新运行新版 `start.bat` 即可；如果存在 `.ppocr_no_sync`，新版启动脚本会一次性检查 `PIL`、`paddleocr`、`paddle` 并补齐缺失运行依赖，不会执行完整 `uv sync` 覆盖 Windows GPU 加速包。

普通 CPU 环境也可以手动执行：

```bat
uv sync
start.bat
```

如果你曾安装旧版 Windows GPU 加速脚本并存在 `.ppocr_no_sync`，建议重新运行新版 GPU 安装脚本：

```bat
install_windows_gpu.bat
```

### uv 下载 numpy 超时

如果 Windows 上看到类似：

```text
Failed to download `numpy`
Failed to download distribution due to network timeout
Try increasing UV_HTTP_TIMEOUT
```

请使用新版 `start.bat` 或新版安装脚本，它们已经自动设置 `UV_HTTP_TIMEOUT=300`。也可以在命令行临时执行：

```bat
set UV_HTTP_TIMEOUT=300
uv sync
```

### 首次 OCR 很慢

PaddleOCR-VL 1.6 是视觉语言模型，首次运行会初始化较大的模型。请先执行 `uv run python download_models.py` 把模型下载到项目内。

### Windows 或 macOS 仍然很慢

首次识别会加载模型，通常比后续识别慢。框选区域越大，识别越慢；尽量只框选需要识别的文字区域。如果你更重视速度而不是段落结构，可以取消“保留原文分段”并勾选“快速模式”。

macOS Apple Silicon 可安装 MLX-VLM 加速；Windows NVIDIA 机器可安装 GPU 版 PaddlePaddle。安装方式见“平台专属加速”。

如果 Windows 日志出现 `Mismatched GPU Architecture`，例如当前 GPU 是 `120`，说明已安装的 Paddle GPU wheel 不支持你的显卡架构。NVIDIA 50 系显卡通常需要 CUDA 13.0 对应的 Paddle GPU 包，请重新运行新版 `install_windows_gpu.bat`。

如果 Windows 日志出现 `Could not locate cublasLt64_13.dll`，说明 CUDA 13 运行库 DLL 没有被 Paddle 找到。新版程序启动时会递归扫描 `.venv\Lib\site-packages\nvidia\` 下所有 DLL 目录并加入搜索路径，同时预加载 `cublasLt64_13.dll`；新版 `start.bat` 也会在 CUDA 13 GPU 环境下检查这个 DLL 是否存在，缺失时自动补装 `nvidia-cublas`。

如果 Windows 日志出现 `CUDA error(2), out of memory` 或 `cudaErrorMemoryAllocation`，说明 GPU 显存不足。新版启动器在 Windows GPU 环境会默认设置 `PPOCR_GPU_MEMORY_MODE=low`，降低截图输入尺寸和 VLM 生成预算；如果预测阶段仍然 OOM，程序会自动重建 CPU OCR 引擎并重试本次识别。显存充足时可以在启动前设置 `PPOCR_GPU_MEMORY_MODE=normal` 关闭低显存模式。

### macOS 无法截图

在“系统设置 -> 隐私与安全性 -> 屏幕录制”中允许终端或 Python 截取屏幕，然后重新启动程序。

### Linux 无法打开窗口

确认当前系统带有桌面环境，并已安装 Tkinter。远程服务器通常需要配置图形会话后才能使用本工具。

## 参考

- [PaddleOCR-VL 1.6 Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [PaddleOCR-VL 使用文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR-VL Apple Silicon 使用文档](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
