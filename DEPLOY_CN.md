# 国内/离线部署说明

## 结论

PaddleOCR-VL 1.6 的依赖和模型体积明显大于传统 PP-OCRv5。新机器部署时需要同时考虑 Python wheel 下载和项目内 `models/` 目录。

## 场景 A：新机器可以访问清华 PyPI 和 Paddle 官方 CPU 包源

```bash
cd /path/to/ppocr_v1.6
uv python install 3.11
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
UV_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
UV_INDEX_STRATEGY=unsafe-best-match \
uv sync
```

也可以让项目启动脚本记住清华源设置：

```bash
echo tuna > .ppocr_use_tuna
```

之后双击 `start.command` 或 `start.bat` 时，会自动使用：

- `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`
- `UV_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/`
- `UV_INDEX_STRATEGY=unsafe-best-match`
- `UV_HTTP_TIMEOUT=300`

## 平台加速安装

macOS Apple Silicon 安装 MLX-VLM 加速：

```bash
./install_macos_accel.command tuna
```

Windows NVIDIA GPU 安装 PaddlePaddle GPU 包：

```bat
install_windows_gpu.bat tuna
```

Windows 脚本会读取 `nvidia-smi` 输出：如果显示 `CUDA Version: 13.x`，安装 Paddle 官方 CUDA 13.0 包；否则安装 CUDA 12.6 包。这两个安装脚本都会先执行 `uv sync` 补齐基础依赖，再安装平台专属加速包。这样即使脚本写入 `.ppocr_no_sync`，后续启动也不会因为跳过同步而缺少 `PIL/Pillow`。
脚本同时会设置 `UV_HTTP_TIMEOUT=300`，减少 `numpy`、`paddlepaddle` 等大 wheel 在国内网络下下载超时。

如果 CUDA 13 环境报 `Could not locate cublasLt64_13.dll`，请重新运行新版 `install_windows_gpu.bat tuna`。新版脚本会显式安装 `nvidia-cublas`，程序启动时也会自动把 `.venv\Lib\site-packages\nvidia\*\bin` 加入 Windows DLL 搜索路径。

如果机器已经有 Python 3.11，不希望 `uv` 下载 Python：

```bash
UV_PYTHON_DOWNLOADS=never \
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
UV_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
UV_INDEX_STRATEGY=unsafe-best-match \
uv sync --python 3.11
```

## 场景 B：新机器完全不能访问外网

在有网机器上准备 wheelhouse。必须与目标机器的操作系统和 CPU 架构一致。

```bash
cd /path/to/ppocr_v1.6
uv sync
uv run python download_models.py
uv export --locked --all-groups --no-emit-project --no-hashes -o requirements.lock.txt
uv run python -m pip download \
  -r requirements.lock.txt \
  -d wheelhouse \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

把整个工程目录、`models/` 和 `wheelhouse/` 一起拷到新机器。

在新机器上执行：

```bash
cd /path/to/ppocr_v1.6
uv sync \
  --no-index \
  --find-links ./wheelhouse \
  --no-python-downloads
```

PaddleOCR-VL 1.6 模型需要提前在有网机器上通过 `uv run python download_models.py` 下载到 `models/official_models/`，或在目标机器首次启动时允许访问 Hugging Face、ModelScope 或 PaddleOCR 官方模型源。

## 场景 C：复用 uv 缓存

如果新机器与构建机器系统/架构一致，也可以复制 `uv` 缓存目录，但 wheelhouse 更直观、可审计。

```bash
uv cache dir
```

查看缓存路径后复制到新机器，并设置：

```bash
export UV_CACHE_DIR=/path/to/copied/uv-cache
uv sync --offline --no-python-downloads
```
