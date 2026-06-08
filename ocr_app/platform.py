import ctypes
import os
import site
import socket
import sys
import sysconfig
from pathlib import Path
from urllib.parse import urlparse


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
