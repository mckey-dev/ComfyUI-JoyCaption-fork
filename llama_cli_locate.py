from __future__ import annotations

import importlib.util
from pathlib import Path


PACKAGE_DIR_NAME = "ComfyUI-llama-cli"
MISSING_INSTALLER = "Install custom_nodes/ComfyUI-llama-cli and restart ComfyUI."


def _custom_node_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_dir():
            return
        seen.add(resolved)
        roots.append(resolved)

    try:
        import folder_paths
        for path in folder_paths.get_folder_paths("custom_nodes"):
            add(Path(path))
        base = Path(folder_paths.base_path)
        add(base / "custom_nodes")
        add(base / "custom_node")
    except Exception:
        pass
    add(Path(__file__).resolve().parent.parent)
    return roots


def _package_dirs(root: Path):
    wanted = PACKAGE_DIR_NAME.casefold()
    try:
        children = root.iterdir()
    except OSError:
        return
    for child in children:
        if child.is_dir() and child.name.casefold() == wanted:
            yield child


def load_llama_cli_installer():
    for root in _custom_node_roots():
        for package in _package_dirs(root):
            binary = package / "llama_binary.py"
            if not binary.is_file():
                continue
            spec = importlib.util.spec_from_file_location("comfyui_llama_cli_binary", binary)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(MISSING_INSTALLER)


def ensure_llama_cli_paths():
    return load_llama_cli_installer().ensure_llama_cli_paths()
