from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import comfy.model_management

from .llama_cli_locate import ensure_llama_cli_paths


PROMPT_ECHO_END = "... (truncated)"
PROMPT_PADDING = " " * 501
PERF_RE = re.compile(r"\[\s*Prompt:\s*[^|\]]+\|\s*Generation:\s*[^\]]+\]")
MMPROJ_EMBEDDING_MISMATCH_RE = re.compile(
    r"mismatch between text model \(n_embd = (?P<model>\d+)\) and mmproj \(n_embd = (?P<mmproj>\d+)\)",
    flags=re.IGNORECASE,
)


def _write_temp_text(prefix: str, text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    os.close(fd)
    text_path = Path(path)
    text_path.write_text(text, encoding="utf-8", newline="\n")
    return text_path


def _write_temp_png(image, size: tuple[int, int]) -> Path:
    import numpy as np
    from PIL import Image

    if hasattr(image, "dim") and image.dim() == 4:
        tensor = image[0]
    else:
        tensor = image
    array = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(size, Image.Resampling.BILINEAR)
    fd, path = tempfile.mkstemp(prefix="joycaption-fork-", suffix=".png")
    os.close(fd)
    pil_image.save(path, format="PNG")
    return Path(path)


def n_gpu_layers_for_mode(processing_mode: str) -> int:
    if processing_mode == "CPU":
        return 0
    if processing_mode == "GPU":
        return -1
    try:
        import torch
        return -1 if torch.cuda.is_available() else 0
    except Exception:
        return 0


def build_command(
    model_path: Path,
    mmproj_path: Path,
    image,
    system_prompt: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    ctx_size: int,
    processing_mode: str,
    image_size: tuple[int, int],
) -> tuple[list[str], tuple[Path, ...]]:
    cli = ensure_llama_cli_paths().cli
    image_path = _write_temp_png(image, image_size)
    prompt_path = _write_temp_text("joycaption-fork-prompt-", prompt.strip() + PROMPT_PADDING)
    system_path = _write_temp_text("joycaption-fork-sys-", system_prompt.strip())
    command = [
        str(cli),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--image", str(image_path),
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--repeat-penalty", "1.1",
        "-c", str(ctx_size),
        "--seed", "1",
        "--single-turn",
        "-ngl", str(n_gpu_layers_for_mode(processing_mode)),
        "-sysf", str(system_path),
        "-f", str(prompt_path),
        "-r", "User:",
        "-r", "Assistant:",
        "-r", "</s>",
    ]
    if top_k > 0:
        command.extend(["--top-k", str(top_k)])
    return command, (image_path, prompt_path, system_path)


def _llama_process_env(command: list[str]) -> dict[str, str] | None:
    if os.name == "nt" or not command:
        return None
    cli_dir = Path(command[0]).resolve().parent
    lib_dirs = [str(cli_dir)]
    lib_dir = cli_dir.parent / "lib"
    if lib_dir.is_dir():
        lib_dirs.append(str(lib_dir))
    env = os.environ.copy()
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + ([previous] if previous else []))
    return env


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _communicate_with_interrupt(process: subprocess.Popen, timeout_seconds: int) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if comfy.model_management.processing_interrupted():
            _stop_process(process)
            comfy.model_management.throw_exception_if_processing_interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(process)
            raise TimeoutError(f"llama.cpp timed out after {timeout_seconds}s")
        try:
            return process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue


def _parse_response(text: str) -> str:
    text = str(text or "")
    if PROMPT_ECHO_END in text:
        text = text.split(PROMPT_ECHO_END, 1)[1]
    perf_match = PERF_RE.search(text)
    content = text[:perf_match.start()] if perf_match else text
    return content.strip()


def _parse_llama_error(stderr: str) -> str:
    match = MMPROJ_EMBEDDING_MISMATCH_RE.search(str(stderr or ""))
    if not match:
        return ""
    return (
        "Selected mmproj does not match the text model "
        f"(model n_embd={match.group('model')}, mmproj n_embd={match.group('mmproj')})."
    )


def run_llama_cli(
    command: list[str],
    timeout_seconds: int,
    cleanup_paths: tuple[Path, ...] = (),
) -> str:
    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=_llama_process_env(command),
        )
        stdout, stderr = _communicate_with_interrupt(process, timeout_seconds)
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        if process is not None:
            _stop_process(process)
        raise
    finally:
        for path in cleanup_paths:
            if path.exists():
                path.unlink()

    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = _parse_llama_error(stderr)
        if message:
            raise RuntimeError(message)
        raise RuntimeError(
            f"llama.cpp inference failed with exit code {result.returncode}:\n{stderr}"
        )
    return _parse_response(result.stdout + "\n" + result.stderr)
