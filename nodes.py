from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from huggingface_hub import hf_hub_download

from .llama_cli import build_command, run_llama_cli
from .model_ram_cache import pin_files, unpin


with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as handle:
    config = json.load(handle)
    CAPTION_TYPE_MAP = config["caption_type_map"]
    EXTRA_OPTIONS = config["extra_options"]
    MODEL_SETTINGS = config["model_settings"]
    CAPTION_LENGTH_CHOICES = config["caption_length_choices"]
    GGUF_MODELS = config["gguf_models"]
    GGUF_SETTINGS = config["gguf_settings"]

custom_path = Path(__file__).parent / "custom_models.json"
if custom_path.exists():
    try:
        with custom_path.open("r", encoding="utf-8") as handle:
            custom_data = json.load(handle) or {}
        GGUF_MODELS.update(custom_data.get("gguf_models", {}))
        print("[JoyCaption-fork] Loaded custom GGUF models")
    except Exception as exc:
        print(f"[JoyCaption-fork] Failed to load custom models: {exc}")


def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2
    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]
    if extra_options:
        prompt += " " + " ".join(extra_options)
    return prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)


def _gguf_dir() -> Path:
    path = Path(folder_paths.models_dir) / "LLM" / "GGUF"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_gguf_model(model_key: str) -> Path:
    spec = GGUF_MODELS[model_key]["name"]
    gguf_dir = _gguf_dir()
    filename = Path(spec).name
    local_path = gguf_dir / filename
    if local_path.exists():
        return local_path
    if "/" not in spec:
        raise ValueError(f"Invalid model path: {spec}")
    repo_path, file_name = spec.rsplit("/", 1)
    return Path(hf_hub_download(
        repo_id=repo_path,
        filename=file_name,
        local_dir=str(gguf_dir),
        local_dir_use_symlinks=False,
    )).resolve()


def resolve_mmproj() -> Path:
    gguf_dir = _gguf_dir()
    filename = GGUF_SETTINGS["mmproj_filename"]
    local_path = gguf_dir / filename
    if local_path.exists():
        return local_path
    return Path(hf_hub_download(
        repo_id="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
        filename=filename,
        local_dir=str(gguf_dir),
        local_dir_use_symlinks=False,
    )).resolve()


def _extra_prompt_parts(extra_options) -> tuple[list[str], str]:
    if not extra_options:
        return [], "{NAME}"
    return extra_options[0], extra_options[1]


def _apply_memory(memory_management: str, model_path: Path, mmproj_path: Path) -> None:
    if memory_management == "Clear After Run":
        unpin()
        return
    pin_files((model_path, mmproj_path))


def _caption(
    image,
    model: str,
    processing_mode: str,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    memory_management: str,
) -> str:
    model_path = resolve_gguf_model(model)
    mmproj_path = resolve_mmproj()
    _apply_memory(memory_management, model_path, mmproj_path)
    command, cleanup_paths = build_command(
        model_path=model_path,
        mmproj_path=mmproj_path,
        image=image,
        system_prompt=MODEL_SETTINGS["default_system_prompt"],
        prompt=prompt_text,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        ctx_size=MODEL_SETTINGS["context_window"],
        processing_mode=processing_mode,
        image_size=tuple(GGUF_SETTINGS["default_image_size"]),
    )
    return run_llama_cli(
        command,
        timeout_seconds=MODEL_SETTINGS["timeout"],
        cleanup_paths=cleanup_paths,
    )


class JoyCaptionForkExtraOptions:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {}}
        for key, value in EXTRA_OPTIONS.items():
            inputs["required"][key] = ("BOOLEAN", {"default": value["default"]})
        inputs["required"]["character_name"] = ("STRING", {
            "default": "",
            "multiline": True,
            "placeholder": "Character Name",
        })
        return inputs

    RETURN_TYPES = ("JOYCAPTION_FORK_EXTRA_OPTIONS",)
    RETURN_NAMES = ("extra_options",)
    FUNCTION = "get_extra_options"
    CATEGORY = "JoyCaption-fork"

    def get_extra_options(self, character_name, **kwargs):
        selected = []
        for key, value in EXTRA_OPTIONS.items():
            if kwargs.get(key, False):
                selected.append(value["description"])
        return ([selected, character_name],)


class JoyCaptionFork:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [key for key, value in GGUF_MODELS.items() if isinstance(value, dict) and "name" in value]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0]}),
                "processing_mode": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any"}),
                "memory_management": (
                    ["Keep in Memory", "Clear After Run", "Global Cache"],
                    {"default": "Keep in Memory"},
                ),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_FORK_EXTRA_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "JoyCaption-fork"
    TITLE = "JoyCaption GGUF (fork)"

    def generate(
        self,
        image,
        model,
        processing_mode,
        prompt_style,
        caption_length,
        memory_management,
        extra_options=None,
    ):
        extras, name = _extra_prompt_parts(extra_options)
        prompt_text = build_prompt(prompt_style, caption_length, extras, name)
        response = _caption(
            image=image,
            model=model,
            processing_mode=processing_mode,
            prompt_text=prompt_text,
            max_new_tokens=MODEL_SETTINGS["default_max_tokens"],
            temperature=MODEL_SETTINGS["default_temperature"],
            top_p=MODEL_SETTINGS["default_top_p"],
            top_k=MODEL_SETTINGS["default_top_k"],
            memory_management=memory_management,
        )
        if memory_management == "Clear After Run":
            unpin()
        return (response,)


class JoyCaptionForkAdv:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [key for key, value in GGUF_MODELS.items() if isinstance(value, dict) and "name" in value]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0]}),
                "processing_mode": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any"}),
                "max_new_tokens": ("INT", {
                    "default": MODEL_SETTINGS["default_max_tokens"],
                    "min": 1,
                    "max": 2048,
                }),
                "temperature": ("FLOAT", {
                    "default": MODEL_SETTINGS["default_temperature"],
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "top_p": ("FLOAT", {
                    "default": MODEL_SETTINGS["default_top_p"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": MODEL_SETTINGS["default_top_k"],
                    "min": 0,
                    "max": 100,
                }),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "memory_management": (
                    ["Keep in Memory", "Clear After Run", "Global Cache"],
                    {"default": "Keep in Memory"},
                ),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_FORK_EXTRA_OPTIONS",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "STRING")
    FUNCTION = "generate"
    CATEGORY = "JoyCaption-fork"
    TITLE = "JoyCaption GGUF Advanced (fork)"

    def generate(
        self,
        image,
        model,
        processing_mode,
        prompt_style,
        caption_length,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        custom_prompt,
        memory_management,
        extra_options=None,
    ):
        extras, name = _extra_prompt_parts(extra_options)
        prompt = custom_prompt.strip() or build_prompt(prompt_style, caption_length, extras, name)
        response = _caption(
            image=image,
            model=model,
            processing_mode=processing_mode,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            memory_management=memory_management,
        )
        if memory_management == "Clear After Run":
            unpin()
        return (prompt, response)


NODE_CLASS_MAPPINGS = {
    "JoyCaptionFork": JoyCaptionFork,
    "JoyCaptionForkAdv": JoyCaptionForkAdv,
    "JoyCaptionForkExtraOptions": JoyCaptionForkExtraOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaptionFork": "JoyCaption GGUF (fork)",
    "JoyCaptionForkAdv": "JoyCaption GGUF Advanced (fork)",
    "JoyCaptionForkExtraOptions": "JoyCaption Extra Options (fork)",
}
