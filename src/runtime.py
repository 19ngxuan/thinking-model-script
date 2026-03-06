from __future__ import annotations

from typing import Any

import torch


def resolve_runtime_device(requested: str) -> str:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    if req == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' but CUDA is not available.")
        return "cuda"

    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("Requested device 'mps' but MPS is not available.")
        return "mps"

    if req == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device: {requested}")


def should_use_device_map_auto(requested: str, resolved_device: str) -> bool:
    return requested.lower() == "auto" and resolved_device == "cuda"


def resolve_inference_dtype(resolved_device: str):
    if resolved_device == "cuda":
        return torch.bfloat16
    if resolved_device == "mps":
        return torch.float16
    return torch.float32


def _normalize_map_device(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, int):
        return f"cuda:{value}"

    text = str(value).lower().strip()
    if text in {"cpu", "disk", "meta"}:
        return None

    if text.startswith("cuda"):
        return text
    if text.startswith("mps"):
        return "mps"

    return None


def resolve_generation_device(model, resolved_device: str, use_device_map_auto: bool) -> str:
    if not use_device_map_auto:
        return resolved_device

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            normalized = _normalize_map_device(value)
            if normalized is not None:
                return normalized

    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return resolved_device
