"""Generate/merge oMLX per-model settings without destroying other models."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def default_model_settings(
    draft_model: str,
    *,
    max_context: int = 262_144,
    kv_bits: float = 4,
    fast_prefill: bool = True,
    compress_last_kv: bool = False,
    draft_quantize_at_load: bool = False,
    use_dflash: bool = False,
    dflash_l1_gib: float = 0.5,
    dflash_l2_gib: float = 50.0,
) -> dict[str, Any]:
    """Settings verified against current oMLX ModelSettings field names.

    Q4 TurboQuant cache keeps the live cache smaller on the supported MLX-VLM
    path. The default drafter checkpoint is already quantized to Q4, so
    quantization-at-load remains off unless explicitly requested. SSD settings
    are for reusable prefix states, not reads of the live KV cache per token.
    """
    if max_context < 1 or max_context > 262_144:
        raise ValueError("max_context must be between 1 and the model-native 262,144-token limit")
    if kv_bits not in (2, 2.5, 3, 3.5, 4, 6, 8):
        raise ValueError("kv_bits must be one of 2, 2.5, 3, 3.5, 4, 6, 8")
    if dflash_l1_gib < 0 or dflash_l2_gib < 0:
        raise ValueError("cache budgets cannot be negative")

    return {
        "max_context_window": max_context,
        "turboquant_kv_enabled": True,
        "turboquant_kv_bits": kv_bits,
        "turboquant_skip_last": not compress_last_kv,
        "qwen35_oq_a8_enabled": fast_prefill,
        "qwen35_oq_a8_min_tokens": 128,
        "dflash_enabled": use_dflash,
        "dflash_draft_model": draft_model,
        "dflash_draft_quant_enabled": draft_quantize_at_load,
        "dflash_draft_quant_weight_bits": 4,
        "dflash_draft_quant_activation_bits": 16,
        "dflash_draft_quant_group_size": 64,
        "dflash_in_memory_cache": use_dflash,
        "dflash_in_memory_cache_max_entries": 2,
        "dflash_in_memory_cache_max_bytes": int(dflash_l1_gib * 1024**3),
        "dflash_ssd_cache": use_dflash and dflash_l2_gib > 0,
        "dflash_ssd_cache_max_bytes": int(dflash_l2_gib * 1024**3),
        "dflash_block_size": 5,
        "dflash_verify_mode": "adaptive",
        # None means no DFlash-to-baseline cutover at an arbitrary context size.
        "dflash_max_ctx": None,
    }


def omlx_settings_path() -> Path:
    override = os.environ.get("QWENFORGE_OMLX_HOME")
    return Path(override).expanduser() / "model_settings.json" if override else Path.home() / ".omlx" / "model_settings.json"


def merge_model_settings(
    model_key: str,
    settings: dict[str, Any],
    destination: Path | None = None,
) -> Path:
    if not model_key.strip():
        raise ValueError("model_key cannot be empty")
    destination = destination or omlx_settings_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            root = json.loads(destination.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Cannot read existing settings {destination}: {exc}") from exc
    else:
        root = {"version": 1, "models": {}}

    if not isinstance(root, dict):
        raise RuntimeError("Existing oMLX settings must be a JSON object")
    root.setdefault("version", 1)
    root.setdefault("models", {})
    if not isinstance(root["models"], dict):
        raise RuntimeError("Existing oMLX settings has a non-object 'models' field")

    existing = root["models"].get(model_key, {})
    if not isinstance(existing, dict):
        existing = {}
    existing.update(settings)
    root["models"][model_key] = existing

    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(destination)
    return destination
