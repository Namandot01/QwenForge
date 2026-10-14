import json
from pathlib import Path

import pytest

from qwenforge.settings import default_model_settings, merge_model_settings


def test_capacity_first_settings_use_q4_kv_and_skip_last_by_default():
    settings = default_model_settings("seabit-ai/Qwen3.8-27B-DFlash2-4bit")
    assert settings["max_context_window"] == 262_144
    assert settings["turboquant_kv_enabled"] is True
    assert settings["turboquant_kv_bits"] == 4
    assert settings["turboquant_skip_last"] is True
    assert settings["dflash_enabled"] is False
    assert settings["dflash_draft_quant_enabled"] is False


def test_full_compression_and_dflash_are_explicit_opt_ins():
    settings = default_model_settings(
        "seabit-ai/Qwen3.8-27B-DFlash2-4bit",
        compress_last_kv=True,
        use_dflash=True,
    )
    assert settings["turboquant_skip_last"] is False
    assert settings["dflash_enabled"] is True
    assert settings["dflash_in_memory_cache"] is True
    assert settings["dflash_ssd_cache"] is True


def test_context_cannot_exceed_native_limit():
    with pytest.raises(ValueError, match="262,144"):
        default_model_settings("draft", max_context=262_145)


def test_settings_merge_preserves_other_model(tmp_path: Path):
    path = tmp_path / "model_settings.json"
    path.write_text(json.dumps({"version": 1, "models": {"other": {"temperature": 0.2}}}))
    merge_model_settings("qwen", {"max_context_window": 262_144}, path)
    data = json.loads(path.read_text())
    assert data["models"]["other"] == {"temperature": 0.2}
    assert data["models"]["qwen"]["max_context_window"] == 262_144
