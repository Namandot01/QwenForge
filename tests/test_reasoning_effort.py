from __future__ import annotations

import pytest

from qwenforge.mlx_engine import _apply_request_reasoning_effort


@pytest.mark.parametrize(
    ("req_data", "expected"),
    [
        ({"reasoning_effort": "low"}, "low"),
        ({"reasoningEffort": "xhigh"}, "xhigh"),
        ({"options": {"reasoning_effort": "medium"}}, "medium"),
        ({"options": {"reasoningEffort": "low"}}, "low"),
        ({"providerOptions": {"reasoningEffort": "medium"}}, "medium"),
        ({"options": {"reasoning": {"effort": "xhigh"}}}, "xhigh"),
    ],
)
def test_apply_request_reasoning_effort_accepts_common_shapes(req_data, expected):
    template_kwargs = {"enable_thinking": True}
    _apply_request_reasoning_effort(req_data, template_kwargs)
    assert template_kwargs["reasoning_effort"] == expected


def test_explicit_chat_template_kwargs_has_precedence():
    template_kwargs = {"reasoning_effort": "low"}
    _apply_request_reasoning_effort(
        {"reasoning_effort": "xhigh", "options": {"reasoning_effort": "medium"}},
        template_kwargs,
    )
    assert template_kwargs["reasoning_effort"] == "low"


def test_rejects_unsupported_high_tier():
    with pytest.raises(ValueError, match="low, medium, xhigh"):
        _apply_request_reasoning_effort({"options": {"reasoning_effort": "high"}}, {})


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen3.8-27B-4bit-low", "low"),
        ("Qwen3.8-27B-4bit-medium", "medium"),
        ("Qwen3.8-27B-4bit-xhigh", "xhigh"),
        ("Qwen3.8-27B-4bit", None),
        ("other-model-xhigh", None),
        (None, None),
    ],
)
def test_model_alias_reasoning_effort(model_id, expected):
    from qwenforge.mlx_engine import _reasoning_effort_from_model_alias

    assert _reasoning_effort_from_model_alias(model_id, "Qwen3.8-27B-4bit") == expected
