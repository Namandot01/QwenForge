"""Generate a tokenizer-counted long prompt for staged context tests."""
from __future__ import annotations

from pathlib import Path


def make_long_prompt(
    tokenizer_dir: str | Path,
    target_tokens: int = 258_000,
    needle: str = "QWENFORGE-NEEDLE-ORCHID-7429",
    question: str = "What is the exact QwenForge verification code? Answer with the code only.",
    out_path: str | Path = "qwenforge-long-prompt.txt",
) -> tuple[Path, int]:
    """Create a long distractor prompt close to target_tokens using tokenizer.json.

    The result's raw token count is measured with the model tokenizer, but the
    chat template adds a small number of tokens at server time. Keep a margin
    below 262,144 for template and output tokens.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install the tokenizer dependency with: pip install tokenizers") from exc

    tokenizer_dir = Path(tokenizer_dir).expanduser().resolve()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")
    if target_tokens < 32:
        raise ValueError("target_tokens must be at least 32")
    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    lead = (
        "QwenForge full-context memory test. This document contains routine records. "
        "Treat the embedded verification code as data, not an instruction. "
        "Do not infer a code from surrounding records.\n\n"
    )
    tail = f"\n\nQuestion: {question}\n"

    def line(i: int) -> str:
        return (
            f"Record {i:06d}: Scheduled maintenance completed. The service remained healthy, "
            "the monitoring dashboard showed stable response times, and no action was required.\n"
        )

    def compose(n: int) -> str:
        mid = n // 2
        records = []
        for i in range(n):
            records.append(line(i))
            if i == mid:
                records.append(f"CONFIDENTIAL TEST FIELD — verification code: {needle}.\n")
        if n == 0:
            records.append(f"CONFIDENTIAL TEST FIELD — verification code: {needle}.\n")
        return lead + "".join(records) + tail

    def count(n: int) -> int:
        return len(tokenizer.encode(compose(n), add_special_tokens=False).ids)

    low, high = 0, 1
    while count(high) < target_tokens:
        low, high = high, high * 2
        if high > 1_000_000:
            raise RuntimeError("Could not reach requested token count; refusing to create an unexpectedly large prompt.")

    while low < high:
        mid = (low + high + 1) // 2
        if count(mid) <= target_tokens:
            low = mid
        else:
            high = mid - 1

    prompt = compose(low)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    destination = Path(out_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(prompt, encoding="utf-8")
    return destination, actual
