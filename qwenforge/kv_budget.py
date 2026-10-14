"""Memory estimates for Qwen3.8-27B's hybrid attention cache.

The estimate deliberately separates packed payload from quantization metadata;
actual allocations vary by MLX/oMLX version, group size, padding and cache scheme.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVEstimate:
    context_tokens: int
    kv_bits: float
    full_attention_layers: int
    kv_heads: int
    head_dim: int
    group_size: int
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / 1_000_000_000

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024**3)


def estimate_qwen35_hybrid_kv(
    context_tokens: int = 262_144,
    kv_bits: float = 4,
    full_attention_layers: int = 16,
    kv_heads: int = 4,
    head_dim: int = 256,
    group_size: int = 64,
    metadata_bytes_per_group: int = 4,
    unquantized_last_layer: bool = False,
) -> KVEstimate:
    """Estimate full-attention KV cache bytes for Qwen3.8-27B.

    GDN/linear-attention recurrent state is fixed-size and excluded. The main
    cache estimate is tokens * layers * K/V(2) * KV heads * head dimension *
    bits / 8, plus estimated scale/bias metadata. Optionally one final
    full-attention layer is priced as FP16 rather than quantized storage.
    """
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if not (0 < kv_bits <= 16):
        raise ValueError("kv_bits must be in (0, 16]")
    for name, value in {
        "full_attention_layers": full_attention_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "group_size": group_size,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if unquantized_last_layer and full_attention_layers < 1:
        raise ValueError("full_attention_layers must be >= 1 when preserving an unquantized layer")

    quantized_layers = full_attention_layers - (1 if unquantized_last_layer else 0)
    quantized_elements = context_tokens * quantized_layers * 2 * kv_heads * head_dim
    payload_bytes = int((quantized_elements * kv_bits + 7) // 8)
    groups = (quantized_elements + group_size - 1) // group_size
    metadata_bytes = groups * metadata_bytes_per_group
    if unquantized_last_layer:
        # The final global-attention layer is kept in FP16 by default in oMLX
        # to avoid cache corruption on sensitive model/runtime combinations.
        last_layer_elements = context_tokens * 2 * kv_heads * head_dim
        payload_bytes += last_layer_elements * 2
    return KVEstimate(
        context_tokens=context_tokens,
        kv_bits=kv_bits,
        full_attention_layers=full_attention_layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        group_size=group_size,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=payload_bytes + metadata_bytes,
    )


def gibibytes(num_bytes: int) -> float:
    return num_bytes / (1024**3)
