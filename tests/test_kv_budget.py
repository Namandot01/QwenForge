from qwenforge.kv_budget import estimate_qwen35_hybrid_kv


def test_q4_full_context_cache_estimate_is_plausible():
    estimate = estimate_qwen35_hybrid_kv(context_tokens=262_144, kv_bits=4)
    assert 4.0 < estimate.total_gib < 5.5


def test_q8_cache_is_larger_than_q4():
    q4 = estimate_qwen35_hybrid_kv(context_tokens=262_144, kv_bits=4)
    q8 = estimate_qwen35_hybrid_kv(context_tokens=262_144, kv_bits=8)
    assert q8.total_bytes > q4.total_bytes
    assert q8.total_bytes / q4.total_bytes > 1.7


def test_cache_scales_linearly_with_context():
    short = estimate_qwen35_hybrid_kv(context_tokens=131_072, kv_bits=4)
    full = estimate_qwen35_hybrid_kv(context_tokens=262_144, kv_bits=4)
    assert full.total_bytes == 2 * short.total_bytes


def test_unquantized_last_layer_costs_more_than_all_q4():
    all_q4 = estimate_qwen35_hybrid_kv(
        context_tokens=262_144, kv_bits=4, unquantized_last_layer=False
    )
    last_fp16 = estimate_qwen35_hybrid_kv(
        context_tokens=262_144, kv_bits=4, unquantized_last_layer=True
    )
    assert last_fp16.total_bytes > all_q4.total_bytes
    assert last_fp16.total_gib - all_q4.total_gib > 0.6
