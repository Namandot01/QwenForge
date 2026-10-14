from qwenforge.mlx_engine import _has_repeated_token_cycle


def test_does_not_stop_after_three_copies_of_short_block():
    block = list(range(101, 109))
    assert not _has_repeated_token_cycle(block * 3)


def test_does_not_stop_after_three_copies_of_medium_block():
    block = list(range(200, 232))
    assert not _has_repeated_token_cycle(block * 3)


def test_detects_sustained_16_token_cycle_after_eight_copies():
    block = list(range(16))
    assert _has_repeated_token_cycle(block * 8)


def test_detects_sustained_24_token_cycle_after_six_copies():
    block = list(range(24))
    assert _has_repeated_token_cycle(block * 6)


def test_does_not_flag_two_repetitions():
    block = list(range(1, 17))
    assert not _has_repeated_token_cycle(block * 2)


def test_does_not_flag_nonrepetitive_tokens():
    assert not _has_repeated_token_cycle(list(range(500)))
