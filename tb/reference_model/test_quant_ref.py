"""pytest unit tests for quant_ref.py — mirrors quant_unit.sv semantics."""

import numpy as np
import pytest

from quant_ref import quant_ref


# ── helper ────────────────────────────────────────────────────────────────────

def _make(n: int, acc_val: int, s: int, sh: int, zp: int):
    """Return uniform N×N inputs for a single-valued channel test."""
    acc     = np.full((n, n), acc_val, dtype=np.int32)
    scale   = np.full(n, s,   dtype=np.uint16)
    shift   = np.full(n, sh,  dtype=np.uint8)
    zero_pt = np.full(n, zp,  dtype=np.int8)
    return acc, scale, shift, zero_pt


# ── passthrough / no-op ───────────────────────────────────────────────────────

def test_passthrough_positive():
    # scale=1, shift=0, zp=0 → q = clip(acc, -128, 127) = acc  (acc in range)
    acc, scale, shift, zp = _make(1, 50, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 50


def test_passthrough_negative():
    acc, scale, shift, zp = _make(1, -50, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -50


def test_zero_acc_zero_zp():
    acc, scale, shift, zp = _make(1, 0, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 0


# ── clamping ──────────────────────────────────────────────────────────────────

def test_clamp_high():
    # 256 * 1 = 256; >>1 = 128; +0 = 128 → clamped to 127
    acc, scale, shift, zp = _make(1, 256, 1, 1, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 127


def test_clamp_low_small():
    # -300 * 1 >>0 +0 = -300 → clamped to -128
    acc, scale, shift, zp = _make(1, -300, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -128


def test_clamp_high_large_acc():
    acc, scale, shift, zp = _make(1, 2**30, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 127


def test_clamp_low_large_negative_acc():
    acc, scale, shift, zp = _make(1, -(2**30), 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -128


def test_boundary_127_not_clamped():
    acc, scale, shift, zp = _make(1, 127, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 127


def test_boundary_neg128_not_clamped():
    acc, scale, shift, zp = _make(1, -128, 1, 0, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -128


# ── arithmetic right-shift (must round toward −∞, not toward 0) ──────────────

def test_arithmetic_rshift_neg7():
    # -7 >> 1 = floor(-7/2) = -4  (NOT -3)
    acc, scale, shift, zp = _make(1, -7, 1, 1, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -4


def test_arithmetic_rshift_neg1():
    # -1 >> 1 = floor(-1/2) = -1  (NOT 0)
    acc, scale, shift, zp = _make(1, -1, 1, 1, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -1


def test_arithmetic_rshift_positive_is_truncation():
    # Positive values: 7 >> 1 = 3  (floor(7/2) = 3, same as truncation)
    acc, scale, shift, zp = _make(1, 7, 1, 1, 0)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 3


def test_shift_zero():
    # shift=0: no shift applied; effectively multiply+bias only
    acc, scale, shift, zp = _make(1, 10, 3, 0, 0)
    # 10 * 3 >>0 +0 = 30
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 30


def test_shift_max_positive():
    # shift=31 on a value < 2^31: result is 0
    acc, scale, shift, zp = _make(1, 2**31 - 1, 1, 31, 0)
    # (2^31-1) >> 31 = floor((2^31-1)/2^31) = 0
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 0


# ── zero-point offset ─────────────────────────────────────────────────────────

def test_zero_point_positive_offset():
    # acc=0, any scale/shift → prod=0 → q = clamp(zp, ...)
    acc, scale, shift, zp = _make(1, 0, 65535, 15, 42)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 42


def test_zero_point_negative_offset():
    acc, scale, shift, zp = _make(1, 0, 1, 0, -10)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == -10


def test_zero_point_pushes_over_limit():
    # acc=100 (in range), but +30 zp → 130 > 127 → clamped to 127
    acc, scale, shift, zp = _make(1, 100, 1, 0, 30)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 127


# ── scale=0 ───────────────────────────────────────────────────────────────────

def test_scale_zero():
    # scale=0 → product is 0 regardless of acc → q = clamp(zp, ...)
    acc, scale, shift, zp = _make(1, 1000, 0, 0, 50)
    assert quant_ref(acc, scale, shift, zp)[0, 0] == 50


# ── per-channel independence ──────────────────────────────────────────────────

def test_per_channel_independent():
    # All acc = 0; zero_pt differs per row → each output row = its own zero_pt
    N = 4
    acc     = np.zeros((N, N), dtype=np.int32)
    scale   = np.ones(N, dtype=np.uint16)
    shift   = np.zeros(N, dtype=np.uint8)
    zero_pt = np.arange(N, dtype=np.int8)   # [0, 1, 2, 3]
    q = quant_ref(acc, scale, shift, zero_pt)
    for i in range(N):
        assert np.all(q[i] == i), f"row {i}: expected {i}, got {q[i]}"


def test_per_channel_different_scales():
    # One acc value, four rows with scale = 1,2,4,8 and shift=0, zp=0
    # Results should be acc*scale for each row (no clamping).
    N = 4
    acc     = np.full((N, N), 3, dtype=np.int32)
    scale   = np.array([1, 2, 4, 8], dtype=np.uint16)
    shift   = np.zeros(N, dtype=np.uint8)
    zero_pt = np.zeros(N, dtype=np.int8)
    q = quant_ref(acc, scale, shift, zero_pt)
    expected = np.array([3, 6, 12, 24], dtype=np.int8)
    for i in range(N):
        assert np.all(q[i] == expected[i])


# ── N=1 edge case ─────────────────────────────────────────────────────────────

def test_n1():
    # 100 * 2 = 200; >>1 = 100; +(-5) = 95
    acc     = np.array([[100]], dtype=np.int32)
    scale   = np.array([2],   dtype=np.uint16)
    shift   = np.array([1],   dtype=np.uint8)
    zero_pt = np.array([-5],  dtype=np.int8)
    q = quant_ref(acc, scale, shift, zero_pt)
    assert q[0, 0] == 95


# ── hand-verifiable 4×4 integration test ─────────────────────────────────────

def test_full_tile_4x4_hand_computed():
    """Each element verified by hand against the RTL formula."""
    acc = np.array([
        [3,   -3,   0,   127],
        [100,  200, -100, -200],
        [0,    0,   0,    0  ],
        [1,    2,   3,    4  ],
    ], dtype=np.int32)
    scale   = np.array([4,  1,  1, 16], dtype=np.uint16)
    shift   = np.array([1,  0,  0,  3], dtype=np.uint8)
    zero_pt = np.array([2,  0, 10,  0], dtype=np.int8)

    q = quant_ref(acc, scale, shift, zero_pt)

    # Row 0: scale=4, shift=1, zp=2
    #   j=0: 3*4=12   >>1=6    +2=8    → 8
    #   j=1: -3*4=-12 >>1=-6   +2=-4   → -4
    #   j=2: 0*4=0    >>1=0    +2=2    → 2
    #   j=3: 127*4=508 >>1=254 +2=256  → clamp→127
    # Row 1: scale=1, shift=0, zp=0  → clip(acc, -128, 127)
    #   100→100, 200→127, -100→-100, -200→-128
    # Row 2: scale=1, shift=0, zp=10, acc=0 → 0+10=10 for all j
    # Row 3: scale=16, shift=3, zp=0
    #   1→16>>3=2, 2→32>>3=4, 3→48>>3=6, 4→64>>3=8
    expected = np.array([
        [8,   -4,  2,   127],
        [100, 127, -100, -128],
        [10,  10,  10,   10 ],
        [2,   4,   6,    8  ],
    ], dtype=np.int8)

    np.testing.assert_array_equal(q, expected)
