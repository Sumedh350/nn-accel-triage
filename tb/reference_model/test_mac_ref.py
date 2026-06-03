"""pytest unit tests for mac_ref.py — mirrors mac_array.sv semantics."""

import numpy as np
import pytest

from mac_ref import mac_ref


# ── helpers ──────────────────────────────────────────────────────────────────

def _i8(arr) -> np.ndarray:
    return np.array(arr, dtype=np.int8)

def _i16(arr) -> np.ndarray:
    return np.array(arr, dtype=np.int16)


# ── baseline ─────────────────────────────────────────────────────────────────

def test_zeros_int8():
    a = np.zeros((4, 4), dtype=np.int8)
    b = np.zeros((4, 4), dtype=np.int8)
    c = mac_ref(a, b, data_type=0)
    assert c.dtype == np.int32
    assert np.all(c == 0)


def test_identity_int8():
    a = np.eye(4, dtype=np.int8)
    b = np.eye(4, dtype=np.int8)
    c = mac_ref(a, b, data_type=0)
    assert c.dtype == np.int32
    np.testing.assert_array_equal(c, np.eye(4, dtype=np.int32))


# ── exact value checks ────────────────────────────────────────────────────────

def test_known_2x2_int8():
    # [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    a = _i8([[1, 2], [3, 4]])
    b = _i8([[5, 6], [7, 8]])
    c = mac_ref(a, b, data_type=0)
    np.testing.assert_array_equal(c, np.array([[19, 22], [43, 50]], dtype=np.int32))


def test_known_4x4_int8():
    rng = np.random.default_rng(42)
    a = rng.integers(-100, 100, size=(4, 4), dtype=np.int8)
    b = rng.integers(-100, 100, size=(4, 4), dtype=np.int8)
    c = mac_ref(a, b, data_type=0)
    expected = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
    assert c.dtype == np.int32
    np.testing.assert_array_equal(c, expected)


# ── sign-correctness ──────────────────────────────────────────────────────────

def test_negative_times_positive_int8():
    # -128 × 127 must give a negative result
    a = _i8([[-128]])
    b = _i8([[127]])
    c = mac_ref(a, b, data_type=0)
    assert c[0, 0] == -128 * 127   # = -16256


def test_min_times_min_gives_positive_int8():
    # (-128) × (-128) = +16384 (sign × sign = positive)
    a = _i8([[-128]])
    b = _i8([[-128]])
    c = mac_ref(a, b, data_type=0)
    assert c[0, 0] == 16384


def test_max_int8_no_overflow():
    # N=4, all 127×127: each PE accumulates 4 × 127 × 127 = 64516 — fits in int32
    a = np.full((4, 4), 127, dtype=np.int8)
    b = np.full((4, 4), 127, dtype=np.int8)
    c = mac_ref(a, b, data_type=0)
    assert c.dtype == np.int32
    assert np.all(c == 4 * 127 * 127)


# ── N=1 edge case ─────────────────────────────────────────────────────────────

def test_n1_int8():
    a = _i8([[-7]])
    b = _i8([[5]])
    c = mac_ref(a, b, data_type=0)
    assert c[0, 0] == -35


def test_n1_identity_int8():
    a = _i8([[1]])
    b = _i8([[1]])
    c = mac_ref(a, b, data_type=0)
    assert c[0, 0] == 1


# ── INT16 mode ────────────────────────────────────────────────────────────────

def test_int16_dtype_and_value():
    rng = np.random.default_rng(7)
    a = rng.integers(-30_000, 30_000, size=(4, 4), dtype=np.int16)
    b = rng.integers(-30_000, 30_000, size=(4, 4), dtype=np.int16)
    c = mac_ref(a, b, data_type=1)
    assert c.dtype == np.int64
    # For these magnitudes, no 48-bit wrap occurs; result equals plain int64 matmul.
    expected = a.astype(np.int64) @ b.astype(np.int64)
    np.testing.assert_array_equal(c, expected)


def test_int16_known_1x1():
    a = _i16([[1000]])
    b = _i16([[-2]])
    c = mac_ref(a, b, data_type=1)
    assert c.dtype == np.int64
    assert c[0, 0] == -2000


def test_int16_48bit_wrap():
    # Manually construct an accumulator value that requires 48-bit truncation.
    # Inject a pre-computed int64 result that exceeds 48-bit range.
    # max 48-bit signed = 2^47 - 1 = 140737488355327
    # Use large N to force accumulation past this limit.
    # With N=8 and all inputs = 32767 (INT16 max):
    #   each PE acc = 8 × 32767 × 32767 = 8589410312 (≈2^33, fits in 48 bits)
    # So real hardware wrap is only triggered with extreme N; verify truncation helper directly.
    from mac_ref import _truncate_to_int48
    # Value just above INT48_MAX should wrap to negative.
    INT48_MAX = (1 << 47) - 1
    x = np.array([INT48_MAX + 1], dtype=np.int64)   # = 2^47
    wrapped = _truncate_to_int48(x)
    assert wrapped[0] == -(1 << 47)                  # 2^47 wraps to -2^47

    # INT48_MAX stays unchanged.
    x2 = np.array([INT48_MAX], dtype=np.int64)
    assert _truncate_to_int48(x2)[0] == INT48_MAX


# ── FP16 stub ─────────────────────────────────────────────────────────────────

def test_fp16_stub_returns_zeros():
    a = np.ones((4, 4), dtype=np.int16)
    b = np.ones((4, 4), dtype=np.int16)
    c = mac_ref(a, b, data_type=2)
    assert c.dtype == np.int32
    assert np.all(c == 0)


def test_fp16_stub_shape():
    a = np.ones((3, 3), dtype=np.int16)
    b = np.ones((3, 3), dtype=np.int16)
    c = mac_ref(a, b, data_type=2)
    assert c.shape == (3, 3)
