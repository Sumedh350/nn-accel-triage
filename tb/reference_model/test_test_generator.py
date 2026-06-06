"""
pytest tests for TestGenerator.
"""
import numpy as np
import pytest
from test_generator import TestGenerator


_tg = TestGenerator()

# ── gen_mac_vectors: shape, dtype, count ────────────────────────────────────

def test_mac_vectors_count():
    vecs = _tg.gen_mac_vectors(n=4, data_type=0, count=5, seed=1)
    assert len(vecs) == 5


def test_mac_vectors_shapes_int8():
    vecs = _tg.gen_mac_vectors(n=4, data_type=0, count=3, seed=2)
    for a, b in vecs:
        assert a.shape == (4, 4)
        assert b.shape == (4, 4)


def test_mac_vectors_shapes_int16():
    vecs = _tg.gen_mac_vectors(n=3, data_type=1, count=2, seed=3)
    for a, b in vecs:
        assert a.shape == (3, 3)
        assert b.shape == (3, 3)


def test_mac_vectors_dtype_int8():
    vecs = _tg.gen_mac_vectors(n=4, data_type=0, count=1, seed=4)
    a, b = vecs[0]
    assert a.dtype == np.int8
    assert b.dtype == np.int8


def test_mac_vectors_dtype_int16():
    vecs = _tg.gen_mac_vectors(n=4, data_type=1, count=1, seed=5)
    a, b = vecs[0]
    assert a.dtype == np.int16
    assert b.dtype == np.int16


# ── gen_mac_vectors: value bounds ────────────────────────────────────────────

def test_mac_vectors_bounds_int8():
    vecs = _tg.gen_mac_vectors(n=4, data_type=0, count=20, seed=6)
    for a, b in vecs:
        assert int(a.min()) >= -128 and int(a.max()) <= 127
        assert int(b.min()) >= -128 and int(b.max()) <= 127


def test_mac_vectors_bounds_int16():
    vecs = _tg.gen_mac_vectors(n=4, data_type=1, count=20, seed=7)
    for a, b in vecs:
        assert int(a.min()) >= -32768 and int(a.max()) <= 32767
        assert int(b.min()) >= -32768 and int(b.max()) <= 32767


# ── gen_mac_vectors: reproducibility ────────────────────────────────────────

def test_mac_vectors_same_seed_reproducible():
    v1 = _tg.gen_mac_vectors(n=4, data_type=0, count=3, seed=42)
    v2 = _tg.gen_mac_vectors(n=4, data_type=0, count=3, seed=42)
    for (a1, b1), (a2, b2) in zip(v1, v2):
        assert np.array_equal(a1, a2)
        assert np.array_equal(b1, b2)


def test_mac_vectors_different_seeds_differ():
    v1 = _tg.gen_mac_vectors(n=4, data_type=0, count=1, seed=10)
    v2 = _tg.gen_mac_vectors(n=4, data_type=0, count=1, seed=11)
    a1, _ = v1[0]
    a2, _ = v2[0]
    assert not np.array_equal(a1, a2)


# ── gen_quant_vectors: shape, dtype, count ───────────────────────────────────

def test_quant_vectors_count():
    vecs = _tg.gen_quant_vectors(n=4, acc_w=32, count=7, seed=1)
    assert len(vecs) == 7


def test_quant_vectors_shapes():
    vecs = _tg.gen_quant_vectors(n=4, acc_w=32, count=2, seed=2)
    for acc, scale, shift, zero_pt in vecs:
        assert acc.shape == (4, 4)
        assert scale.shape == (4,)
        assert shift.shape == (4,)
        assert zero_pt.shape == (4,)


def test_quant_vectors_dtypes_acc32():
    vecs = _tg.gen_quant_vectors(n=4, acc_w=32, count=1, seed=3)
    acc, scale, shift, zero_pt = vecs[0]
    assert acc.dtype == np.int32
    assert scale.dtype == np.uint16
    assert shift.dtype == np.uint8
    assert zero_pt.dtype == np.int8


def test_quant_vectors_dtypes_acc48():
    vecs = _tg.gen_quant_vectors(n=4, acc_w=48, count=1, seed=4)
    acc, scale, shift, zero_pt = vecs[0]
    assert acc.dtype == np.int64
    assert scale.dtype == np.uint16
    assert shift.dtype == np.uint8
    assert zero_pt.dtype == np.int8


# ── gen_quant_vectors: value bounds ──────────────────────────────────────────

def test_quant_vectors_acc_bounds_32():
    # CLAUDE.md invariant: acc must be bounded to ACC_W-bit range, not int32 max
    acc_min = -(1 << 31)
    acc_max = (1 << 31) - 1
    vecs = _tg.gen_quant_vectors(n=4, acc_w=32, count=20, seed=5)
    for acc, *_ in vecs:
        assert int(acc.min()) >= acc_min
        assert int(acc.max()) <= acc_max


def test_quant_vectors_acc_bounds_48():
    # Critical: must NOT use full int64 range — only 48-bit signed range
    acc_min = -(1 << 47)
    acc_max = (1 << 47) - 1
    vecs = _tg.gen_quant_vectors(n=4, acc_w=48, count=20, seed=6)
    for acc, *_ in vecs:
        assert int(acc.min()) >= acc_min, "acc underflows 48-bit range"
        assert int(acc.max()) <= acc_max, "acc overflows 48-bit range"


def test_quant_vectors_shift_bounds():
    vecs = _tg.gen_quant_vectors(n=4, acc_w=32, count=20, seed=7)
    for _, _, shift, _ in vecs:
        assert int(shift.min()) >= 0
        assert int(shift.max()) <= 31


# ── gen_quant_vectors: reproducibility ──────────────────────────────────────

def test_quant_vectors_same_seed_reproducible():
    v1 = _tg.gen_quant_vectors(n=4, acc_w=32, count=3, seed=99)
    v2 = _tg.gen_quant_vectors(n=4, acc_w=32, count=3, seed=99)
    for (a1, s1, sh1, z1), (a2, s2, sh2, z2) in zip(v1, v2):
        assert np.array_equal(a1, a2)
        assert np.array_equal(s1, s2)
        assert np.array_equal(sh1, sh2)
        assert np.array_equal(z1, z2)


def test_quant_vectors_different_seeds_differ():
    v1 = _tg.gen_quant_vectors(n=4, acc_w=32, count=1, seed=20)
    v2 = _tg.gen_quant_vectors(n=4, acc_w=32, count=1, seed=21)
    acc1, *_ = v1[0]
    acc2, *_ = v2[0]
    assert not np.array_equal(acc1, acc2)


# ── edge cases ───────────────────────────────────────────────────────────────

def test_edge_case_n1_count1_mac():
    vecs = _tg.gen_mac_vectors(n=1, data_type=0, count=1, seed=0)
    assert len(vecs) == 1
    a, b = vecs[0]
    assert a.shape == (1, 1)
    assert a.dtype == np.int8


def test_edge_case_n1_count1_quant():
    vecs = _tg.gen_quant_vectors(n=1, acc_w=32, count=1, seed=0)
    assert len(vecs) == 1
    acc, scale, shift, zero_pt = vecs[0]
    assert acc.shape == (1, 1)
    assert scale.shape == (1,)
    assert shift.shape == (1,)
    assert zero_pt.shape == (1,)
