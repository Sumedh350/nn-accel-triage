"""
Reproducible test vector generator for mac_array and quant_unit.
"""
from __future__ import annotations

import numpy as np


class TestGenerator:
    def gen_mac_vectors(
        self,
        n: int,
        data_type: int,
        count: int,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate `count` (a, b) input pairs for mac_array.

        Each array has shape (n, n) with dtype int8 (data_type=0) or int16
        (data_type=1).  Values span the full hardware input range.
        """
        assert data_type in (0, 1), f"data_type must be 0 or 1, got {data_type}"
        dtype = np.int8 if data_type == 0 else np.int16
        lo = int(np.iinfo(dtype).min)
        hi = int(np.iinfo(dtype).max)
        rng = np.random.default_rng(seed)
        return [
            (
                rng.integers(lo, hi + 1, size=(n, n), dtype=dtype),
                rng.integers(lo, hi + 1, size=(n, n), dtype=dtype),
            )
            for _ in range(count)
        ]

    def gen_quant_vectors(
        self,
        n: int,
        acc_w: int,
        count: int,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate `count` (acc, scale, shift, zero_pt) tuples for quant_unit.

        acc  shape (n, n): bounded to ACC_W-bit signed range (not full int64).
             dtype int32 for acc_w<=32, int64 for acc_w=48.
        scale shape (n,): uint16 per-channel scale, range [0, 65535].
        shift shape (n,): uint8 per-channel shift, range [0, 31].
        zero_pt shape (n,): int8 per-channel zero point, range [-128, 127].
        """
        acc_min = -(1 << (acc_w - 1))
        acc_max = (1 << (acc_w - 1)) - 1
        acc_dtype = np.int32 if acc_w <= 32 else np.int64
        rng = np.random.default_rng(seed)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(count):
            acc = rng.integers(acc_min, acc_max + 1, size=(n, n), dtype=acc_dtype)
            scale = rng.integers(0, 65536, size=(n,), dtype=np.uint16)
            shift = rng.integers(0, 32, size=(n,), dtype=np.uint8)
            zero_pt = rng.integers(-128, 128, size=(n,), dtype=np.int8)
            results.append((acc, scale, shift, zero_pt))
        return results
