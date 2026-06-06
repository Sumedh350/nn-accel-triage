"""
CocoTB 2.0.x testbench for mac_array_wrap / mac_array.

Configuration is passed via environment variables set by the Makefile:
  MAC_N          : array dimension (default 4)
  MAC_DATA_TYPE  : 0=INT8, 1=INT16, 2=FP16-stub (default 0)

All expected values are computed by mac_ref.py — no hardcoded results.
"""

import os
import sys
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

# Locate the reference model relative to this file's position in the repo.
_REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_model")
if _REF_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_REF_DIR))

from mac_ref import mac_ref  # noqa: E402

# ── Compile-time configuration (read once from env) ───────────────────────────
N         = int(os.environ.get("MAC_N", "4"))
DATA_TYPE = int(os.environ.get("MAC_DATA_TYPE", "0"))
DATA_W    = 8  if DATA_TYPE == 0 else 16
ACC_W     = 48 if DATA_TYPE == 1 else 32
CLK_NS    = 10  # clock period in nanoseconds

# Numpy dtype for random generation
_IN_DTYPE = np.int8 if DATA_TYPE == 0 else np.int16


# ── Packing helpers ───────────────────────────────────────────────────────────

def pack_matrix(m: np.ndarray, elem_bits: int) -> int:
    """Row-major pack: element [i][j] → bit offset (i*N+j)*elem_bits."""
    n    = m.shape[0]
    mask = (1 << elem_bits) - 1
    val  = 0
    for i in range(n):
        for j in range(n):
            val |= (int(m[i, j]) & mask) << ((i * n + j) * elem_bits)
    return val


def unpack_matrix(raw: int, n: int, elem_bits: int) -> np.ndarray:
    """Row-major unpack with sign reconstruction."""
    mask     = (1 << elem_bits) - 1
    sign_bit = 1 << (elem_bits - 1)
    out      = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            e = (raw >> ((i * n + j) * elem_bits)) & mask
            if e & sign_bit:
                e -= (1 << elem_bits)
            out[i, j] = e
    return out


def assert_tile_equal(got: np.ndarray, expected: np.ndarray, label: str = ""):
    """Compare DUT output tile against reference; raise on mismatch."""
    got_i64 = got.astype(np.int64)
    exp_i64 = expected.astype(np.int64)
    if not np.array_equal(got_i64, exp_i64):
        diff = got_i64 - exp_i64
        raise AssertionError(
            f"{label}\n  expected:\n{exp_i64}\n  got:\n{got_i64}\n  diff:\n{diff}"
        )


# ── Coroutines ────────────────────────────────────────────────────────────────

async def reset_dut(dut):
    """Apply 5-cycle synchronous active-low reset; leave in clean IDLE state."""
    dut.rst_n.value     = 0
    dut.in_valid.value  = 0
    dut.out_ready.value = 0
    dut.flat_a.value    = 0
    dut.flat_b.value    = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_tile(dut, flat_a: int, flat_b: int):
    """Drive flat_a/flat_b and complete the in_valid/in_ready handshake."""
    dut.flat_a.value   = flat_a
    dut.flat_b.value   = flat_b
    dut.in_valid.value = 1
    # Sample in_ready in the settled (ReadOnly) phase before each edge.
    await ReadOnly()
    while not dut.in_ready.value:
        await RisingEdge(dut.clk)
        await ReadOnly()
    # in_ready is 1 now; the next rising edge is the handshake edge.
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0


async def recv_tile(dut) -> int:
    """Assert out_ready, wait for out_valid, complete handshake, return flat_c."""
    dut.out_ready.value = 1
    await ReadOnly()
    while not dut.out_valid.value:
        await RisingEdge(dut.clk)
        await ReadOnly()
    result = int(dut.flat_c.value)
    # Take the handshake edge; after this the DUT transitions DONE → IDLE.
    await RisingEdge(dut.clk)
    dut.out_ready.value = 0
    return result


async def run_tile(dut, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Send one tile, receive result, return unpacked (N×N) numpy array."""
    fa = pack_matrix(a, DATA_W)
    fb = pack_matrix(b, DATA_W)
    await send_tile(dut, fa, fb)
    raw = await recv_tile(dut)
    return unpack_matrix(raw, N, ACC_W)


# ── Tests ─────────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_zeros(dut):
    """All-zero A and B must produce an all-zero result tile."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    a = np.zeros((N, N), dtype=_IN_DTYPE)
    b = np.zeros((N, N), dtype=_IN_DTYPE)
    got      = await run_tile(dut, a, b)
    expected = mac_ref(a, b, data_type=DATA_TYPE)
    assert_tile_equal(got, expected, "test_zeros")


@cocotb.test()
async def test_random(dut):
    """Five random tile pairs must match mac_ref exactly."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=42)
    for iteration in range(5):
        lo, hi = np.iinfo(_IN_DTYPE).min, np.iinfo(_IN_DTYPE).max
        a = rng.integers(lo, hi + 1, size=(N, N), dtype=_IN_DTYPE)
        b = rng.integers(lo, hi + 1, size=(N, N), dtype=_IN_DTYPE)
        got      = await run_tile(dut, a, b)
        expected = mac_ref(a, b, data_type=DATA_TYPE)
        assert_tile_equal(got, expected, f"test_random iter={iteration}")


@cocotb.test()
async def test_max_values(dut):
    """Tiles filled with max-positive and max-negative values."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    iinfo = np.iinfo(_IN_DTYPE)
    for val in (iinfo.max, iinfo.min):
        a = np.full((N, N), val, dtype=_IN_DTYPE)
        b = np.full((N, N), val, dtype=_IN_DTYPE)
        got      = await run_tile(dut, a, b)
        expected = mac_ref(a, b, data_type=DATA_TYPE)
        assert_tile_equal(got, expected, f"test_max_values val={val}")


@cocotb.test()
async def test_identity(dut):
    """I × M == M and M × I == M for a random tile M."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng   = np.random.default_rng(seed=7)
    iinfo = np.iinfo(_IN_DTYPE)
    m     = rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE)
    eye   = np.eye(N, dtype=_IN_DTYPE)

    # I × M
    got      = await run_tile(dut, eye, m)
    expected = mac_ref(eye, m, data_type=DATA_TYPE)
    assert_tile_equal(got, expected, "test_identity I×M")

    # M × I
    got      = await run_tile(dut, m, eye)
    expected = mac_ref(m, eye, data_type=DATA_TYPE)
    assert_tile_equal(got, expected, "test_identity M×I")


@cocotb.test()
async def test_reset(dut):
    """Assert reset while a computation is in-flight; DUT must recover cleanly."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng   = np.random.default_rng(seed=99)
    iinfo = np.iinfo(_IN_DTYPE)
    a     = rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE)
    b     = rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE)

    # Kick off a transaction then immediately reset.
    fa = pack_matrix(a, DATA_W)
    fb = pack_matrix(b, DATA_W)
    dut.flat_a.value   = fa
    dut.flat_b.value   = fb
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)          # handshake (DUT was in IDLE)
    dut.in_valid.value = 0

    # Assert reset mid-COMPUTE.
    await RisingEdge(dut.clk)
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # DUT should be back in IDLE — read directly in Active phase (no ReadOnly needed).
    assert dut.in_ready.value  == 1, "in_ready not 1 after reset"
    assert dut.out_valid.value == 0, "out_valid not 0 after reset"

    # Verify a normal transaction works after recovery.
    got      = await run_tile(dut, a, b)
    expected = mac_ref(a, b, data_type=DATA_TYPE)
    assert_tile_equal(got, expected, "test_reset post-recovery")


@cocotb.test()
async def test_backpressure(dut):
    """Withhold out_ready for several cycles; DUT must hold out_valid stable."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng   = np.random.default_rng(seed=55)
    iinfo = np.iinfo(_IN_DTYPE)
    a     = rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE)
    b     = rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE)

    fa = pack_matrix(a, DATA_W)
    fb = pack_matrix(b, DATA_W)
    await send_tile(dut, fa, fb)

    # Wait for out_valid without asserting out_ready.
    dut.out_ready.value = 0
    await ReadOnly()
    for _ in range(N + 4):
        await RisingEdge(dut.clk)
        await ReadOnly()

    # out_valid should now be asserted (DONE state).
    # We are already in ReadOnly phase from the last loop iteration — read directly.
    assert dut.out_valid.value == 1, "out_valid not asserted while in DONE"

    # Hold out_ready=0 for 5 more cycles; out_valid must stay high.
    for cycle in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert dut.out_valid.value == 1, f"out_valid dropped at backpressure cycle {cycle}"

    # Now accept the result.
    # We are in ReadOnly phase after the loop — capture result here (reading is OK),
    # then take one more edge to enter Active phase before driving out_ready.
    result = int(dut.flat_c.value)
    await RisingEdge(dut.clk)      # → Active phase; DUT still holds DONE/out_valid=1
    dut.out_ready.value = 1
    await RisingEdge(dut.clk)      # handshake edge
    dut.out_ready.value = 0

    got      = unpack_matrix(result, N, ACC_W)
    expected = mac_ref(a, b, data_type=DATA_TYPE)
    assert_tile_equal(got, expected, "test_backpressure result")


@cocotb.test()
async def test_pipeline(dut):
    """Three tiles processed sequentially; recv completes before next send so
    in_ready is guaranteed high (DUT in IDLE) before each new handshake."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng   = np.random.default_rng(seed=13)
    iinfo = np.iinfo(_IN_DTYPE)

    tiles = [
        (
            rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE),
            rng.integers(iinfo.min, iinfo.max + 1, size=(N, N), dtype=_IN_DTYPE),
        )
        for _ in range(3)
    ]

    for idx, (a, b) in enumerate(tiles):
        # recv_tile ends with the DONE→IDLE transition edge, so the DUT is
        # back in IDLE before this loop iteration's send_tile begins.
        got      = await run_tile(dut, a, b)
        expected = mac_ref(a, b, data_type=DATA_TYPE)
        assert_tile_equal(got, expected, f"test_pipeline tile={idx}")
