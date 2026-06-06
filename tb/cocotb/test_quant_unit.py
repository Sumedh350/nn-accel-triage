"""
CocoTB 2.0.x testbench for quant_unit_wrap / quant_unit.

Configuration is passed via environment variables set by the Makefile:
  QUANT_N      : tile dimension (default 4)
  QUANT_ACC_W  : accumulator width in bits (default 32; 48 for INT16-sourced accs)

All expected values are computed by quant_ref.py — no hardcoded INT8 results.
"""

import os
import sys
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

_REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_model")
if _REF_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_REF_DIR))

from quant_ref import quant_ref  # noqa: E402

# ── Compile-time configuration ─────────────────────────────────────────────────
N       = int(os.environ.get("QUANT_N",    "4"))
ACC_W   = int(os.environ.get("QUANT_ACC_W", "32"))
CLK_NS  = 10

# Accumulator dtype wide enough for ACC_W bits; range clamped to ACC_W-bit signed.
_ACC_DTYPE = np.int32 if ACC_W <= 32 else np.int64
_ACC_MIN   = -(1 << (ACC_W - 1))
_ACC_MAX   = (1 << (ACC_W - 1)) - 1

# Fixed quant parameter widths (match quant_unit.sv defaults)
SCALE_W = 16
SHIFT_W = 5


# ── Packing helpers ────────────────────────────────────────────────────────────

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
    return out.astype(np.int8)


def pack_vector(v: np.ndarray, elem_bits: int) -> int:
    """1-D channel pack: element [i] → bit offset i*elem_bits."""
    mask = (1 << elem_bits) - 1
    val  = 0
    for i, x in enumerate(v):
        val |= (int(x) & mask) << (i * elem_bits)
    return val


def assert_tile_equal(got: np.ndarray, expected: np.ndarray, label: str = ""):
    got_i = got.astype(np.int64)
    exp_i = expected.astype(np.int64)
    if not np.array_equal(got_i, exp_i):
        diff = got_i - exp_i
        raise AssertionError(
            f"{label}\n  expected:\n{exp_i}\n  got:\n{got_i}\n  diff:\n{diff}"
        )


# ── Coroutines ─────────────────────────────────────────────────────────────────

async def reset_dut(dut):
    """5-cycle synchronous active-low reset; leave DUT in clean IDLE state."""
    dut.rst_n.value      = 0
    dut.in_valid.value   = 0
    dut.out_ready.value  = 0
    dut.flat_acc.value   = 0
    dut.flat_scale.value = 0
    dut.flat_shift.value = 0
    dut.flat_zero_pt.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_quant_tile(
    dut,
    flat_acc: int,
    flat_scale: int,
    flat_shift: int,
    flat_zero_pt: int,
):
    """Drive quantization inputs and complete the in_valid/in_ready handshake."""
    dut.flat_acc.value     = flat_acc
    dut.flat_scale.value   = flat_scale
    dut.flat_shift.value   = flat_shift
    dut.flat_zero_pt.value = flat_zero_pt
    dut.in_valid.value     = 1
    await ReadOnly()
    while not dut.in_ready.value:
        await RisingEdge(dut.clk)
        await ReadOnly()
    # in_ready is 1 — next rising edge completes the handshake.
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0


async def recv_quant_tile(dut) -> int:
    """Assert out_ready, wait for out_valid, complete handshake, return flat_q."""
    dut.out_ready.value = 1
    await ReadOnly()
    while not dut.out_valid.value:
        await RisingEdge(dut.clk)
        await ReadOnly()
    result = int(dut.flat_q.value)
    await RisingEdge(dut.clk)
    dut.out_ready.value = 0
    return result


async def run_quant_tile(
    dut,
    acc: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    zero_pt: np.ndarray,
) -> np.ndarray:
    """Pack inputs, run one tile through DUT, unpack and return INT8 result."""
    fa  = pack_matrix(acc,  ACC_W)
    fs  = pack_vector(scale,   SCALE_W)
    fsh = pack_vector(shift,   SHIFT_W)
    fzp = pack_vector(zero_pt, 8)
    await send_quant_tile(dut, fa, fs, fsh, fzp)
    raw = await recv_quant_tile(dut)
    return unpack_matrix(raw, N, 8)


# ── Helper: default no-op quantization params ─────────────────────────────────

def _passthrough_params():
    """scale=1, shift=0, zero_pt=0 → output equals acc (after INT8 clamp)."""
    return (
        np.ones(N,  dtype=np.uint16),
        np.zeros(N, dtype=np.uint8),
        np.zeros(N, dtype=np.int8),
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_passthrough(dut):
    """scale=1, shift=0, zero_pt=0: output must equal clamp(acc, -128, 127)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=1)
    acc = rng.integers(
        _ACC_MIN, _ACC_MAX + 1,
        size=(N, N), dtype=_ACC_DTYPE,
    )
    scale, shift, zero_pt = _passthrough_params()
    got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_passthrough")


@cocotb.test()
async def test_random(dut):
    """Five random tiles with random per-channel params must match quant_ref."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=42)
    for iteration in range(5):
        acc     = rng.integers(
            _ACC_MIN, _ACC_MAX + 1,
            size=(N, N), dtype=_ACC_DTYPE,
        )
        scale   = rng.integers(0, 2**SCALE_W,      size=N, dtype=np.uint16)
        shift   = rng.integers(0, 2**SHIFT_W,       size=N, dtype=np.uint8)
        zero_pt = rng.integers(-128, 128,            size=N, dtype=np.int8)
        got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
        expected = quant_ref(acc, scale, shift, zero_pt)
        assert_tile_equal(got, expected, f"test_random iter={iteration}")


@cocotb.test()
async def test_clamp_high(dut):
    """Positive accumulators × large scale must saturate at +127."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    # Fill acc with large positive values; scale large, shift=0 → product >> INT8_MAX.
    acc     = np.full((N, N), _ACC_MAX, dtype=_ACC_DTYPE)
    scale   = np.full(N, 2**SCALE_W - 1, dtype=np.uint16)
    shift   = np.zeros(N, dtype=np.uint8)
    zero_pt = np.zeros(N, dtype=np.int8)
    got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_clamp_high")
    assert np.all(got == 127), f"expected all +127, got:\n{got}"


@cocotb.test()
async def test_clamp_low(dut):
    """Negative accumulators × large scale must saturate at -128."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    acc     = np.full((N, N), _ACC_MIN, dtype=_ACC_DTYPE)
    scale   = np.full(N, 2**SCALE_W - 1, dtype=np.uint16)
    shift   = np.zeros(N, dtype=np.uint8)
    zero_pt = np.zeros(N, dtype=np.int8)
    got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_clamp_low")
    assert np.all(got == -128), f"expected all -128, got:\n{got}"


@cocotb.test()
async def test_arith_rshift(dut):
    """Sweep shift values; verify arithmetic (floor) right-shift for negatives."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    # Use a negative accumulator so we can distinguish arithmetic vs logical shift.
    acc     = np.full((N, N), -4, dtype=_ACC_DTYPE)
    scale   = np.ones(N, dtype=np.uint16)
    zero_pt = np.zeros(N, dtype=np.int8)

    for sh_val in [0, 1, 2, 4, 8, 15, 31]:
        shift = np.full(N, sh_val, dtype=np.uint8)
        got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
        expected = quant_ref(acc, scale, shift, zero_pt)
        assert_tile_equal(got, expected, f"test_arith_rshift shift={sh_val}")


@cocotb.test()
async def test_per_channel_independence(dut):
    """Each row uses distinct scale/shift/zero_pt; rows must not bleed into each other."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=77)
    acc = rng.integers(
        _ACC_MIN, _ACC_MAX + 1,
        size=(N, N), dtype=_ACC_DTYPE,
    )
    # Make every channel parameter unique so a cross-channel bug would show.
    scale   = np.array([i + 1          for i in range(N)], dtype=np.uint16)
    shift   = np.array([i % SHIFT_W    for i in range(N)], dtype=np.uint8)
    zero_pt = np.array([i - N // 2     for i in range(N)], dtype=np.int8)

    got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_per_channel_independence")


@cocotb.test()
async def test_scale_zero(dut):
    """scale=0: output must equal zero_pt broadcast across each row (RTL-defined behavior).

    With scale=0: prod = acc*0 = 0; shifted = 0; biased = zero_pt → q = clamp(zero_pt).
    Since zero_pt is INT8, the clamp is a no-op and output equals zero_pt exactly.
    """
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=17)
    acc     = rng.integers(_ACC_MIN, _ACC_MAX + 1, size=(N, N), dtype=_ACC_DTYPE)
    scale   = np.zeros(N, dtype=np.uint16)
    shift   = rng.integers(0, 2**SHIFT_W, size=N, dtype=np.uint8)
    zero_pt = rng.integers(-128, 128, size=N, dtype=np.int8)
    got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_scale_zero")
    for i in range(N):
        assert np.all(got[i] == zero_pt[i]), \
            f"test_scale_zero row {i}: expected all {zero_pt[i]}, got {got[i]}"


@cocotb.test()
async def test_reset(dut):
    """Assert reset while DUT is in DONE state; verify clean recovery."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=99)
    acc     = rng.integers(
        _ACC_MIN, _ACC_MAX + 1,
        size=(N, N), dtype=_ACC_DTYPE,
    )
    scale, shift, zero_pt = _passthrough_params()

    # Start a transaction (IDLE → DONE).
    fa  = pack_matrix(acc, ACC_W)
    fs  = pack_vector(scale,   SCALE_W)
    fsh = pack_vector(shift,   SHIFT_W)
    fzp = pack_vector(zero_pt, 8)
    dut.flat_acc.value     = fa
    dut.flat_scale.value   = fs
    dut.flat_shift.value   = fsh
    dut.flat_zero_pt.value = fzp
    dut.in_valid.value     = 1
    await RisingEdge(dut.clk)   # IDLE→DONE, out_valid=1 now
    dut.in_valid.value = 0

    # Assert reset while in DONE (out_valid=1, out_ready still 0).
    dut.rst_n.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # DUT should be back in IDLE.
    assert dut.in_ready.value  == 1, "in_ready not 1 after reset"
    assert dut.out_valid.value == 0, "out_valid not 0 after reset"

    # Full transaction must succeed after recovery.
    got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_reset post-recovery")


@cocotb.test()
async def test_backpressure(dut):
    """Withhold out_ready N+4 cycles; out_valid must remain stable throughout."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=55)
    acc     = rng.integers(
        _ACC_MIN, _ACC_MAX + 1,
        size=(N, N), dtype=_ACC_DTYPE,
    )
    scale   = rng.integers(1, 2**SCALE_W, size=N, dtype=np.uint16)
    shift   = rng.integers(0, 2**SHIFT_W, size=N, dtype=np.uint8)
    zero_pt = rng.integers(-128, 128,      size=N, dtype=np.int8)

    fa  = pack_matrix(acc, ACC_W)
    fs  = pack_vector(scale,   SCALE_W)
    fsh = pack_vector(shift,   SHIFT_W)
    fzp = pack_vector(zero_pt, 8)
    await send_quant_tile(dut, fa, fs, fsh, fzp)

    # Hold out_ready=0 and wait.
    dut.out_ready.value = 0
    await ReadOnly()
    for _ in range(N + 4):
        await RisingEdge(dut.clk)
        await ReadOnly()

    assert dut.out_valid.value == 1, "out_valid not asserted in DONE state"

    # Verify it stays stable under continued backpressure.
    for cycle in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert dut.out_valid.value == 1, \
            f"out_valid dropped at backpressure cycle {cycle}"

    # Capture result while still in ReadOnly, then move to Active before driving.
    result = int(dut.flat_q.value)
    await RisingEdge(dut.clk)       # → Active phase
    dut.out_ready.value = 1
    await RisingEdge(dut.clk)       # handshake edge
    dut.out_ready.value = 0

    got      = unpack_matrix(result, N, 8)
    expected = quant_ref(acc, scale, shift, zero_pt)
    assert_tile_equal(got, expected, "test_backpressure result")


@cocotb.test()
async def test_back_to_back(dut):
    """Three sequential tiles; each recv completes before the next send begins."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(seed=13)
    tiles = []
    for _ in range(3):
        acc     = rng.integers(
            _ACC_MIN, _ACC_MAX + 1,
            size=(N, N), dtype=_ACC_DTYPE,
        )
        scale   = rng.integers(1, 2**SCALE_W, size=N, dtype=np.uint16)
        shift   = rng.integers(0, 2**SHIFT_W, size=N, dtype=np.uint8)
        zero_pt = rng.integers(-128, 128,      size=N, dtype=np.int8)
        tiles.append((acc, scale, shift, zero_pt))

    for idx, (acc, scale, shift, zero_pt) in enumerate(tiles):
        got      = await run_quant_tile(dut, acc, scale, shift, zero_pt)
        expected = quant_ref(acc, scale, shift, zero_pt)
        assert_tile_equal(got, expected, f"test_back_to_back tile={idx}")
