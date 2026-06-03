"""
NumPy reference model for quant_unit.sv.

Per-channel asymmetric INT8 quantization:

  q[i][j] = clamp(((acc[i][j] * scale[i]) >> shift[i]) + zero_pt[i], -128, 127)

where '>>' is arithmetic right-shift (rounds toward −∞), matching SV's '>>>'.
Channel axis = output row index i; each channel has its own scale, shift, zero_pt.

RTL intermediate width: PROD_W = ACC_W + SCALE_W + 1 (= 49 bits for default params).
Python int is arbitrary-precision, so no intermediate overflow even for INT16 mode
where PROD_W = 65 bits.
"""

import numpy as np


def quant_ref(
    acc: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    zero_pt: np.ndarray,
) -> np.ndarray:
    """
    Quantize one N×N accumulator tile to INT8, matching quant_unit.sv exactly.

    Formula per element (channel = row i):
        prod    = signed(acc[i][j]) * unsigned(scale[i])   [exact bigint]
        shifted = prod >> shift[i]                          [arithmetic, rounds toward −∞]
        biased  = shifted + int(zero_pt[i])
        q[i][j] = clamp(biased, -128, 127)

    Parameters
    ----------
    acc      : (N, N) signed integer array (int32 for INT8 mac, int64 for INT16 mac)
    scale    : (N,) unsigned per-channel multiplier (uint16 range, always ≥ 0)
    shift    : (N,) per-channel arithmetic right-shift amount (0..31)
    zero_pt  : (N,) per-channel signed INT8 bias

    Returns
    -------
    (N, N) np.int8 array
    """
    assert acc.ndim == 2 and acc.shape[0] == acc.shape[1], \
        "acc must be a square (N×N) 2-D array"
    N = acc.shape[0]
    assert scale.shape  == (N,), f"scale must have shape ({N},)"
    assert shift.shape  == (N,), f"shift must have shape ({N},)"
    assert zero_pt.shape == (N,), f"zero_pt must have shape ({N},)"

    result = np.empty((N, N), dtype=np.int8)

    for i in range(N):
        s  = int(scale[i])    # non-negative (unsigned multiplier)
        sh = int(shift[i])    # right-shift amount 0..31
        zp = int(zero_pt[i])  # signed bias

        # Use Python int (arbitrary precision) for exact hardware match.
        # The inner list comprehension vectorises over j while keeping bigint safety.
        row = [max(-128, min(127, (int(v) * s >> sh) + zp)) for v in acc[i]]
        result[i] = row

    return result
