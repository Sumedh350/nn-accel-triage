"""
NumPy reference model for mac_array.sv.

Computes C = A × B for one N×N signed-integer tile.

DATA_TYPE semantics (mirrors mac_array.sv parameters):
  0 = INT8  : int8  inputs, 32-bit signed accumulator  → int32 output
  1 = INT16 : int16 inputs, 48-bit signed accumulator → int64 output (48-bit wrapped)
  2 = FP16  : stubbed; always returns a zero tile       → int32 output
"""

import numpy as np

# 48-bit signed arithmetic constants (for INT16 / ACC_W=48 mode)
_INT48_SIGN_BIT = np.int64(1 << 47)
_INT48_MASK     = np.int64((1 << 48) - 1)
_INT48_WRAP     = np.int64(1 << 48)


def _truncate_to_int48(x: np.ndarray) -> np.ndarray:
    """Wrap int64 array to 48-bit signed 2's complement range (matches hardware ACC_W=48)."""
    x = x & _INT48_MASK
    return np.where(x & _INT48_SIGN_BIT != 0, x - _INT48_WRAP, x)


def mac_ref(
    a: np.ndarray,
    b: np.ndarray,
    data_type: int = 0,
) -> np.ndarray:
    """
    Compute C = A × B, matching mac_array.sv semantics exactly.

    The RTL sign-extends each operand to ACC_W before multiplying and accumulates
    N steps of pe_prod[i][j] += A[i][k] * B[k][j].  This is standard matrix
    multiply; NumPy matmul on int64 operands reproduces it without overflow for
    any practical N (wrap only occurs at N > ~131 000 for INT8, >> N > ~8 192 for INT16).

    Parameters
    ----------
    a, b       : (N, N) signed integer arrays.
                 Typically int8 for DATA_TYPE=0, int16 for DATA_TYPE=1.
    data_type  : 0=INT8 (ACC_W=32), 1=INT16 (ACC_W=48), 2=FP16 stub (zeros).

    Returns
    -------
    np.ndarray of shape (N, N):
        dtype int32  for DATA_TYPE 0 and 2.
        dtype int64  for DATA_TYPE 1 (value is in 48-bit signed range).
    """
    assert a.ndim == 2 and a.shape == b.shape and a.shape[0] == a.shape[1], \
        "a and b must be matching square (N×N) 2-D arrays"
    assert data_type in (0, 1, 2), f"data_type must be 0, 1, or 2 — got {data_type}"

    if data_type == 2:
        return np.zeros(a.shape, dtype=np.int32)

    # Promote to int64 before matmul to prevent intermediate overflow.
    c64: np.ndarray = a.astype(np.int64) @ b.astype(np.int64)

    if data_type == 0:
        # Truncate to 32-bit signed — matches hardware ACC_W=32 wrap-around.
        return c64.astype(np.int32)
    else:  # data_type == 1
        # Truncate to 48-bit signed — matches hardware ACC_W=48 wrap-around.
        return _truncate_to_int48(c64)
