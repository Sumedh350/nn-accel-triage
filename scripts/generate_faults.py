"""Generate RTL fault variants from mac_array.sv and quant_unit.sv.

Each fault applies exactly one targeted string substitution to the original
source and writes the result to rtl/faults/.  A unified diff is printed to
confirm exactly one changed hunk per file.

Run from the project root:
    python scripts/generate_faults.py
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).parent.parent
_SRC_MAC = _ROOT / "rtl" / "mac_array.sv"
_SRC_QUANT = _ROOT / "rtl" / "quant_unit.sv"
_OUT_DIR = _ROOT / "rtl" / "faults"


class Fault(NamedTuple):
    src: Path
    out_name: str
    old: str
    new: str
    description: str


# ---------------------------------------------------------------------------
# Fault definitions — one targeted substitution per entry
# ---------------------------------------------------------------------------

# The COMPUTE block accumulation loop — unique because of the pe_prod RHS.
_MAC_COMPUTE_LOOP_OLD = (
    "          for (int i = 0; i < int'(N); i++)\n"
    "            for (int j = 0; j < int'(N); j++)\n"
    "              acc[i][j] <= acc[i][j] + pe_prod[i][j];"
)
_MAC_COMPUTE_LOOP_NEW = (
    "          for (int i = 0; i <= int'(N); i++)\n"
    "            for (int j = 0; j < int'(N); j++)\n"
    "              acc[i][j] <= acc[i][j] + pe_prod[i][j];"
)

# quant_unit saturation block — all three lines together are unique.
_QUANT_CLAMP_OLD = (
    "        if      (q_bias[i][j] > $signed(PROD_W'(INT8_MAX))) out_q[i][j] = 8'h7F;  // +127\n"
    "        else if (q_bias[i][j] < $signed(PROD_W'(INT8_MIN))) out_q[i][j] = 8'h80; // -128\n"
    "        else                              out_q[i][j] = $signed(q_bias[i][j][7:0]);"
)
_QUANT_CLAMP_NEW = "        out_q[i][j] = $signed(q_bias[i][j][7:0]);"

FAULTS: list[Fault] = [
    # ── mac_array faults ─────────────────────────────────────────────────────
    Fault(
        src=_SRC_MAC,
        out_name="mac_array_fault_acc_w24.sv",
        old="(DATA_TYPE == 0) ? 32 :",
        new="(DATA_TYPE == 0) ? 24 :",
        description="ACC_W INT8: 32→24 bits (narrower accumulator, earlier overflow)",
    ),
    Fault(
        src=_SRC_MAC,
        out_name="mac_array_fault_acc_w20.sv",
        old="(DATA_TYPE == 0) ? 32 :",
        new="(DATA_TYPE == 0) ? 20 :",
        description="ACC_W INT8: 32→20 bits (even narrower accumulator)",
    ),
    Fault(
        src=_SRC_MAC,
        out_name="mac_array_fault_loop_over.sv",
        old=_MAC_COMPUTE_LOOP_OLD,
        new=_MAC_COMPUTE_LOOP_NEW,
        description="COMPUTE loop bound i < N → i <= N (spatial loop runs N+1 times, OOB write)",
    ),
    Fault(
        src=_SRC_MAC,
        out_name="mac_array_fault_subtract.sv",
        old="acc[i][j] <= acc[i][j] + pe_prod[i][j];",
        new="acc[i][j] <= acc[i][j] - pe_prod[i][j];",
        description="Accumulator subtracts instead of adds partial products",
    ),
    Fault(
        src=_SRC_MAC,
        out_name="mac_array_fault_b_unsigned.sv",
        old="logic signed [DATA_W-1:0]  B_reg [N][N];",
        new="logic [DATA_W-1:0]  B_reg [N][N];",
        description="B_reg loses signed qualifier — negative weights zero-extend instead of sign-extend",
    ),
    # ── quant_unit faults ────────────────────────────────────────────────────
    Fault(
        src=_SRC_QUANT,
        out_name="quant_unit_fault_reset.sv",
        old="    if (!rst_n) begin",
        new="    if (rst_n) begin",
        description="Reset polarity inverted — unit held in reset when rst_n=1 (active)",
    ),
    Fault(
        src=_SRC_QUANT,
        out_name="quant_unit_fault_shift_fixed.sv",
        old=">>> shift[i];",
        new=">>> 1;",
        description="Arithmetic right-shift uses fixed 1 instead of per-channel shift[i]",
    ),
    Fault(
        src=_SRC_QUANT,
        out_name="quant_unit_fault_no_clamp.sv",
        old=_QUANT_CLAMP_OLD,
        new=_QUANT_CLAMP_NEW,
        description="Saturation clamp removed — output truncates (wraps) instead of saturating",
    ),
    Fault(
        src=_SRC_QUANT,
        out_name="quant_unit_fault_wrong_sign_zp.sv",
        old="PROD_W'($signed(zero_pt[i]))",
        new="PROD_W'(zero_pt[i])",
        description="zero_pt added as unsigned instead of signed — negative zero-points corrupted",
    ),
]


# ---------------------------------------------------------------------------
# Core application logic
# ---------------------------------------------------------------------------

def apply_fault(fault: Fault) -> bool:
    """Apply one fault substitution, print a diff, return True on success."""
    dst = _OUT_DIR / fault.out_name
    text = fault.src.read_text()

    count = text.count(fault.old)
    if count != 1:
        print(
            f"  ERROR: expected exactly 1 occurrence of target string, found {count}.\n"
            f"  Target: {fault.old[:60]!r}..."
        )
        return False

    new_text = text.replace(fault.old, fault.new, 1)
    dst.write_text(new_text)

    # Print diff (only changed lines, no context).
    diff = list(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=fault.src.name,
            tofile=dst.name,
            n=0,
        )
    )
    changed_lines = [l for l in diff if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
    print(f"  diff ({len(changed_lines)} lines changed):")
    for line in changed_lines:
        print(f"    {line}", end="")
    if changed_lines and not changed_lines[-1].endswith("\n"):
        print()
    return True


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = 0
    for fault in FAULTS:
        print(f"\n{'='*60}")
        print(f"Generating: {fault.out_name}")
        print(f"  Fault: {fault.description}")
        ok = apply_fault(fault)
        if not ok:
            errors += 1

    print(f"\n{'='*60}")
    if errors:
        print(f"FAILED: {errors} fault(s) could not be generated.")
        sys.exit(1)
    print(f"Done: {len(FAULTS)} fault variants written to {_OUT_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
