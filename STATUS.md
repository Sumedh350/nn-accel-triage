# Project Status

## Current Phase: Phase 1 — DUT Modeling & Testbench Setup
## Current Step: Step 2 — CocoTB testbench + NumPy reference model

## Completed
- GitHub repo created
- Folder structure initialized
- CLAUDE.md files written for root, rtl/, tb/, triage/
- `rtl/mac_array.sv` written and lint-clean (Verilator 4.038, exit 0, zero warnings)
  - Parameterized NxN systolic-style MAC array (default N=4)
  - DATA_TYPE: 0=INT8 (8-bit in, 32-bit acc), 1=INT16 (16-bit in, 48-bit acc), 2=FP16 stub
  - Synchronous active-low reset, AXI-style valid/ready on input and output
  - FSM: IDLE → COMPUTE (N cycles) → DONE
  - Elaboration-time $fatal/$warning guards for bad parameters

## Next Session Goal
- Write `tb/reference_model/mac_ref.py`: pure NumPy NxN matrix multiply reference
  - Must match mac_array.sv semantics exactly (signed INT8/INT16, ACC_W accumulation)
- Write `tb/cocotb/test_mac_array.py`: CocoTB testbench
  - Drive valid/ready handshake, compare DUT output vs reference model
  - Test cases: INT8 default (N=4), INT16 (N=4), edge cases (N=1, all-zero, max values)
- Append first regression entries to regression_db.jsonl
- Create `/docs/mismatch_schema.json` (scoreboard mismatch record format)

## Blockers / Open Questions
- Verilator v5+ not yet installed (Ubuntu 22.04 apt has v4.038 only)
  - v4.038 passes lint fine; upgrade to v5 before simulation if needed
  - Install path: WSL Ubuntu-22.04, requires interactive sudo