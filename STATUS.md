# Project Status

## Current Phase: Phase 1 — DUT Modeling & Testbench Setup
## Current Step: Step 5 — NumPy reference model (RTL complete)

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
- `rtl/quant_unit.sv` written and lint-clean (Verilator 4.038, exit 0, zero warnings)
  - Per-channel INT8 asymmetric quantization: q[i][j] = clamp(((acc[i][j]*scale[i])>>>shift[i])+zero_pt[i], -128, 127)
  - Channel axis = output row (index i); per-channel scale (16-bit unsigned), shift (5-bit), zero_pt (INT8 signed)
  - Combinational always_comb quantization path; PROD_W = ACC_W+SCALE_W+1 = 49-bit intermediate
  - Synchronous active-low reset, AXI-style valid/ready on input and output
  - FSM: IDLE → DONE (single-cycle latency, 1 tile per 2 cycles throughput)
  - Elaboration-time $fatal guards for N, SCALE_W, SHIFT_W
- `rtl/mem_arbiter.sv` written and lint-clean (Verilator 4.038, exit 0, zero warnings)
  - Shared SRAM arbiter for weight (A) and activation (B) memory requestors
  - Round-robin arbitration via last_grant register; alternates each cycle when both contend
  - Bank-conflict detection: addr[BANK_BITS-1:0] compared across both requestors
  - Stall injection: 1-cycle STALL penalty on conflict; bank_conflict output asserted
  - Combinational SRAM read model (sram_rdata valid same cycle as sram_ce/sram_addr)
  - AXI-style req_x_ready/req_x_rvld handshake; req_x_rdata from sram_rdata
  - Elaboration-time $fatal for ADDR_W<1, DATA_W<1, NUM_BANKS<1
  - FSM: IDLE ↔ STALL (2-state, 1-bit enum)
- `rtl/control_fsm.sv` written and lint-clean (Verilator 4.038, exit 0, zero warnings)
  - Top-level sequencer: instantiates mac_array, quant_unit, mem_arbiter
  - FSM: IDLE → LOAD → COMPUTE → QUANTIZE → STORE → DONE
  - LOAD: fetches N×N weights (req_a) and N×N activations (req_b) concurrently
  - COMPUTE: drives mac_array; mac→quant handshake collapses into one cycle
  - QUANTIZE: latches INT8 quant_result tile from quant_unit in one cycle
  - STORE: writes N×N INT8 results back to SRAM via req_a
  - start/done pulse handshake; synchronous active-low reset
  - Elaboration-time $fatal guards for N, DATA_TYPE, ADDR_W, NUM_BANKS
- Verilator 4.038 installed in WSL Ubuntu-22.04

## Next Session Goal
- Write `tb/reference_model/mac_ref.py`: pure NumPy NxN matrix multiply reference
  - Must match mac_array.sv semantics exactly (signed INT8/INT16, ACC_W accumulation)
- Write `tb/reference_model/quant_ref.py`: NumPy reference for quant_unit
  - Match quant_unit.sv formula exactly (multiply, arithmetic right-shift, clamp)
- Write `tb/cocotb/test_mac_array.py`: CocoTB testbench for mac_array
  - Drive valid/ready handshake, compare DUT output vs reference model
  - Test cases: INT8 (N=4), INT16 (N=4), edge cases (N=1, all-zero, max values)
- Append first regression entries to regression_db.jsonl
- Create `/docs/mismatch_schema.json` (scoreboard mismatch record format)

## Blockers / Open Questions
- Verilator v5+ not yet installed (Ubuntu 22.04 apt has v4.038 only)
  - v4.038 passes lint fine; upgrade to v5 before simulation if needed