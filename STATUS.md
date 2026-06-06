# Project Status

## Current Phase: Phase 1 — DUT Modeling & Testbench Setup
## Current Step: Step 8 — mismatch_schema.json, regression_db.jsonl first entries, fault injection

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

- `tb/reference_model/mac_ref.py` written and test-clean (36 pytest tests, 0 failures)
  - Pure NumPy NxN matrix multiply: INT8→int32, INT16→int64(48-bit), FP16→zeros
  - Matches mac_array.sv exactly: sign-extension, 32/48-bit accumulator wrap
- `tb/reference_model/quant_ref.py` written and test-clean
  - Per-channel asymmetric INT8: (acc*scale)>>>shift + zero_pt, clamped to [-128,127]
  - Arithmetic right-shift via Python int (arbitrary-precision; no int64 overflow)
  - Matches quant_unit.sv formula and PROD_W=ACC_W+SCALE_W+1 intermediate exactly
- `tb/reference_model/test_mac_ref.py` — 14 tests (sign, shape, wrap, FP16 stub)
- `tb/reference_model/test_quant_ref.py` — 22 tests (clamp, arith-shift, per-channel)
- pytest installed; `~/.local/bin` added to PATH in `~/.bashrc`

- `tb/cocotb/mac_array_wrap.sv` — lint-clean SV wrapper; flattens unpacked 2-D ports to
  packed flat buses (row-major: element [i][j] at [(i*N+j)*W +: W])
- `tb/cocotb/test_mac_array.py` — 7 CocoTB 2.0.1 tests × 3 configs = 21 tests, 0 failures
  - Configs: INT8/N=4, INT16/N=4, INT8/N=1
  - Tests: zeros, random (5 iters), max_values, identity, reset, backpressure, pipeline
  - All expected values from mac_ref.py — no hardcoded results
- `tb/cocotb/Makefile` — restructured with DUT variable (DUT=mac_array|quant_unit);
  per-config sim_build dirs; mac targets: test_int8/test_int16/test_n1/test_all;
  quant targets: quant_int8/quant_int16/quant_all
- CocoTB 2.x phase discipline: never drive or await ReadOnly() while in ReadOnly phase;
  Verilator bakes params into binary — use separate SIM_BUILD dirs per config
- Verilator 5.049 confirmed working for simulation (upgraded from 4.038)

- `tb/cocotb/quant_unit_wrap.sv` — lint-clean SV wrapper; flattens unpacked ports to
  packed flat buses (row-major 2D, channel-index-major 1D for scale/shift/zero_pt)
- `tb/cocotb/test_quant_unit.py` — 10 CocoTB 2.0.1 tests × 3 configs = 30 tests, 0 failures
  - Configs: ACC_W=32/N=4 (INT8-sourced), ACC_W=48/N=4 (INT16-sourced), ACC_W=32/N=1 (edge case)
  - Tests: passthrough, random (5 iters), clamp_high, clamp_low, arith_rshift (shift sweep
    0/1/2/4/8/15/31 — includes ACC_W-1 boundary), per_channel_independence, scale_zero,
    reset, backpressure, back_to_back
  - All expected values from quant_ref.py; acc values bounded to ACC_W-bit signed range
  - scale_zero: RTL-defined behavior (acc*0=0 → output = zero_pt); verifies per-row semantics
- `tb/cocotb/Makefile` updated with quant_int8/quant_int16/quant_n1/quant_all targets

## Step 7: COMPLETE (including post-review fixes)
30 CocoTB tests passing (10×N=4/AW=32, 10×N=4/AW=48, 10×N=1/AW=32). 21 mac_array tests unchanged.

## Step 8: COMPLETE
- `tb/reference_model/test_generator.py` — TestGenerator class (gen_mac_vectors, gen_quant_vectors)
  - INT8/INT16 input bounds; acc bounded to ACC_W-bit signed range (CLAUDE.md invariant)
  - 20 new pytest tests; total suite: 56 tests, 0 failures
- `rtl/faults/` (4 files, each a one-line change from mac_array.sv):
  - mac_array_fault_acc_overflow.sv  — ACC_W INT8: 32→16 (accumulator too narrow)
  - mac_array_fault_wrong_sign.sv    — A_reg: signed→unsigned (zero-extends instead of sign-extends)
  - mac_array_fault_off_by_one.sv    — termination: N-1→N-2 (misses last outer product)
  - mac_array_fault_reset.sv         — polarity: !rst_n→rst_n (active-high instead of active-low)
- `docs/mismatch_schema.json` — JSON Schema (draft-2020-12) validating regression_db records
- `scripts/gen_initial_regression.py` — bootstrap script using software fault models
- `regression_db.jsonl` — 27 entries: 15 golden (all pass), 12 fault (all fail)
  - Seeds [1,9,13] chosen to guarantee int16 overflow for fault_acc_overflow detection

## Phase 2 Goals
- Extend CocoTB testbench to append live entries to regression_db.jsonl during simulation
- Add Makefile targets for fault-variant Verilator builds (per-fault sim_build dirs)
- Implement failure feature extractor (triage/feature_extractor.py)
- Implement failure clusterer (triage/clusterer.py)

## Blockers / Open Questions
- None