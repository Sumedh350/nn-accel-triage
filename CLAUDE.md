# NN Accelerator Triage Project
## Project Mission
Agentic AI for Failure Triage and Root-Cause Hinting in NN Accelerator Verification.

Build a layered testbench for a small neural-network accelerator and integrate 
an AI assistant that groups failures, summarizes likely causes, and recommends 
next debug steps using logs, traces, scoreboard mismatches, and commit metadata.

## Five Core Objectives (must all be satisfied)
(i)   Verify functional correctness across tensor shapes, quantization, and memory behaviors
(ii)  Cluster regression failures automatically
(iii) Generate natural-language debug summaries and likely root-cause hints
(iv)  Compare triage time against manual methods (benchmark)
(v)   Package a reproducible failure-analysis benchmark others can run

## What This Project Is
A layered testbench for a small neural-network accelerator RTL design,
with an AI triage agent that clusters regression failures, summarizes
root causes, and recommends debug steps.

## Project Structure
- /rtl/          → synthesizable RTL modules (SystemVerilog)
- /rtl/faults/   → RTL variants with deliberately injected bugs (FROZEN after creation)
- /tb/cocotb/    → CocoTB-based testbench drivers, monitors, scoreboard
- /tb/reference_model/ → NumPy/PyTorch golden reference model
- /triage/       → failure feature extractor, clusterer, AI triage agent
- /benchmark/    → ground truth failure dataset and eval metrics
- /docs/         → schema files, architecture notes
- regression_db.jsonl → central log of all test runs (NEVER overwrite, append only)

## Key Conventions
- Python 3.11+, use type hints everywhere
- SystemVerilog for all RTL
- Simulation via Verilator
- All test runs append one JSON line to regression_db.jsonl
- Mismatch records follow the schema in /docs/mismatch_schema.json (once created)
- Commit after every working milestone, never commit broken code

## Build & Test Commands
```bash
# Reference model unit tests
pytest tb/reference_model/ -v

# CocoTB / Verilator simulation (from tb/cocotb/)
make test_int8    # N=4, INT8
make test_int16   # N=4, INT16
make test_n1      # N=1, INT8 edge case
make test_all     # all three configurations
```

## CocoTB 2.x Phase Discipline
CocoTB 2.x enforces strict simulator phase rules — violations raise RuntimeError:
- **Never `await ReadOnly()` while already in ReadOnly phase.** After a loop that
  ends with `await ReadOnly()`, you are still in that phase; a second `await ReadOnly()`
  is illegal.
- **Never drive a signal (`.value = x`) during ReadOnly phase.** Always return to
  Active phase (via `await RisingEdge(clk)`) before driving any signal.
- **Coroutines that internally call `await ReadOnly()` (e.g. `send_tile`) must be
  called from Active phase**, not from within a ReadOnly context.
- **Verilator bakes parameters into the compiled C++ binary.** Each distinct
  (N, DATA_TYPE) configuration needs its own `sim_build_*` directory — controlled
  via `SIM_BUILD` in the Makefile — otherwise stale binaries silently use wrong port widths.

## Test Vector Generation — Accumulator Range
When generating random accumulator values, **always bound them to the ACC_W-bit
signed range, not the NumPy dtype range**:
```python
_ACC_MIN = -(1 << (ACC_W - 1))
_ACC_MAX =  (1 << (ACC_W - 1)) - 1
acc = rng.integers(_ACC_MIN, _ACC_MAX + 1, size=(N, N), dtype=_ACC_DTYPE)
```
The packing mask `int(v) & ((1 << ACC_W) - 1)` silently truncates upper bits.
A 64-bit value that fits `np.int64` but not a 48-bit accumulator will have the
correct sign in the reference model (which sees the full Python int) but the
**wrong sign in the DUT** (which sees only the lower ACC_W bits). The mismatch
is seed-dependent and easy to miss — some seeds produce values that happen to
sign-extend cleanly; others don't.

## DO NOT MODIFY
- /rtl/faults/     → frozen fault variants, never edit after Stage 1
- /benchmark/failure_dataset/*.jsonl → ground truth labels, human-verified

## Current Phase
Phase 1, Step 7 complete — CocoTB testbench for quant_unit done (18 tests pass: 9×ACC_W=32, 9×ACC_W=48)
Phase 1, Step 8 next — mismatch_schema.json, regression_db.jsonl first entries, fault injection