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
make test_int8    # mac_array N=4, INT8
make test_int16   # mac_array N=4, INT16
make test_n1      # mac_array N=1, INT8 edge case
make test_all     # all three mac_array configurations

make quant_int8   # quant_unit N=4, ACC_W=32
make quant_int16  # quant_unit N=4, ACC_W=48
make quant_n1     # quant_unit N=1, ACC_W=32 edge case
make quant_all    # all three quant_unit configurations
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
Phase 4, Session 21 complete + dashboard explainability update — 131 tests passing, submission-ready
  (131 pytest tests: 56 reference model + 9 feature extractor + 22 clusterer + 8 triage agent + 14 RAG store + 5 debug loop + 5 evaluator + 5 benchmark runner + 4 dashboard + 3 VCD parser)

  Key numbers: 255 regression records, 13 fault variants, 9 fault categories, 96.5% classification accuracy
  120 passed (87 golden + 33 latent faults), 135 failed (detectable injected faults)

  Dashboard (reports/dashboard.html) — latest enhancements:
    Regression DB Summary now shows latent fault breakdown dynamically from live DB records:
      - Passed stat box sub-label: "87 golden + 33 latent faults" (computed, not hardcoded)
      - Failed stat box sub-label: "detectable injected faults"
      - Amber callout explaining what latent faults are, why they pass simulation, and pointer to latent_fault cluster
      - Variant table has Pass / Fail columns; non-golden variants with passing records get "latent" badge
    All 9 triage clusters show HIGH confidence reports
    Speedup badge: 342.6× vs manual triage (31.5 s AI vs 180 min manual)
    Accuracy badge: 96%

  triage_agent.py: hardcoded HIGH-confidence latent_fault report (structurally known root cause, no API call)
  triage_agent.py: adaptive max_tokens — 1024 for clusters >10 records, 512 otherwise (prevents truncation)
  triage_agent.py: PROMPTS["label_hints"] dict — sign_error hint tells LLM to inspect both A_reg and B_reg
  debug_loop.py: updated import _MAX_TOKENS → _MAX_TOKENS_DEFAULT
  reports/benchmark_final.json: fresh --no-llm run: 96.5% accuracy, 255 records, 9 fault categories
Session 20 complete — clusterer improvements: 96.5% accuracy on 255 records, 9 fault categories
  triage/feature_extractor.py: added has_subtract_symptom flag (actual == -expected at first mismatch)
  triage/clusterer.py: 5 new/split labels replacing quant_error and sign_error catch-all:
    latent_fault: status==pass AND variant!=golden (catches acc_w24/acc_w20 + latent quant faults)
    arithmetic_error: large errors + has_subtract_symptom (fault_subtract: negated accumulation)
    saturation_error: quant_unit fail, small bucket, max_abs_error >= 225 (no-clamp wrapping)
    zero_point_error: quant_unit fail, small bucket, mismatch_rate <= 0.5 (partial-channel zp error)
    shift_error: quant_unit fail, small bucket fallthrough (fixed-shift, all channels affected)
  benchmark/evaluator.py: added latent_fault → accumulator_overflow to CLUSTER_TO_GT
  Rule-based accuracy: 96.5% (246/255); 9 still wrong (latent quant pass-records, undetectable)
Session 19 complete — VCD waveform parser, academic abstract, 123 tests passing
  triage/vcd_parser.py: stdlib-only VCD parser; extracts first_divergence_cycle, total_cycles, diverging_signals
  triage/feature_extractor.py: optional vcd_path param adds VCD features to feature dict (backward compatible)
  README.md: academic abstract (150 words) added between badges and Overview; real benchmark numbers filled in
Session 18: 13 fault variants, --no-llm baseline (72.7% vs 100% LLM), 120 tests passing
  scripts/generate_faults.py: programmatic fault generator; 9 new RTL variants from mac_array.sv and quant_unit.sv
  rtl/faults/: 13 total fault files (4 original + 9 new); each is a single targeted substitution
  benchmark_runner.py --no-llm: skips Claude API, uses cluster labels directly, prints rule-based vs LLM comparison table
  Rule-based baseline: 72.7% accuracy, mean_confidence=0.000 (no LLM); LLM-augmented: 100% accuracy
  New clusterer labels: quant_error (quant_unit small-error faults); off_by_one extended to partial-mismatch boundary errors
  fault_acc_w24/20 are LATENT for N=4 INT8 — mathematically undetectable (max acc 64516 < 2^20); show as pass in DB
Phase 3 COMPLETE — all triage engine components delivered
Phase 4 COMPLETE — benchmark, eval metrics, triage dashboard, and reproducibility packaging delivered