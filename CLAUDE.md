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
```

## DO NOT MODIFY
- /rtl/faults/     → frozen fault variants, never edit after Stage 1
- /benchmark/failure_dataset/*.jsonl → ground truth labels, human-verified

## Current Phase
Phase 1, Step 5 complete — NumPy reference models done (mac_ref.py, quant_ref.py, 36 tests pass)