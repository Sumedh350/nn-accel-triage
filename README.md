# NN Accelerator Triage

> Agentic AI for Failure Triage and Root-Cause Hinting in NN Accelerator Verification

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Verilator 5.x](https://img.shields.io/badge/verilator-5.x-green)
![CocoTB 2.x](https://img.shields.io/badge/cocotb-2.x-orange)
![Tests](https://img.shields.io/badge/tests-120%20passing-brightgreen)

## Overview

This project builds a layered testbench for a small neural-network accelerator RTL design and integrates an AI triage agent that automatically clusters regression failures, generates natural-language root-cause summaries, and recommends next debug steps. The AI pipeline processes mismatch logs, trace data, and scoreboard records from CocoTB/Verilator simulations, then uses Claude to produce structured per-cluster triage reports — achieving a **342.6× speedup** over manual triage with **100% accuracy** on the benchmark dataset.

**Five Core Objectives:**
- (i) Verify functional correctness across tensor shapes, quantization, and memory behaviors
- (ii) Cluster regression failures automatically
- (iii) Generate natural-language debug summaries and likely root-cause hints
- (iv) Compare triage time against manual methods (benchmark)
- (v) Package a reproducible failure-analysis benchmark others can run

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  RTL Design (/rtl/)                                                  │
│  mac_array.sv  quant_unit.sv  mem_arbiter.sv  control_fsm.sv        │
│  /rtl/faults/  (4 injected-bug variants — frozen)                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Verilator simulation
┌───────────────────────────▼─────────────────────────────────────────┐
│  CocoTB Testbench (/tb/cocotb/)                                      │
│  test_mac_array.py  test_quant_unit.py  + reference model (/tb/ref) │
│  51 simulation tests (21 mac_array + 30 quant_unit)                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ appends JSON records
┌───────────────────────────▼─────────────────────────────────────────┐
│  regression_db.jsonl  (append-only central log)                      │
└───┬───────────────────────────────────────────────────────────────┬─┘
    │                                                               │
    ▼                                                               ▼
┌───────────────────┐    ┌──────────────────────────────────────────┐
│ Feature Extractor │───▶│ Clusterer (DBSCAN + rule-based labels)   │
│ (13 features/rec) │    │ reset_fault | overflow_fault | sign_error │
└───────────────────┘    │ off_by_one  | clean_pass                 │
                         └───────────────────┬──────────────────────┘
                                             │
                         ┌───────────────────▼──────────────────────┐
                         │  Triage Agent (Claude API)                │
                         │  + RAG Store (TF-IDF similarity)          │
                         │  + Multi-turn Debug Loop                  │
                         │  → likely_cause, confidence,              │
                         │    recommended_debug_steps                │
                         └───────────────────┬──────────────────────┘
                                             │
                    ┌────────────────────────┴──────────────────┐
                    ▼                                           ▼
        ┌───────────────────────┐               ┌──────────────────────┐
        │  Benchmark Runner     │               │  Dashboard           │
        │  342.6x speedup       │               │  reports/dashboard   │
        │  100% accuracy        │               │  .html               │
        └───────────────────────┘               └──────────────────────┘
```

## Quick Start (Docker)

Requires Docker and Docker Compose. Stages 1–3 and 5 run without an API key; stage 4 (live benchmark) requires `ANTHROPIC_API_KEY`.

```bash
# Clone the repo
git clone <repo-url>
cd nn-accel-triage

# One-command full pipeline
ANTHROPIC_API_KEY=sk-ant-... docker compose -f docker/docker-compose.yml run --rm nn-accel-triage

# Without API key (skips live benchmark, uses cached triage reports)
docker compose -f docker/docker-compose.yml run --rm nn-accel-triage
```

The `reports/` directory is volume-mounted, so `reports/dashboard.html` is accessible on the host after the run.

## Manual Setup (WSL/Linux)

**Prerequisites:**
- Python 3.10+
- Verilator 5.x (Ubuntu 22.04 apt gives 4.x — build from source; see [Verilator install docs](https://verilator.org/guide/latest/install.html))

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run reference model tests
pytest tb/reference_model/ -v

# Run triage/benchmark/dashboard unit tests
pytest triage/ benchmark/ dashboard/ -v

# Run CocoTB simulation tests (from tb/cocotb/)
make -C tb/cocotb test_all    # mac_array: INT8 N=4, INT16 N=4, INT8 N=1
make -C tb/cocotb quant_all   # quant_unit: AW=32 N=4, AW=48 N=4, AW=32 N=1
```

## Running the Benchmark

Requires `ANTHROPIC_API_KEY` (calls Claude API per failure cluster).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark/benchmark_runner.py
# Output saved to reports/benchmark_YYYYMMDD.json
```

Results are appended to `regression_db.jsonl` and a dated JSON report is saved to `reports/`.

## Opening the Dashboard

```bash
# Use cached triage reports (no API key needed)
python -m dashboard.dashboard
# Output: reports/dashboard.html

# Re-run triage via Claude API and refresh dashboard
ANTHROPIC_API_KEY=sk-ant-... python -m dashboard.dashboard --triage
```

Open `reports/dashboard.html` in any browser — it is self-contained (no JS libraries, no server needed).

## Project Structure

```
nn-accel-triage/
├── rtl/                        # Synthesizable SystemVerilog RTL
│   ├── mac_array.sv            # Parameterized NxN systolic MAC array
│   ├── quant_unit.sv           # Per-channel INT8 asymmetric quantization
│   ├── mem_arbiter.sv          # Round-robin shared SRAM arbiter
│   ├── control_fsm.sv          # Top-level sequencer (LOAD→COMPUTE→QUANT→STORE)
│   └── faults/                 # Frozen fault-injected variants (4 bugs)
├── tb/
│   ├── cocotb/                 # CocoTB 2.x simulation tests + Makefile
│   └── reference_model/        # NumPy golden reference (mac_ref.py, quant_ref.py)
├── triage/                     # AI triage engine
│   ├── feature_extractor.py    # 13 features per regression record
│   ├── clusterer.py            # DBSCAN + rule-based cluster labels
│   ├── triage_agent.py         # Claude API integration, structured reports
│   ├── rag_store.py            # TF-IDF similarity store for historical failures
│   └── debug_loop.py           # Multi-turn agentic debug conversation
├── benchmark/                  # Benchmark and evaluation
│   ├── benchmark_runner.py     # End-to-end pipeline timer + speedup calculation
│   ├── evaluator.py            # Precision/recall/F1 against ground truth
│   └── ground_truth.json       # Human-verified fault → label mapping
├── dashboard/
│   └── dashboard.py            # Static HTML dashboard generator
├── reports/                    # Generated outputs (dashboard.html, benchmark JSON)
├── docs/
│   └── mismatch_schema.json    # JSON Schema for regression_db records
├── regression_db.jsonl         # Append-only central regression log (27 entries)
├── requirements.txt
└── docker/
    ├── Dockerfile
    ├── run_pipeline.sh
    └── docker-compose.yml
```

## Key Results

| Metric | Value |
|--------|-------|
| AI pipeline time | 31.5 s |
| Manual baseline (est.) | 180 min (45 min × 4 failure clusters) |
| **Speedup vs manual** | **342.6×** |
| Overall clustering accuracy | 100% (27/27 records) |
| Mean triage confidence | High (1.000) |
| Test suite | 120 tests, 0 failures |
| Failure clusters detected | 4 (overflow, sign_error, off_by_one, reset) |

Breakdown of triage pipeline stages:

| Stage | Time |
|-------|------|
| Load + feature extraction | < 0.01 s |
| Clustering | < 0.01 s |
| Claude API triage (4 clusters) | ~31.5 s |
| Evaluation | < 0.01 s |

## Citation

```bibtex
@misc{nn-accel-triage-2026,
  title  = {Agentic AI for Failure Triage and Root-Cause Hinting in NN Accelerator Verification},
  author = {sum},
  year   = {2026},
  url    = {https://github.com/<your-github>/nn-accel-triage},
  note   = {CocoTB/Verilator testbench with Claude-powered triage agent; 342.6x speedup over manual triage}
}
```
