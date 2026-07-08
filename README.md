# NN Accelerator Triage

> Agentic AI for Failure Triage and Root-Cause Hinting in NN Accelerator Verification

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Verilator 5.x](https://img.shields.io/badge/verilator-5.x-green)
![CocoTB 2.x](https://img.shields.io/badge/cocotb-2.x-orange)
![Tests](https://img.shields.io/badge/tests-120%20passing-brightgreen)

## Abstract

We present NN-Accel-Triage, an agentic AI system for automated failure triage in neural-network accelerator verification. Given a regression database of RTL simulation mismatches, the system clusters failures by symptom, generates structured root-cause reports via a large language model with retrieval-augmented generation and multi-turn reasoning, and benchmarks triage quality against ground truth labels. On an expanded benchmark of 13 fault variants across 9 fault categories and 255 records in a parameterized MAC array and quantization unit, our approach achieves 71.8% clustering accuracy with mean LLM confidence 0.800, reducing triage time from an estimated 225 minutes to 46 seconds (293× speedup). On the original 4-fault, 27-record benchmark, accuracy is 100% with mean confidence 1.000 (342× speedup). A rule-based baseline achieves 71.8% accuracy without LLM augmentation; the LLM adds structured root-cause explanations and confidence scores. We release the benchmark dataset, RTL fault variants, triage pipeline, and dashboard as open-source artifacts.

## Overview

This project builds a layered testbench for a small neural-network accelerator RTL design and integrates an AI triage agent that automatically clusters regression failures, generates natural-language root-cause summaries, and recommends next debug steps. The AI pipeline processes mismatch logs, trace data, and scoreboard records from CocoTB/Verilator simulations, then uses Claude to produce structured per-cluster triage reports — achieving a **293× speedup** over manual triage with **71.8% accuracy** on the expanded 13-fault, 255-record benchmark (100% on the original 4-fault benchmark).

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
        │  293x speedup         │               │  reports/dashboard   │
        │  71.8% accuracy       │               │  .html               │
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
│   └── faults/                 # Frozen fault-injected variants (13 bugs)
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
├── regression_db.jsonl         # Append-only central regression log (255+ entries)
├── requirements.txt
└── docker/
    ├── Dockerfile
    ├── run_pipeline.sh
    └── docker-compose.yml
```

## Key Results

### Expanded benchmark (13 faults, 255 records)

| Metric | LLM-augmented | Rule-based only |
|--------|--------------|-----------------|
| Overall accuracy | **71.8%** (183/255) | 71.8% (183/255) |
| Mean confidence | 0.800 | 0.000 |
| AI pipeline time | 46 s | 0.1 s |
| Manual baseline (est.) | 225 min (45 min × 5 clusters) | 225 min |
| **Speedup vs manual** | **293×** | **63,000×** |
| Failure clusters | 5 (overflow, sign_error, off_by_one, reset, quant_error) | 5 |

### Original benchmark (4 faults, 27 records)

| Metric | Value |
|--------|-------|
| AI pipeline time | 31.5 s |
| Manual baseline (est.) | 180 min (45 min × 4 failure clusters) |
| **Speedup vs manual** | **342.6×** |
| Overall clustering accuracy | **100%** (27/27 records) |
| Mean triage confidence | High (1.000) |
| Test suite | 123 tests, 0 failures |

Breakdown of triage pipeline stages (expanded benchmark):

| Stage | Time |
|-------|------|
| Load + feature extraction | < 0.01 s |
| Clustering | < 0.1 s |
| Claude API triage (5 clusters) | ~46 s |
| Evaluation | < 0.01 s |

## Citation

```bibtex
@misc{nn-accel-triage-2026,
  title  = {Agentic AI for Failure Triage and Root-Cause Hinting in NN Accelerator Verification},
  author = {sum},
  year   = {2026},
  url    = {https://github.com/<your-github>/nn-accel-triage},
  note   = {CocoTB/Verilator testbench with Claude-powered triage agent; 293x speedup (expanded 13-fault benchmark), 342.6x on original 4-fault benchmark}
}
```
