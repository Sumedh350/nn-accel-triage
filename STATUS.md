# Project Status

## Current Phase: Phase 3 — Triage Agent & Live Integration
## Current Step: Session 12 complete — RAG store for historical failures (101 tests passing)

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

## Session 9: COMPLETE
- `triage/feature_extractor.py` — extracts 13 features per regression_db record
  - Identity pass-through: run_id, dut, variant, test_name, status
  - Flattened config: config_n, config_data_type (None for quant_unit), config_acc_w
  - Numeric: mismatch_rate (count/total), max_abs_error (0 for pass)
  - Bucket: error_magnitude_bucket — "none" | "small" (<256) | "medium" (<32768) | "large" (≥32768)
  - Flags: has_reset_symptom (first_mismatch.actual == 0), has_overflow_symptom (power-of-2 error ≥ 32768)
  - Public API: extract_features(record) and load_and_extract(db_path)
- `triage/test_feature_extractor.py` — 8 pytest tests, all passing
  - Synthetic unit tests: pass record, small/medium/large error, reset symptom, overflow symptom, bucket boundaries
  - Integration test: validates all 27 regression_db.jsonl records (15 pass, 12 fail counts verified)
- Total test suite: 64 tests (56 reference model + 8 feature extractor), 0 failures

## Session 10: COMPLETE (including post-review fixes)
- `triage/clusterer.py` — assigns cluster_label + outlier flag to each feature dict
  - assign_rule_label(feat): priority-ordered rules → reset_fault | overflow_fault | sign_error | off_by_one | clean_pass | uncategorized
  - _to_numeric_vector(feat): 5-element normalized vector (mismatch_rate, max_abs_error/65536, bucket/3, reset, overflow)
  - cluster(features, eps=0.5, min_samples=2): returns enriched dicts; originals unmodified; DBSCAN via scikit-learn
  - All 27 regression_db records label without "uncategorized": 15 clean_pass, 3 each for reset/overflow/off_by_one/sign_error
- `triage/test_clusterer.py` — 15 pytest tests, all passing
  - One test per label (6), two priority-order tests (reset>overflow, overflow>off_by_one)
  - cluster() field/immutability checks, regression DB label-count verification
  - DBSCAN: detects singleton outlier, no false positives on uniform data, single-record edge case
- Post-review fixes (commit c2aca95):
  - Rule ordering: sign_error now fires before off_by_one — prevents fault_wrong_sign being mislabeled
    off_by_one when max_abs_error lands in the medium bucket (256–32767) with all elements wrong
  - Float equality: mismatch_rate >= 0.999 replaces == 1.0, guarding against future float precision drift
  - Brittle test: test_all_real_records_labeled uses >= instead of == for len and per-label counts,
    so it survives future appends to the append-only regression_db.jsonl
- Total test suite: 79 tests (56 reference model + 8 feature extractor + 15 clusterer), 0 failures

## Session 11: COMPLETE
- `triage/triage_agent.py` — Claude API integration; per-cluster structured triage reports
  - triage(features, client=None): groups by cluster_label, skips clean_pass, returns dict[label→report]
  - _triage_cluster: builds rich prompt (test names, mismatch rates, error ranges, symptom flags,
    outlier count), calls claude-sonnet-4-6 (max_tokens=512), parses JSON response
  - PROMPTS dict at top for easy tuning: "system" + "cluster_user" templates
  - affected_configs built deterministically from records (not left to model)
  - Fallback on anthropic.AnthropicError: confidence=low, empty steps, no propagation
  - Report shape: likely_cause (str), confidence (high|medium|low),
    recommended_debug_steps (list[str], 2-3 items), affected_configs (list[str])
- `triage/test_triage_agent.py` — 8 pytest tests, all passing; zero real API calls
  - Mock client injected via client= parameter (MagicMock(spec=anthropic.Anthropic))
  - Tests: grouping/call-count, required fields, confidence validity, debug-steps shape,
    affected-configs non-empty, clean_pass skipped, empty input, API error fallback
- Total test suite: 87 tests (56 reference model + 8 feature extractor + 15 clusterer + 8 triage agent), 0 failures

## Session 12: COMPLETE
- `triage/rag_store.py` — TF-IDF-based store of historical failure records and triage reports
  - `_record_to_text`: converts structured fields → token string (label, bucket, dut, n, dtype, accw, rate, symptoms)
  - `RAGStore.add(record, report)`: appends entry, marks index dirty
  - `RAGStore.query(record, top_k=3)`: rebuilds TF-IDF lazily, returns cosine-ranked hits
  - Each hit: `{record, report, similarity: float, matched_fields: list[str]}`
  - `matched_fields`: lists fields where query and stored record agree (explainability)
  - `save(path)` / `load(path)`: JSON round-trip (text field excluded from file)
  - Empty store returns [] without error; top_k > len clamped gracefully
- `triage/triage_agent.py` updated:
  - `triage(features, client=None, rag_store=None)` — new optional rag_store parameter
  - `_triage_cluster(..., rag_store=None)` — queries store before API call, appends to prompt
  - `PROMPTS["rag_context"]` template added for the similar-failures section
  - After successful API call: each record in cluster added to store (grows over runs)
  - Error fallback path unchanged (store not updated on API error)
- `triage/test_rag_store.py` — 14 pytest tests, all passing
  - Empty store, add/query, top_k clamping, save/load round-trip, semantic ranking,
    unseen record type, matched_fields list, similarity float, result dict shape
- Total test suite: 101 tests (56 reference model + 8 feature extractor + 15 clusterer + 8 triage agent + 14 RAG store), 0 failures

## Phase 2 Goals
- [x] Implement failure feature extractor (triage/feature_extractor.py)
- [x] Implement failure clusterer (triage/clusterer.py)
- [x] Implement triage agent (triage/triage_agent.py): Claude API, per-cluster structured reports
- [ ] Extend CocoTB testbench to append live entries to regression_db.jsonl during simulation
- [ ] Add Makefile targets for fault-variant Verilator builds (per-fault sim_build dirs)

## Phase 3 Goals
- [x] RAG store for historical failures (triage/rag_store.py)
- [ ] Live regression_db.jsonl appends from CocoTB during simulation
- [ ] Fault-variant Makefile targets (per-fault sim_build dirs)
- [ ] End-to-end benchmark: run triage on live fault sim results, measure cluster accuracy

## Blockers / Open Questions
- None