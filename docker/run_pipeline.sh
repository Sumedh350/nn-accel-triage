#!/usr/bin/env bash
set -euo pipefail

cd /workspace

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; exit 1; }

echo "================================================================"
echo " NN Accelerator Triage — Full Pipeline"
echo "================================================================"
echo ""

echo "=== Stage 1: Reference model unit tests ==="
pytest tb/reference_model/ -v \
  && pass "reference model" \
  || fail "reference model"
echo ""

echo "=== Stage 2: Triage pipeline tests (triage/ benchmark/ dashboard/) ==="
pytest triage/ benchmark/ dashboard/ -v \
  && pass "triage pipeline" \
  || fail "triage pipeline"
echo ""

echo "=== Stage 3: CocoTB simulation (mac_array + quant_unit) ==="
make -C tb/cocotb test_all \
  && make -C tb/cocotb quant_all \
  && pass "CocoTB simulation" \
  || fail "CocoTB simulation"
echo ""

echo "=== Stage 4: Benchmark runner ==="
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "[SKIP] ANTHROPIC_API_KEY not set — skipping live benchmark (requires Claude API)"
else
  python3 benchmark/benchmark_runner.py \
    && pass "benchmark runner" \
    || fail "benchmark runner"
fi
echo ""

echo "=== Stage 5: Dashboard ==="
python3 -m dashboard.dashboard \
  && pass "dashboard" \
  || fail "dashboard"
echo ""

echo "================================================================"
echo " All stages complete. Dashboard: reports/dashboard.html"
echo "================================================================"
