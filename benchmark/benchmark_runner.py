"""Benchmark runner: times the full triage pipeline end-to-end vs manual baseline.

Manual baseline assumptions (45 minutes per failure cluster):
  - 5 min  — read test logs and simulation artifacts
  - 15 min — form hypotheses about RTL root cause
  - 20 min — verify hypothesis (run targeted tests, inspect waveforms)
  - 5 min  — document findings
Total: 45 min × number_of_failure_clusters

Pipeline stages timed:
  1. load             — read regression_db.jsonl
  2. extract_features — feature_extractor.extract_features per record
  3. cluster          — clusterer.cluster (rule-based + DBSCAN)
  4. triage           — triage_agent.triage (one Claude API call per failure cluster)
  5. evaluate         — evaluator.evaluate (score against ground truth)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure triage/ modules are importable regardless of invocation context.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "triage"))
sys.path.insert(0, str(_ROOT))

import anthropic  # noqa: E402

from benchmark.evaluator import evaluate  # noqa: E402
from clusterer import cluster  # noqa: E402
from feature_extractor import extract_features  # noqa: E402
from triage_agent import triage  # noqa: E402

# ---------------------------------------------------------------------------
# Manual baseline constants
# ---------------------------------------------------------------------------
# Minutes a single engineer spends triaging one failure cluster by hand.
MANUAL_MINUTES_PER_CLUSTER: int = 45
MANUAL_SECONDS_PER_CLUSTER: int = MANUAL_MINUTES_PER_CLUSTER * 60  # 2700 s

_DEFAULT_DB = _ROOT / "regression_db.jsonl"
_DEFAULT_GT = Path(__file__).parent / "ground_truth.json"


def _rule_based_triage(clustered: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build triage reports from cluster labels without calling the LLM."""
    cluster_configs: dict[str, list[str]] = defaultdict(list)
    for feat in clustered:
        label = feat.get("cluster_label", "uncategorized")
        if label in ("clean_pass",):
            continue
        cfg = feat.get("config", {})
        cfg_str = f"{feat.get('dut', '?')}/N={cfg.get('n', '?')}"
        cluster_configs[label].append(cfg_str)
    return {
        label: {
            "likely_cause": f"Rule-based cluster '{label}' — no LLM analysis",
            "confidence": "rule-based",
            "recommended_debug_steps": [],
            "affected_configs": sorted(set(configs)),
        }
        for label, configs in cluster_configs.items()
    }


def load_regression_db(db_path: Path | str) -> list[dict[str, Any]]:
    """Read all JSON records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with Path(db_path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "record_type" not in record:  # skip meta-records (benchmark_run etc.)
                records.append(record)
    return records


def run_benchmark(
    db_path: Path | str = _DEFAULT_DB,
    ground_truth_path: Path | str = _DEFAULT_GT,
    client: anthropic.Anthropic | None = None,
    append_to_db: bool = False,
    no_llm: bool = False,
) -> dict[str, Any]:
    """Run the full triage pipeline and record wall-clock time per stage.

    Args:
        db_path:           path to regression_db.jsonl
        ground_truth_path: path to benchmark/ground_truth.json
        client:            Anthropic client; created automatically if None
        append_to_db:      when True, appends a benchmark_run entry to db_path
        no_llm:            when True, skips Claude API call and uses cluster
                           labels directly as the triage result

    Returns:
        {
            "stage_times":             {stage_name: seconds},
            "total_ai_time":           float,
            "manual_baseline_seconds": float,
            "speedup_factor":          float,
            "eval_metrics":            dict from evaluator.evaluate(),
            "timestamp":               ISO-8601 string,
            "mode":                    "no_llm" | "llm",
        }
    """
    t_total_start = time.perf_counter()
    stage_times: dict[str, float] = {}

    t0 = time.perf_counter()
    records = load_regression_db(db_path)
    stage_times["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    features = [extract_features(r) for r in records]
    stage_times["extract_features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clustered = cluster(features)
    stage_times["cluster"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if no_llm:
        reports = _rule_based_triage(clustered)
    else:
        reports = triage(clustered, client=client)
    stage_times["triage"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    eval_metrics = evaluate(clustered, reports, ground_truth_path)
    stage_times["evaluate"] = time.perf_counter() - t0

    total_ai_time = time.perf_counter() - t_total_start

    # triage() skips clean_pass via _SKIP_LABELS, so len(reports) == failure clusters only.
    n_failure_clusters = len(reports)
    manual_baseline_seconds = float(n_failure_clusters * MANUAL_SECONDS_PER_CLUSTER)
    speedup_factor = (
        manual_baseline_seconds / total_ai_time if total_ai_time > 0 else float("inf")
    )

    result: dict[str, Any] = {
        "stage_times": stage_times,
        "total_ai_time": total_ai_time,
        "manual_baseline_seconds": manual_baseline_seconds,
        "speedup_factor": speedup_factor,
        "eval_metrics": eval_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "no_llm" if no_llm else "llm",
    }

    if append_to_db:
        entry: dict[str, Any] = {
            "record_type": "benchmark_run",
            "timestamp": result["timestamp"],
            "total_ai_time": result["total_ai_time"],
            "manual_baseline_seconds": result["manual_baseline_seconds"],
            "speedup_factor": result["speedup_factor"],
            "eval_metrics": result["eval_metrics"],
        }
        with Path(db_path).open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    return result


def _print_result(result: dict[str, Any], label: str) -> None:
    from benchmark.evaluator import format_report

    print(f"\n--- {label} ---")
    print(f"Total time:       {result['total_ai_time']:.2f}s")
    print(
        f"Manual baseline:  {result['manual_baseline_seconds']:.0f}s"
        f" ({result['manual_baseline_seconds'] / 60:.0f} min)"
    )
    print(f"Speedup factor:   {result['speedup_factor']:.1f}x")
    print("\nStage times:")
    for stage, t in result["stage_times"].items():
        print(f"  {stage:<20} {t:.4f}s")
    print()
    print(format_report(result["eval_metrics"]))


def _comparison_table(rb: dict[str, Any], llm: dict[str, Any] | None) -> str:
    """Return a side-by-side comparison of rule-based vs LLM-augmented metrics."""
    rb_acc = rb["eval_metrics"]["overall_accuracy"]
    rb_conf = rb["eval_metrics"]["mean_confidence"]
    rb_spd = rb["speedup_factor"]
    lines = [
        f"\n{'Mode':<20} {'Accuracy':>10} {'Mean Conf':>10} {'Speedup':>10}",
        "-" * 54,
        f"{'rule-based':<20} {rb_acc:>10.3f} {rb_conf:>10.3f} {rb_spd:>9.1f}x",
    ]
    if llm is not None:
        llm_acc = llm["eval_metrics"]["overall_accuracy"]
        llm_conf = llm["eval_metrics"]["mean_confidence"]
        llm_spd = llm["speedup_factor"]
        lines.append(
            f"{'llm-augmented':<20} {llm_acc:>10.3f} {llm_conf:>10.3f} {llm_spd:>9.1f}x"
        )
        delta_acc = llm_acc - rb_acc
        lines.append("-" * 54)
        lines.append(f"  LLM delta accuracy: {delta_acc:+.3f}")
    else:
        lines.append(f"{'llm-augmented':<20} {'(skipped — ANTHROPIC_API_KEY not set)':>34}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run triage pipeline benchmark")
    parser.add_argument("--db", default=str(_DEFAULT_DB), help="regression_db.jsonl path")
    parser.add_argument("--gt", default=str(_DEFAULT_GT), help="ground_truth.json path")
    parser.add_argument("--out", default=None, help="Output JSON path (default: reports/benchmark_YYYYMMDD.json)")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Claude API call; use cluster labels as triage result and compare vs LLM if key available",
    )
    args = parser.parse_args()

    if args.no_llm:
        rb_result = run_benchmark(
            db_path=args.db,
            ground_truth_path=args.gt,
            append_to_db=False,
            no_llm=True,
        )
        _print_result(rb_result, "Rule-based (no LLM)")

        llm_result: dict[str, Any] | None = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("\nANTHROPIC_API_KEY found — running LLM pipeline for comparison...")
            llm_result = run_benchmark(
                db_path=args.db,
                ground_truth_path=args.gt,
                append_to_db=True,
                no_llm=False,
            )
            _print_result(llm_result, "LLM-augmented")
        else:
            print("\nANTHROPIC_API_KEY not set — skipping LLM comparison run.")

        print(_comparison_table(rb_result, llm_result))
        result = rb_result
    else:
        result = run_benchmark(
            db_path=args.db,
            ground_truth_path=args.gt,
            append_to_db=True,
        )
        _print_result(result, "LLM-augmented")

    out_path = args.out
    if out_path is None:
        reports_dir = _ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = "_no_llm" if args.no_llm else ""
        out_path = str(reports_dir / f"benchmark_{date_str}{suffix}.json")

    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_path}")
