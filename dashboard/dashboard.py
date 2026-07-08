"""Generate a static HTML triage dashboard from regression_db.jsonl and benchmark JSON.

Usage (from project root):
    python -m dashboard.dashboard
Writes: reports/dashboard.html
"""

from __future__ import annotations

import html
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure triage/ modules are importable regardless of invocation context.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "triage"))
sys.path.insert(0, str(_ROOT))

_DEFAULT_DB = _ROOT / "regression_db.jsonl"
_DEFAULT_REPORTS_DIR = _ROOT / "reports"
_DEFAULT_OUTPUT = _DEFAULT_REPORTS_DIR / "dashboard.html"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _find_latest_benchmark(reports_dir: Path) -> dict[str, Any] | None:
    files = sorted(reports_dir.glob("benchmark_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _find_latest_triage_reports(reports_dir: Path) -> dict[str, Any] | None:
    files = sorted(reports_dir.glob("triage_reports_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _load_db_records(db_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in db_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "record_type" not in rec:
            records.append(rec)
    return records


def _compute_cluster_stats(
    clustered: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for feat in clustered:
        groups[feat["cluster_label"]].append(feat)

    stats: dict[str, dict[str, Any]] = {}
    for label, group in sorted(groups.items()):
        fail = [f for f in group if f.get("status") == "fail"]
        stats[label] = {
            "count": len(group),
            "fail_count": len(fail),
            "mismatch_rate_min": min((f["mismatch_rate"] for f in fail), default=0.0),
            "mismatch_rate_max": max((f["mismatch_rate"] for f in fail), default=0.0),
            "max_abs_error_min": min((f["max_abs_error"] for f in fail), default=0),
            "max_abs_error_max": max((f["max_abs_error"] for f in fail), default=0),
        }
    return stats


def _db_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for r in records if r.get("status") == "pass")
    failed = total - passed
    golden_passed = sum(
        1 for r in records if r.get("status") == "pass" and r.get("variant") == "golden"
    )
    latent_passed = sum(
        1 for r in records if r.get("status") == "pass" and r.get("variant") != "golden"
    )

    variant_counts: dict[str, int] = defaultdict(int)
    variant_pass: dict[str, int] = defaultdict(int)
    variant_fail: dict[str, int] = defaultdict(int)
    for r in records:
        v = r.get("variant", "unknown")
        variant_counts[v] += 1
        if r.get("status") == "pass":
            variant_pass[v] += 1
        else:
            variant_fail[v] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "golden_passed": golden_passed,
        "latent_passed": latent_passed,
        "variants": dict(sorted(variant_counts.items())),
        "variant_pass": dict(variant_pass),
        "variant_fail": dict(variant_fail),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f0f4f8; color: #2d3748; font-size: 14px; }

/* Header */
header { background: #1a1a2e; color: #e2e8f0; padding: 24px 32px; }
header h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 10px; }
.badges { display: flex; gap: 12px; flex-wrap: wrap; }
.badge { padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-green  { background: #276749; color: #c6f6d5; }
.badge-blue   { background: #2b6cb0; color: #bee3f8; }
.badge-purple { background: #553c9a; color: #e9d8fd; }
.badge-gray   { background: #4a5568; color: #e2e8f0; }

/* Sections */
main { max-width: 1100px; margin: 0 auto; padding: 28px 24px; }
section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
          margin-bottom: 28px; padding: 22px 24px; }
section h2 { font-size: 1.1rem; font-weight: 700; color: #1a1a2e;
             border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 16px; }

/* Tables */
table { width: 100%; border-collapse: collapse; }
th { background: #edf2f7; text-align: left; padding: 8px 12px;
     font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; color: #4a5568; }
td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f7fafc; }
tr:hover td { background: #ebf4ff; }

/* Bar chart */
.chart { margin-top: 4px; }
.chart-note { font-size: 0.78rem; color: #718096; margin-bottom: 12px; font-style: italic; }
.bar-row { display: flex; align-items: center; margin-bottom: 8px; gap: 10px; }
.bar-label { width: 130px; font-size: 0.82rem; color: #4a5568; flex-shrink: 0; }
.bar-track { flex: 1; background: #edf2f7; border-radius: 4px; height: 20px; }
.bar-fill { height: 100%; border-radius: 4px; background: #4a79a5; transition: width .3s; }
.bar-fill-dominant { background: #2b6cb0; }
.bar-value { width: 80px; font-size: 0.78rem; color: #718096; flex-shrink: 0; text-align: right; }

/* Triage cards */
.card { border: 1px solid #e2e8f0; border-radius: 6px; margin-bottom: 16px; padding: 16px 18px; }
.card h3 { font-size: 0.95rem; font-weight: 700; color: #1a1a2e; margin-bottom: 10px; }
.card-row { margin-bottom: 6px; font-size: 0.88rem; }
.card-row strong { color: #4a5568; }
.conf-high   { color: #276749; font-weight: 700; }
.conf-medium { color: #c05621; font-weight: 700; }
.conf-low    { color: #9b2c2c; font-weight: 700; }
.steps-list { margin: 4px 0 0 20px; }
.steps-list li { margin-bottom: 2px; }
.placeholder { background: #fffbeb; border: 1px solid #f6e05e; border-radius: 6px;
               padding: 14px 18px; color: #744210; font-size: 0.88rem; }

/* DB summary stat boxes */
.stat-boxes { display: flex; gap: 14px; margin-bottom: 18px; flex-wrap: wrap; }
.stat-box { flex: 1; min-width: 120px; background: #edf2f7; border-radius: 6px;
            padding: 14px 16px; text-align: center; }
.stat-box .num { font-size: 1.8rem; font-weight: 800; color: #1a1a2e; }
.stat-box .lbl { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em;
                 color: #718096; margin-top: 2px; }
.stat-box .stat-sub { font-size: 0.72rem; color: #a0aec0; margin-top: 4px; font-style: italic; }
.stat-box.pass .num { color: #276749; }
.stat-box.fail .num { color: #9b2c2c; }

/* Latent fault callout */
.latent-note { background: #fffbeb; border-left: 4px solid #f6ad55; border-radius: 4px;
               padding: 12px 16px; margin-bottom: 16px; font-size: 0.88rem; color: #744210;
               line-height: 1.5; }
.latent-note strong { color: #c05621; }
.latent-note code { background: #feebc8; padding: 1px 4px; border-radius: 3px; }

/* Latent tag in variant table */
.tag-latent { display: inline-block; background: #feebc8; color: #c05621;
              font-size: 0.7rem; font-weight: 700; padding: 1px 6px; border-radius: 10px;
              margin-left: 6px; vertical-align: middle; letter-spacing: .03em; }
td.pass-cell { color: #276749; font-weight: 600; }
td.fail-cell { color: #9b2c2c; font-weight: 600; }
td.zero-cell { color: #a0aec0; }

footer { text-align: center; color: #a0aec0; font-size: 0.78rem; padding: 12px; }
"""


def _e(text: Any) -> str:
    return html.escape(str(text))


def _confidence_class(conf: str) -> str:
    return {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}.get(
        conf.lower(), "conf-low"
    )


def _render_header(benchmark: dict[str, Any] | None, generated_at: str) -> str:
    if benchmark:
        speedup = f"{benchmark['speedup_factor']:.1f}×"
        acc = benchmark["eval_metrics"]["overall_accuracy"]
        accuracy = f"{acc * 100:.0f}%"
        conf_val = benchmark["eval_metrics"].get("mean_confidence", 0.0)
        confidence = f"{conf_val:.2f}"
        ts = benchmark.get("timestamp", "")[:10]
        badges = (
            f'<span class="badge badge-blue">Speedup {_e(speedup)}</span>'
            f'<span class="badge badge-green">Accuracy {_e(accuracy)}</span>'
            f'<span class="badge badge-purple">Confidence {_e(confidence)}</span>'
            f'<span class="badge badge-gray">Benchmark {_e(ts)}</span>'
        )
    else:
        badges = '<span class="badge badge-gray">No benchmark data found</span>'

    return (
        f"<header>"
        f"<h1>NN Accelerator Triage Dashboard</h1>"
        f'<div class="badges">{badges}</div>'
        f"</header>\n"
    )


def _render_timing_chart(benchmark: dict[str, Any] | None) -> str:
    if not benchmark:
        return ""

    stage_times: dict[str, float] = benchmark.get("stage_times", {})
    if not stage_times:
        return ""

    # Log-scale widths so short stages remain visible alongside triage (31 s).
    EPS = 1e-3
    log_values = {k: math.log10(v + EPS) for k, v in stage_times.items()}
    log_min = min(log_values.values())
    log_max = max(log_values.values())
    log_range = log_max - log_min if log_max != log_min else 1.0

    dominant_stage = max(stage_times, key=lambda k: stage_times[k])
    total_ai = benchmark.get("total_ai_time", 0.0)

    rows = ""
    for stage, t in stage_times.items():
        pct = (log_values[stage] - log_min) / log_range * 95 + 5  # 5–100%
        is_dom = stage == dominant_stage
        fill_cls = "bar-fill bar-fill-dominant" if is_dom else "bar-fill"
        unit = "ms" if t < 1 else "s"
        val_str = f"{t * 1000:.1f} ms" if t < 1 else f"{t:.3f} s"
        rows += (
            f'<div class="bar-row">'
            f'<span class="bar-label">{_e(stage.replace("_", " "))}</span>'
            f'<div class="bar-track"><div class="{fill_cls}" style="width:{pct:.1f}%"></div></div>'
            f'<span class="bar-value">{_e(val_str)}</span>'
            f"</div>\n"
        )

    dom_pct = stage_times[dominant_stage] / total_ai * 100 if total_ai else 0
    note = (
        f"Bar widths use log₁₀ scale — "
        f"<strong>triage</strong> dominates at "
        f"{dom_pct:.1f}% of total AI time ({stage_times[dominant_stage]:.2f} s). "
        f"Total pipeline: {total_ai:.2f} s."
    )

    return (
        f"<section>"
        f"<h2>Benchmark Timing</h2>"
        f'<p class="chart-note">{note}</p>'
        f'<div class="chart">{rows}</div>'
        f"</section>\n"
    )


def _render_cluster_table(cluster_stats: dict[str, dict[str, Any]]) -> str:
    rows = ""
    for label, s in cluster_stats.items():
        mr_range = (
            f"{s['mismatch_rate_min']:.3f}–{s['mismatch_rate_max']:.3f}"
            if s["fail_count"]
            else "—"
        )
        err_range = (
            f"{s['max_abs_error_min']:,}–{s['max_abs_error_max']:,}"
            if s["fail_count"]
            else "—"
        )
        rows += (
            f"<tr>"
            f"<td><code>{_e(label)}</code></td>"
            f"<td>{_e(s['count'])}</td>"
            f"<td>{_e(s['fail_count'])}</td>"
            f"<td>{_e(mr_range)}</td>"
            f"<td>{_e(err_range)}</td>"
            f"</tr>\n"
        )

    return (
        f"<section>"
        f"<h2>Cluster Summary</h2>"
        f"<table>"
        f"<thead><tr>"
        f"<th>Cluster Label</th><th>Records</th><th>Failures</th>"
        f"<th>Mismatch Rate</th><th>Max Abs Error</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table>"
        f"</section>\n"
    )


def _render_triage_reports(
    cluster_stats: dict[str, dict[str, Any]],
    triage_reports: dict[str, Any] | None,
) -> str:
    if triage_reports is None:
        placeholder = (
            '<div class="placeholder">'
            "<strong>Triage reports not available.</strong> "
            "Pass <code>triage_reports</code> to <code>generate_dashboard()</code> "
            "or run the benchmark pipeline with a Claude API client to generate them."
            "</div>"
        )
        return (
            f"<section><h2>Per-Cluster Triage Reports</h2>{placeholder}</section>\n"
        )

    cards = ""
    for label in cluster_stats:
        report = triage_reports.get(label)
        if report is None:
            continue
        cause = _e(report.get("likely_cause", "—"))
        conf = report.get("confidence", "low")
        conf_cls = _confidence_class(conf)
        steps = report.get("recommended_debug_steps", [])
        steps_html = "".join(f"<li>{_e(s)}</li>" for s in steps)
        configs = ", ".join(_e(c) for c in report.get("affected_configs", []))

        cards += (
            f'<div class="card">'
            f"<h3><code>{_e(label)}</code></h3>"
            f'<div class="card-row"><strong>Likely cause:</strong> {cause}</div>'
            f'<div class="card-row"><strong>Confidence:</strong> '
            f'<span class="{conf_cls}">{_e(conf.upper())}</span></div>'
        )
        if steps:
            cards += (
                f'<div class="card-row"><strong>Debug steps:</strong>'
                f'<ol class="steps-list">{steps_html}</ol></div>'
            )
        if configs:
            cards += f'<div class="card-row"><strong>Affected configs:</strong> {configs}</div>'
        cards += "</div>\n"

    if not cards:
        cards = '<div class="placeholder">No failure cluster reports to display.</div>'

    return f"<section><h2>Per-Cluster Triage Reports</h2>{cards}</section>\n"


def _render_db_summary(db_sum: dict[str, Any]) -> str:
    golden_p = db_sum.get("golden_passed", 0)
    latent_p = db_sum.get("latent_passed", 0)
    variant_pass = db_sum.get("variant_pass", {})
    variant_fail = db_sum.get("variant_fail", {})

    stat_boxes = (
        f'<div class="stat-boxes">'
        f'<div class="stat-box"><div class="num">{_e(db_sum["total"])}</div>'
        f'<div class="lbl">Total Records</div></div>'
        f'<div class="stat-box pass"><div class="num">{_e(db_sum["passed"])}</div>'
        f'<div class="lbl">Passed</div>'
        f'<div class="stat-sub">{_e(golden_p)} golden + {_e(latent_p)} latent faults</div></div>'
        f'<div class="stat-box fail"><div class="num">{_e(db_sum["failed"])}</div>'
        f'<div class="lbl">Failed</div>'
        f'<div class="stat-sub">detectable injected faults</div></div>'
        f"</div>\n"
    )

    latent_note = ""
    if latent_p > 0:
        latent_note = (
            f'<div class="latent-note">'
            f"<strong>&#9888; {_e(latent_p)} latent faults passed simulation</strong> "
            f"despite containing injected RTL bugs — flagged as suspicious by the triage pipeline. "
            f"These are mathematically undetectable with the current test vectors: "
            f"the accumulator values never exceed the narrowed bit-width, so the truncation "
            f"never triggers an observable mismatch. "
            f"See the <code>latent_fault</code> cluster above for the structural root-cause analysis."
            f"</div>\n"
        )

    rows = ""
    for variant, count in db_sum["variants"].items():
        p = variant_pass.get(variant, 0)
        f = variant_fail.get(variant, 0)
        is_latent = p > 0 and variant != "golden"
        latent_tag = ' <span class="tag-latent">latent</span>' if is_latent else ""
        p_cell = f"<td class='pass-cell'>{_e(p)}</td>" if p else "<td class='zero-cell'>—</td>"
        f_cell = f"<td class='fail-cell'>{_e(f)}</td>" if f else "<td class='zero-cell'>—</td>"
        rows += (
            f"<tr>"
            f"<td><code>{_e(variant)}</code>{latent_tag}</td>"
            f"<td>{_e(count)}</td>"
            f"{p_cell}"
            f"{f_cell}"
            f"</tr>\n"
        )

    return (
        f"<section>"
        f"<h2>Regression DB Summary</h2>"
        f"{stat_boxes}"
        f"{latent_note}"
        f"<table>"
        f"<thead><tr>"
        f"<th>Variant</th><th>Total</th><th>Pass</th><th>Fail</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table>"
        f"</section>\n"
    )


def _render_html(
    benchmark: dict[str, Any] | None,
    cluster_stats: dict[str, dict[str, Any]],
    triage_reports: dict[str, Any] | None,
    db_summary: dict[str, Any],
    generated_at: str,
) -> str:
    header = _render_header(benchmark, generated_at)
    timing = _render_timing_chart(benchmark)
    cluster_table = _render_cluster_table(cluster_stats)
    triage_section = _render_triage_reports(cluster_stats, triage_reports)
    db_section = _render_db_summary(db_summary)
    footer = f"<footer>Generated {_e(generated_at)}</footer>"

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>NN Accelerator Triage Dashboard</title>\n"
        f"<style>\n{_CSS}\n</style>\n"
        "</head>\n"
        "<body>\n"
        f"{header}"
        f"<main>\n"
        f"{timing}"
        f"{cluster_table}"
        f"{triage_section}"
        f"{db_section}"
        f"</main>\n"
        f"{footer}\n"
        "</body>\n"
        "</html>\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dashboard(
    db_path: Path | str = _DEFAULT_DB,
    reports_dir: Path | str = _DEFAULT_REPORTS_DIR,
    triage_reports: dict[str, Any] | None = None,
    output_path: Path | str = _DEFAULT_OUTPUT,
) -> Path:
    """Generate dashboard.html and return its path."""
    from feature_extractor import load_and_extract
    from clusterer import cluster

    db_path = Path(db_path)
    reports_dir = Path(reports_dir)
    output_path = Path(output_path)

    records = _load_db_records(db_path)
    features = load_and_extract(db_path)
    clustered = cluster(features)
    cluster_stats = _compute_cluster_stats(clustered)
    benchmark = _find_latest_benchmark(reports_dir)
    if triage_reports is None:
        triage_reports = _find_latest_triage_reports(reports_dir)
    db_sum = _db_summary(records)
    generated_at = datetime.now().isoformat(timespec="seconds")

    html_content = _render_html(benchmark, cluster_stats, triage_reports, db_sum, generated_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate triage dashboard HTML.")
    parser.add_argument(
        "--triage",
        action="store_true",
        help=(
            "Call the Claude API to generate per-cluster triage reports. "
            "Requires ANTHROPIC_API_KEY in the environment. "
            "Saves results to reports/triage_reports_YYYYMMDD.json for reuse."
        ),
    )
    args = parser.parse_args()

    triage_reports: dict[str, Any] | None = None
    if args.triage:
        import anthropic
        from feature_extractor import load_and_extract
        from clusterer import cluster
        from triage_agent import triage

        print("Running triage pipeline (calls Claude API)…")
        features = load_and_extract(_DEFAULT_DB)
        clustered = cluster(features)
        client = anthropic.Anthropic()
        triage_reports = triage(clustered, client=client)

        date_str = datetime.now().strftime("%Y%m%d")
        save_path = _DEFAULT_REPORTS_DIR / f"triage_reports_{date_str}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(triage_reports, indent=2), encoding="utf-8")
        print(f"Triage reports saved to {save_path}")

    out = generate_dashboard(triage_reports=triage_reports)
    print(f"Dashboard written to {out}")
