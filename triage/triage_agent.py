"""Triage agent: calls Claude API to generate structured reports per failure cluster.

Input:  list of feature dicts already enriched by clusterer.cluster()
        (each dict has cluster_label and outlier fields)
Output: dict mapping cluster_label -> triage report
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import anthropic

from rag_store import RAGStore


PROMPTS: dict[str, str] = {
    "system": (
        "You are a hardware verification triage expert for a neural-network accelerator.\n"
        "Analyze the cluster of regression test failures described below and identify the "
        "most likely RTL root cause.\n"
        "Respond ONLY with a JSON object — no markdown, no explanation, no text outside the JSON.\n"
        "The JSON must have exactly these four keys:\n"
        '  "likely_cause"            — one sentence root-cause hypothesis\n'
        '  "confidence"              — "high", "medium", or "low"\n'
        '  "recommended_debug_steps" — list of 2-3 short, actionable next steps\n'
        '  "affected_configs"        — list of config strings from the cluster\n'
    ),
    "cluster_user": (
        "Cluster label: {label}\n\n"
        "Tests ({n_tests} records):\n"
        "{test_rows}\n\n"
        "Symptom summary:\n"
        "  Mismatch rate range : {mismatch_rate_min:.3f} – {mismatch_rate_max:.3f}\n"
        "  Max absolute error  : {max_err_min} – {max_err_max}\n"
        "  Error buckets       : {buckets}\n"
        "  Reset symptom       : {has_reset}\n"
        "  Overflow symptom    : {has_overflow}\n"
        "  Outlier count       : {n_outliers}\n\n"
        "Identify the most likely RTL root cause for this failure cluster."
    ),
    "rag_context": (
        "\nPast similar failures (for reference only — do not copy blindly):\n"
        "{similar_block}\n"
    ),
}

_SKIP_LABELS: frozenset[str] = frozenset({"clean_pass"})


def _strip_fences(text: str) -> str:
    """Strip markdown code fences that some model versions add around JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text
_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 512

_DATA_TYPE_NAMES: dict[int, str] = {0: "INT8", 1: "INT16", 2: "FP16"}


def _build_config_str(record: dict[str, Any]) -> str:
    parts = [f"{record['dut']}/N={record['config_n']}"]
    dt = record.get("config_data_type")
    if dt is not None:
        parts.append(_DATA_TYPE_NAMES.get(dt, f"DT{dt}"))
    acc_w = record.get("config_acc_w")
    if acc_w is not None:
        parts.append(f"AW{acc_w}")
    parts.append(record["variant"])
    return "/".join(parts)


def _build_rag_block(records: list[dict[str, Any]], rag_store: RAGStore) -> str:
    """Return a formatted string of top-3 similar past failures, or ""."""
    if not records:
        return ""
    similar = rag_store.query(records[0], top_k=3)
    if not similar:
        return ""
    lines: list[str] = []
    for i, hit in enumerate(similar, start=1):
        rep = hit["report"]
        matched = hit["matched_fields"]
        lines.append(
            f"[{i}] matched_fields={matched} | "
            f"cause: {rep.get('likely_cause', '')} | "
            f"confidence: {rep.get('confidence', '')}"
        )
    return "\n".join(lines)


def _triage_cluster(
    label: str,
    records: list[dict[str, Any]],
    client: anthropic.Anthropic,
    rag_store: RAGStore | None = None,
) -> dict[str, Any]:
    affected_configs = sorted({_build_config_str(r) for r in records})

    test_rows = "\n".join(
        f"  {r['test_name']} | {r['variant']} | N={r['config_n']} | "
        f"mismatch={r['mismatch_rate']:.3f} | max_err={r['max_abs_error']}"
        for r in records
    )
    rates = [r["mismatch_rate"] for r in records]
    errors = [r["max_abs_error"] for r in records]
    buckets = ", ".join(sorted({r["error_magnitude_bucket"] for r in records}))
    has_reset = any(r["has_reset_symptom"] for r in records)
    has_overflow = any(r["has_overflow_symptom"] for r in records)
    n_outliers = sum(1 for r in records if r.get("outlier", False))

    user_msg = PROMPTS["cluster_user"].format(
        label=label,
        n_tests=len(records),
        test_rows=test_rows,
        mismatch_rate_min=min(rates),
        mismatch_rate_max=max(rates),
        max_err_min=min(errors),
        max_err_max=max(errors),
        buckets=buckets,
        has_reset=has_reset,
        has_overflow=has_overflow,
        n_outliers=n_outliers,
    )

    if rag_store is not None:
        rag_block = _build_rag_block(records, rag_store)
        if rag_block:
            user_msg += PROMPTS["rag_context"].format(similar_block=rag_block)

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=PROMPTS["system"],
            messages=[{"role": "user", "content": user_msg}],
        )
        report: dict[str, Any] = json.loads(_strip_fences(response.content[0].text))
        report["affected_configs"] = affected_configs
        if rag_store is not None:
            for r in records:
                rag_store.add(r, report)
        return report
    except (anthropic.AnthropicError, json.JSONDecodeError) as exc:
        return {
            "likely_cause": f"API error: {exc}",
            "confidence": "low",
            "recommended_debug_steps": [],
            "affected_configs": affected_configs,
        }


def triage(
    features: list[dict[str, Any]],
    client: anthropic.Anthropic | None = None,
    rag_store: RAGStore | None = None,
) -> dict[str, dict[str, Any]]:
    """Group clustered feature dicts by label and call Claude for each failure cluster.

    Args:
        features:  list of dicts from clusterer.cluster() (must have cluster_label field)
        client:    Anthropic client; created automatically if None
        rag_store: optional RAGStore; if provided, similar past failures are included in
                   the prompt and each generated report is added to the store

    Returns:
        dict mapping cluster_label -> structured triage report
    """
    if not features:
        return {}

    if client is None:
        client = anthropic.Anthropic()

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in features:
        groups[record["cluster_label"]].append(record)

    result: dict[str, dict[str, Any]] = {}
    for label, records in groups.items():
        if label in _SKIP_LABELS:
            continue
        result[label] = _triage_cluster(label, records, client, rag_store=rag_store)

    return result
