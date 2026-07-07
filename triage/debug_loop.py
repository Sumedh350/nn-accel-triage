"""Multi-turn agentic debug loop wrapping the triage agent.

Iterates Claude conversations for a single failure cluster until confidence
reaches "high" or max_turns is exhausted.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from rag_store import RAGStore
from triage_agent import (
    PROMPTS,
    _MAX_TOKENS,
    _MODEL,
    _build_config_str,
    _build_rag_block,
    _strip_fences,
)


def _build_initial_message(label: str, records: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("records must not be empty")
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

    return PROMPTS["cluster_user"].format(
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


def _follow_up_question(report: dict[str, Any], records: list[dict[str, Any]]) -> str:
    cause = report.get("likely_cause", "").lower()
    has_overflow = any(r["has_overflow_symptom"] for r in records)
    has_reset = any(r["has_reset_symptom"] for r in records)

    if has_overflow and "overflow" not in cause and "accumulator" not in cause:
        focus = (
            "overflow symptoms were detected in the failing tests — "
            "could the accumulator width be too narrow, causing truncation?"
        )
    elif has_reset and "reset" not in cause and "polarity" not in cause:
        focus = (
            "reset symptoms (output stuck at zero) were detected — "
            "could an active-high reset be applied where active-low is expected?"
        )
    else:
        focus = (
            "what alternative RTL bug could produce the same symptom pattern "
            "(mismatch rate, error magnitude, and flag combination)?"
        )

    confidence = report.get("confidence", "low")
    return (
        f"Your current hypothesis has {confidence} confidence. "
        f"Consider: {focus} "
        "Respond with an updated JSON report using the same four keys."
    )


class DebugLoop:
    """Multi-turn conversation loop for a single failure cluster."""

    def run(
        self,
        label: str,
        records: list[dict[str, Any]],
        client: anthropic.Anthropic | None = None,
        rag_store: RAGStore | None = None,
        max_turns: int = 3,
    ) -> dict[str, Any]:
        """Iterate until confidence=="high" or max_turns exhausted.

        Returns:
            {
                "report":    final triage report dict,
                "turns":     number of turns taken,
                "history":   list of {"turn", "prompt", "response"} dicts,
                "converged": True iff final confidence == "high",
            }
        """
        if max_turns < 1:
            return {
                "report": {
                    "likely_cause": "No turns requested.",
                    "confidence": "low",
                    "recommended_debug_steps": [],
                    "affected_configs": sorted({_build_config_str(r) for r in records}),
                },
                "turns": 0,
                "history": [],
                "converged": False,
            }

        if client is None:
            client = anthropic.Anthropic()

        affected_configs = sorted({_build_config_str(r) for r in records})

        initial_msg = _build_initial_message(label, records)
        if rag_store is not None:
            rag_block = _build_rag_block(records, rag_store)
            if rag_block:
                initial_msg += PROMPTS["rag_context"].format(similar_block=rag_block)

        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_msg}]
        history: list[dict[str, Any]] = []
        last_report: dict[str, Any] | None = None
        turn = 0

        for turn in range(1, max_turns + 1):
            try:
                response = client.messages.create(
                    model=_MODEL,
                    max_tokens=_MAX_TOKENS,
                    system=PROMPTS["system"],
                    messages=messages,
                )
                raw_text = _strip_fences(response.content[0].text)
                report: dict[str, Any] = json.loads(raw_text)
                report["affected_configs"] = list(affected_configs)
                last_report = report

                history.append({
                    "turn": turn,
                    "prompt": messages[-1]["content"],
                    "response": report,
                })

                if report.get("confidence") == "high":
                    break

                if turn < max_turns:
                    follow_up = _follow_up_question(report, records)
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({"role": "user", "content": follow_up})

            except (anthropic.AnthropicError, json.JSONDecodeError, ValueError) as exc:
                fallback: dict[str, Any] = {
                    "likely_cause": f"API error: {exc}",
                    "confidence": "low",
                    "recommended_debug_steps": [],
                    "affected_configs": list(affected_configs),
                }
                history.append({
                    "turn": turn,
                    "prompt": messages[-1]["content"],
                    "response": fallback,
                })
                if last_report is None:
                    last_report = fallback
                break

        if (
            last_report is not None
            and rag_store is not None
            and not last_report.get("likely_cause", "").startswith("API error:")
        ):
            for r in records:
                rag_store.add(r, last_report)

        converged = last_report is not None and last_report.get("confidence") == "high"

        return {
            "report": last_report,
            "turns": turn,
            "history": history,
            "converged": converged,
        }
