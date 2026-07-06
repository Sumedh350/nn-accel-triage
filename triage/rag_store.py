"""RAG store for historical triage failures.

Stores failure records + their triage reports and retrieves the most
similar past entries using TF-IDF cosine similarity (sklearn only — no
vector DB, no embedding API).

Similarity is computed over a token string derived from structured fields:
    label_<cluster_label> bucket_<magnitude> dut_<name> n_<config_n>
    dtype_<data_type> accw_<acc_w> rate_<bucket> reset_symptom overflow_symptom
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_RATE_BUCKETS = ((0.1, "rate_low"), (0.5, "rate_medium"), (0.9, "rate_high"))

_EXPLAIN_FIELDS = [
    "cluster_label",
    "error_magnitude_bucket",
    "dut",
    "config_n",
    "config_data_type",
    "config_acc_w",
]
_EXPLAIN_FLAGS = ["has_reset_symptom", "has_overflow_symptom"]


def _record_to_text(record: dict[str, Any]) -> str:
    parts: list[str] = []

    label = record.get("cluster_label")
    if label is not None:
        parts.append(f"label_{label}")

    bucket = record.get("error_magnitude_bucket", "none")
    parts.append(f"bucket_{bucket}")

    dut = record.get("dut", "")
    if dut:
        parts.append(f"dut_{dut}")

    n = record.get("config_n")
    if n is not None:
        parts.append(f"n_{n}")

    dt = record.get("config_data_type")
    if dt is not None:
        parts.append(f"dtype_{dt}")

    acc_w = record.get("config_acc_w")
    if acc_w is not None:
        parts.append(f"accw_{acc_w}")

    rate = record.get("mismatch_rate", 0.0)
    rate_label = "rate_very_high"
    for threshold, name in _RATE_BUCKETS:
        if rate < threshold:
            rate_label = name
            break
    parts.append(rate_label)

    if record.get("has_reset_symptom"):
        parts.append("reset_symptom")
    if record.get("has_overflow_symptom"):
        parts.append("overflow_symptom")

    return " ".join(parts)


def _matched_fields(query: dict[str, Any], stored: dict[str, Any]) -> list[str]:
    matched: list[str] = []
    for field in _EXPLAIN_FIELDS:
        if query.get(field) == stored.get(field):
            matched.append(field)
    for flag in _EXPLAIN_FLAGS:
        if query.get(flag) and stored.get(flag):
            matched.append(flag)
    return matched


class RAGStore:
    """TF-IDF-backed store of historical failure records and triage reports."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None  # sparse CSR matrix
        self._dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: dict[str, Any], report: dict[str, Any]) -> None:
        """Store one failure record together with its triage report."""
        text = _record_to_text(record)
        self._entries.append({"record": record, "report": report, "text": text})
        self._dirty = True

    def query(self, record: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        """Return up to top_k most similar past records with their reports.

        Each result dict has keys: record, report, similarity (float), matched_fields (list[str]).
        Returns [] when the store is empty.
        """
        if not self._entries:
            return []

        if self._dirty:
            self._rebuild_index()

        query_text = _record_to_text(record)
        query_vec = self._vectorizer.transform([query_text])  # type: ignore[union-attr]
        sims = cosine_similarity(query_vec, self._matrix)[0]

        k = min(top_k, len(self._entries))
        top_indices = np.argsort(sims)[::-1][:k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            entry = self._entries[idx]
            results.append({
                "record": entry["record"],
                "report": entry["report"],
                "similarity": float(sims[idx]),
                "matched_fields": _matched_fields(record, entry["record"]),
            })
        return results

    def save(self, path: str | Path) -> None:
        """Persist all entries to a JSON file (text field excluded)."""
        data = [{"record": e["record"], "report": e["report"]} for e in self._entries]
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RAGStore":
        """Restore a RAGStore from a previously saved JSON file."""
        store = cls()
        data: list[dict[str, Any]] = json.loads(Path(path).read_text())
        for item in data:
            store.add(item["record"], item["report"])
        return store

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        texts = [e["text"] for e in self._entries]
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(texts)
        self._dirty = False
