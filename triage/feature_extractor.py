"""Feature extractor for regression_db.jsonl records.

Reads raw regression records and emits one structured feature dict per
record for downstream clustering and triage.
"""

from __future__ import annotations

import json
from pathlib import Path


def _magnitude_bucket(max_abs_error: int) -> str:
    if max_abs_error == 0:
        return "none"
    if max_abs_error < 256:
        return "small"
    if max_abs_error < 32768:
        return "medium"
    return "large"


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def extract_features(record: dict, vcd_path: str | Path | None = None) -> dict:
    """Return a feature dict for one regression_db record.

    vcd_path: optional path to a VCD file; if provided, adds first_divergence_cycle
    and total_cycles to the returned dict.
    """
    status = record["status"]
    config = record["config"]
    details = record.get("mismatch_details")

    if details is not None:
        total = details["total_elements"]
        count = details["mismatch_count"]
        max_err = int(details["max_abs_error"])
        mismatch_rate = count / total
        first_actual = details["first_mismatch"]["actual"]
    else:
        mismatch_rate = 0.0
        max_err = 0
        first_actual = None

    has_reset_symptom = (
        status == "fail" and first_actual is not None and first_actual == 0
    )
    has_overflow_symptom = max_err >= 32768 and _is_power_of_two(max_err)

    # True when actual == -expected at first mismatch — signature of a subtract fault.
    if details is not None:
        fm = details["first_mismatch"]
        has_subtract_symptom = int(fm["actual"]) == -int(fm["expected"])
    else:
        has_subtract_symptom = False

    feat: dict = {
        "run_id": record["run_id"],
        "dut": record["dut"],
        "variant": record["variant"],
        "test_name": record["test_name"],
        "status": status,
        "config_n": config["n"],
        "config_data_type": config.get("data_type"),
        "config_acc_w": config.get("acc_w"),
        "mismatch_rate": mismatch_rate,
        "max_abs_error": max_err,
        "error_magnitude_bucket": _magnitude_bucket(max_err),
        "has_reset_symptom": has_reset_symptom,
        "has_overflow_symptom": has_overflow_symptom,
        "has_subtract_symptom": has_subtract_symptom,
    }

    if vcd_path is not None:
        from triage.vcd_parser import parse_vcd
        vcd = parse_vcd(vcd_path)
        feat["first_divergence_cycle"] = vcd["first_divergence_cycle"]
        feat["total_cycles"] = vcd["total_cycles"]
    else:
        feat["first_divergence_cycle"] = None
        feat["total_cycles"] = None

    return feat


def load_and_extract(db_path: str | Path) -> list[dict]:
    """Read regression_db.jsonl and return a feature dict per record."""
    path = Path(db_path)
    features: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "record_type" in record:
                continue  # skip meta-records (e.g. benchmark_run entries)
            features.append(extract_features(record))
    return features
