"""VCD waveform parser for triage feature extraction.

Extracts cycle-level timing features from Verilator-generated VCD files
using Python stdlib only.
"""

from __future__ import annotations

from pathlib import Path


def parse_vcd(vcd_path: str | Path) -> dict:
    """Parse a VCD file and return waveform summary features.

    Returns a dict with:
      - first_divergence_cycle: int | None  (cycle when out_valid first rises)
      - total_cycles: int | None            (total rising edges of clk)
      - diverging_signals: list[str] | None (signal names that changed at all)

    Returns all-None dict on missing/empty file.
    """
    _none: dict = {"first_divergence_cycle": None, "total_cycles": None, "diverging_signals": None}

    try:
        text = Path(vcd_path).read_text()
    except (FileNotFoundError, OSError):
        return _none

    if not text.strip():
        return _none

    # --- Header: collect id → name mapping ---
    id_to_name: dict[str, str] = {}
    in_defs = False
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("$var"):
            # $var wire <width> <id> <name> [<range>] $end
            tokens = line.split()
            if len(tokens) >= 5:
                sig_id = tokens[3]
                sig_name = tokens[4]
                if sig_id not in id_to_name:
                    id_to_name[sig_id] = sig_name
        if "$enddefinitions" in line:
            in_defs = True
            i += 1
            break
        i += 1

    if not in_defs:
        return _none

    # Reverse map: signal name → first-declared id
    name_to_id: dict[str, str] = {}
    for sig_id, name in id_to_name.items():
        if name not in name_to_id:
            name_to_id[name] = sig_id

    clk_id = name_to_id.get("clk")
    ov_id = name_to_id.get("out_valid")

    # --- Simulation body: track value changes ---
    last_value: dict[str, str] = {}
    changed_ids: set[str] = set()
    rising_edges = 0
    first_divergence_cycle: int | None = None

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("$") or line.startswith("#"):
            continue

        if line.startswith("b"):
            # b<val> <id>
            parts = line.split()
            if len(parts) == 2:
                val, sig_id = parts[0][1:], parts[1]
                prev = last_value.get(sig_id)
                if prev != val:
                    changed_ids.add(sig_id)
                    last_value[sig_id] = val
        elif len(line) >= 2:
            # <value><id>  (1-bit)
            val, sig_id = line[0], line[1:]
            prev = last_value.get(sig_id)
            if prev != val:
                changed_ids.add(sig_id)
                last_value[sig_id] = val

                if sig_id == clk_id and prev == "0" and val == "1":
                    rising_edges += 1

                # Detect out_valid rising — record current clock cycle count
                if (
                    sig_id == ov_id
                    and prev == "0"
                    and val == "1"
                    and first_divergence_cycle is None
                ):
                    first_divergence_cycle = rising_edges if rising_edges > 0 else 1

    diverging_signals = sorted(
        id_to_name[sid] for sid in changed_ids if sid in id_to_name
    )

    return {
        "first_divergence_cycle": first_divergence_cycle,
        "total_cycles": rising_edges,
        "diverging_signals": diverging_signals,
    }
