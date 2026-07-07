"""Tests for triage/vcd_parser.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from triage.vcd_parser import parse_vcd

# Minimal synthetic VCD: 2 rising clk edges; out_valid rises on cycle 2.
_SYNTHETIC_VCD = """\
$version test $end
$timescale 1ps $end
 $scope module top $end
  $var wire 1 ! clk $end
  $var wire 1 " out_valid $end
  $var wire 8 # flat_q [7:0] $end
 $upscope $end
$enddefinitions $end
#0
0!
0"
b00000000 #
#5
1!
#10
0!
#15
1!
1"
b00001010 #
#20
0!
"""

# VCD where out_valid never rises (only clk toggles twice).
_NO_DIVERGE_VCD = """\
$version test $end
$timescale 1ps $end
 $scope module top $end
  $var wire 1 ! clk $end
  $var wire 1 " out_valid $end
 $upscope $end
$enddefinitions $end
#0
0!
0"
#5
1!
#10
0!
#15
1!
#20
0!
"""


def test_parse_synthetic_vcd(tmp_path: Path) -> None:
    vcd_file = tmp_path / "dump.vcd"
    vcd_file.write_text(_SYNTHETIC_VCD)

    result = parse_vcd(vcd_file)

    assert result["total_cycles"] == 2
    assert result["first_divergence_cycle"] == 2
    assert "clk" in result["diverging_signals"]
    assert "out_valid" in result["diverging_signals"]
    assert "flat_q" in result["diverging_signals"]


def test_missing_file() -> None:
    result = parse_vcd("/nonexistent/path/dump.vcd")

    assert result["first_divergence_cycle"] is None
    assert result["total_cycles"] is None
    assert result["diverging_signals"] is None


def test_first_divergence_none_when_no_out_valid_rise(tmp_path: Path) -> None:
    vcd_file = tmp_path / "dump.vcd"
    vcd_file.write_text(_NO_DIVERGE_VCD)

    result = parse_vcd(vcd_file)

    assert result["total_cycles"] == 2
    assert result["first_divergence_cycle"] is None
    assert "clk" in result["diverging_signals"]
