SIM           = verilator
TOPLEVEL_LANG = verilog
TOPLEVEL      = mac_array_wrap
MODULE        = test_mac_array

# Absolute paths so make works regardless of working directory.
REPO_ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
VERILOG_SOURCES  = $(REPO_ROOT)/rtl/mac_array.sv \
                   $(abspath mac_array_wrap.sv)

# Configuration — override per convenience target.
MAC_N        ?= 4
MAC_DATA_TYPE ?= 0
export MAC_N
export MAC_DATA_TYPE

# Verilator compilation flags:
#   --trace / --trace-structs : generate VCD for waveform debug
#   -GN / -GDATA_TYPE         : override wrapper top-level parameters
EXTRA_ARGS += --trace --trace-structs
EXTRA_ARGS += -GN=$(MAC_N) -GDATA_TYPE=$(MAC_DATA_TYPE)

# Make mac_ref importable without modifying the test file.
export PYTHONPATH := $(REPO_ROOT)/tb/reference_model:$(PYTHONPATH)

# Each (N, DATA_TYPE) pair gets its own build directory so Verilator's
# baked-in parameter values don't bleed between configurations.
SIM_BUILD           = sim_build_N$(MAC_N)_DT$(MAC_DATA_TYPE)
COCOTB_RESULTS_FILE = results.xml

include $(shell cocotb-config --makefiles)/Makefile.sim

# ── Convenience targets ────────────────────────────────────────────────────────
.PHONY: test_int8 test_int16 test_n1 test_all

test_int8:
	$(MAKE) MAC_N=4 MAC_DATA_TYPE=0

test_int16:
	$(MAKE) MAC_N=4 MAC_DATA_TYPE=1

test_n1:
	$(MAKE) MAC_N=1 MAC_DATA_TYPE=0

test_all: test_int8 test_int16 test_n1
