SIM           = verilator
TOPLEVEL_LANG = verilog

# Absolute paths so make works regardless of working directory.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)

# ── DUT selection ──────────────────────────────────────────────────────────────
# Pass DUT=quant_unit (or DUT=mac_array) on the command line to choose the
# design under test.  Convenience targets below always pass DUT explicitly.
DUT ?= mac_array

ifeq ($(DUT),quant_unit)
  TOPLEVEL        = quant_unit_wrap
  MODULE          = test_quant_unit
  VERILOG_SOURCES = $(REPO_ROOT)/rtl/quant_unit.sv \
                    $(abspath quant_unit_wrap.sv)

  QUANT_N     ?= 4
  QUANT_ACC_W ?= 32
  export QUANT_N
  export QUANT_ACC_W

  EXTRA_ARGS += --trace --trace-structs
  EXTRA_ARGS += -GN=$(QUANT_N) -GACC_W=$(QUANT_ACC_W)
  SIM_BUILD    = sim_build_quant_N$(QUANT_N)_AW$(QUANT_ACC_W)

else   # mac_array (default)
  TOPLEVEL        = mac_array_wrap
  MODULE          = test_mac_array
  VERILOG_SOURCES = $(REPO_ROOT)/rtl/mac_array.sv \
                    $(abspath mac_array_wrap.sv)

  MAC_N         ?= 4
  MAC_DATA_TYPE ?= 0
  export MAC_N
  export MAC_DATA_TYPE

  EXTRA_ARGS += --trace --trace-structs
  EXTRA_ARGS += -GN=$(MAC_N) -GDATA_TYPE=$(MAC_DATA_TYPE)
  SIM_BUILD    = sim_build_N$(MAC_N)_DT$(MAC_DATA_TYPE)
endif

# Both DUTs import from tb/reference_model/.
export PYTHONPATH := $(REPO_ROOT)/tb/reference_model:$(PYTHONPATH)

COCOTB_RESULTS_FILE = results.xml

include $(shell cocotb-config --makefiles)/Makefile.sim

# ── mac_array convenience targets ─────────────────────────────────────────────
.PHONY: test_int8 test_int16 test_n1 test_all

test_int8:
	$(MAKE) DUT=mac_array MAC_N=4 MAC_DATA_TYPE=0

test_int16:
	$(MAKE) DUT=mac_array MAC_N=4 MAC_DATA_TYPE=1

test_n1:
	$(MAKE) DUT=mac_array MAC_N=1 MAC_DATA_TYPE=0

test_all: test_int8 test_int16 test_n1

# ── quant_unit convenience targets ────────────────────────────────────────────
.PHONY: quant_int8 quant_int16 quant_n1 quant_all

quant_int8:
	$(MAKE) DUT=quant_unit QUANT_N=4 QUANT_ACC_W=32

quant_int16:
	$(MAKE) DUT=quant_unit QUANT_N=4 QUANT_ACC_W=48

quant_n1:
	$(MAKE) DUT=quant_unit QUANT_N=1 QUANT_ACC_W=32

quant_all: quant_int8 quant_int16 quant_n1
