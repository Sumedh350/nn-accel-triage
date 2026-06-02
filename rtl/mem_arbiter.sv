// rtl/mem_arbiter.sv
//
// Shared SRAM arbiter for weight (A) and activation (B) memory requestors.
//
// Round-robin arbitration grants one requestor per cycle.  When both requestors
// target the same bank simultaneously a 1-cycle stall is injected before serving
// the winner; bank_conflict is asserted during that stall cycle so the triage
// agent can distinguish true bank conflicts from ordinary serialisation.
//
// SRAM model: combinational read — sram_rdata is valid the same cycle that
// sram_ce and sram_addr are driven (no registered pipeline in the SRAM itself).
//
// Handshake (AXI-style valid/ready):
//   Requestor holds req_x_valid=1 until req_x_ready=1 (accepted).
//   Read data (req_x_rdata) and req_x_rvld are valid the same cycle as ready.
//
// Latency / Throughput:
//   Single requestor active       : 1-cycle latency, 1 req/cycle throughput
//   Both requestors, no conflict  : alternating 1-cycle grants (round-robin)
//   Both requestors, bank conflict: 2-cycle penalty (1 STALL + 1 SERVE)
//
// Parameters:
//   ADDR_W    – address bus width in bits (default 16)
//   DATA_W    – data bus width in bits    (default 32)
//   NUM_BANKS – number of SRAM banks; bank index = addr[BANK_BITS-1:0] (default 2)

`default_nettype none

module mem_arbiter #(
  parameter int unsigned ADDR_W    = 16,
  parameter int unsigned DATA_W    = 32,
  parameter int unsigned NUM_BANKS = 2
) (
  input  logic               clk,
  input  logic               rst_n,         // synchronous reset, active-low

  // ── Requestor A (weight loader) ──────────────────────────────────────────
  input  logic               req_a_valid,
  output logic               req_a_ready,
  input  logic [ADDR_W-1:0]  req_a_addr,
  input  logic               req_a_we,
  input  logic [DATA_W-1:0]  req_a_wdata,
  output logic [DATA_W-1:0]  req_a_rdata,
  output logic               req_a_rvld,    // read data valid (same cycle as ready)

  // ── Requestor B (activation loader) ─────────────────────────────────────
  input  logic               req_b_valid,
  output logic               req_b_ready,
  input  logic [ADDR_W-1:0]  req_b_addr,
  input  logic               req_b_we,
  input  logic [DATA_W-1:0]  req_b_wdata,
  output logic [DATA_W-1:0]  req_b_rdata,
  output logic               req_b_rvld,

  // ── Shared SRAM port ─────────────────────────────────────────────────────
  output logic               sram_ce,       // chip enable (active high)
  output logic               sram_we,       // write enable
  output logic [ADDR_W-1:0]  sram_addr,
  output logic [DATA_W-1:0]  sram_wdata,
  input  logic [DATA_W-1:0]  sram_rdata,   // combinational read data from SRAM

  // ── Triage / observability ───────────────────────────────────────────────
  output logic               bank_conflict  // 1 during the stall cycle paid for a conflict
);

  // ── Derived localparam ───────────────────────────────────────────────────
  // BANK_BITS: number of address LSBs used as the bank index.
  // Guard against $clog2(1)=0 which would produce an illegal 0-wide vector.
  localparam int unsigned BANK_BITS = (NUM_BANKS > 1) ? $clog2(NUM_BANKS) : 1;

  // ── Elaboration-time checks ──────────────────────────────────────────────
  initial begin
    if (ADDR_W < 1)
      $fatal(1, "mem_arbiter: ADDR_W must be >= 1 (got %0d).",    ADDR_W);
    if (DATA_W < 1)
      $fatal(1, "mem_arbiter: DATA_W must be >= 1 (got %0d).",    DATA_W);
    if (NUM_BANKS < 1)
      $fatal(1, "mem_arbiter: NUM_BANKS must be >= 1 (got %0d).", NUM_BANKS);
  end

  // ── FSM state encoding ───────────────────────────────────────────────────
  typedef enum logic { IDLE = 1'b0, STALL = 1'b1 } state_t;
  state_t state;

  // ── Round-robin and pending-winner state ─────────────────────────────────
  logic last_grant;    // 0 = A was most recently granted, 1 = B was
  logic pending_grant; // winner saved at conflict detection: 0 = A, 1 = B

  // ── Bank-conflict detection (combinational) ──────────────────────────────
  logic [BANK_BITS-1:0] bank_a, bank_b;
  logic                 conflict;

  assign bank_a   = req_a_addr[BANK_BITS-1:0];
  assign bank_b   = req_b_addr[BANK_BITS-1:0];
  assign conflict = req_a_valid & req_b_valid & (bank_a == bank_b);

  // ── Arbitration (combinational) ──────────────────────────────────────────
  // Exactly one of grant_a / grant_b is 1 each cycle that a request is served.
  logic grant_a, grant_b;

  always_comb begin
    grant_a = 1'b0;
    grant_b = 1'b0;
    case (state)

      // ── IDLE: normal arbitration ──────────────────────────────────────
      IDLE: begin
        if (!conflict) begin
          if (req_a_valid && req_b_valid) begin
            // Both want service: round-robin — serve whoever was NOT served last.
            grant_a = (last_grant == 1'b1);
            grant_b = (last_grant == 1'b0);
          end else begin
            grant_a = req_a_valid;
            grant_b = req_b_valid;
          end
        end
        // conflict: neither granted; FSM transitions to STALL this posedge.
      end

      // ── STALL: 1-cycle penalty; serve the pre-selected winner ────────
      STALL: begin
        grant_a = (pending_grant == 1'b0);
        grant_b = (pending_grant == 1'b1);
      end

      default: begin
        grant_a = 1'b0;
        grant_b = 1'b0;
      end

    endcase
  end

  // ── Handshake and read-data outputs (combinational) ──────────────────────
  assign req_a_ready = grant_a;
  assign req_b_ready = grant_b;
  assign req_a_rvld  = grant_a & ~req_a_we;
  assign req_b_rvld  = grant_b & ~req_b_we;
  assign req_a_rdata = sram_rdata;   // meaningful only while req_a_rvld=1
  assign req_b_rdata = sram_rdata;

  // ── SRAM drive (combinational) ───────────────────────────────────────────
  assign sram_ce    = grant_a | grant_b;
  assign sram_we    = (grant_a & req_a_we) | (grant_b & req_b_we);
  assign sram_addr  = grant_a ? req_a_addr  : req_b_addr;
  assign sram_wdata = grant_a ? req_a_wdata : req_b_wdata;

  // ── bank_conflict output (combinational from registered state) ───────────
  assign bank_conflict = (state == STALL);

  // ── FSM + round-robin state (sequential) ─────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state         <= IDLE;
      last_grant    <= 1'b0;
      pending_grant <= 1'b0;
    end else begin
      case (state)

        // ── IDLE ──────────────────────────────────────────────────────────
        IDLE: begin
          if (conflict) begin
            // Bank conflict: stall this cycle, save the alternate winner.
            // ~last_grant ensures we never grant the same side twice in a row.
            pending_grant <= ~last_grant;
            state         <= STALL;
          end else if (req_a_valid && req_b_valid) begin
            // Both active, no conflict: toggle round-robin state.
            last_grant <= ~last_grant;
          end
          // Single-requestor or no request: last_grant unchanged.
        end

        // ── STALL: serve pending winner, return to IDLE ────────────────
        STALL: begin
          last_grant <= pending_grant; // record who was just served
          state      <= IDLE;
        end

        default: state <= IDLE;

      endcase
    end
  end

endmodule

`default_nettype wire
