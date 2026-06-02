// rtl/control_fsm.sv
//
// Top-level sequencer: instantiates mem_arbiter, mac_array, and quant_unit,
// and drives the full pipeline for one N×N tile per start/done transaction:
//   LOAD   — fetch N×N weights (req_a) and N×N activations (req_b) from SRAM
//   COMPUTE — run mac_array, forward result to quant_unit via direct wiring
//   QUANTIZE — wait for quant_unit to finish, latch INT8 tile
//   STORE   — write N×N quantized bytes back to SRAM via req_a
//
// Handshake: start is sampled at posedge clk; done pulses for exactly one cycle.
// Reset: synchronous active-low (rst_n).

`default_nettype none

module control_fsm #(
  parameter  int unsigned N         = 4,
  parameter  int unsigned DATA_TYPE = 0,      // 0=INT8, 1=INT16, 2=FP16-stub
  parameter  int unsigned ADDR_W    = 16,
  parameter  int unsigned NUM_BANKS = 2,
  parameter  int unsigned SCALE_W   = 16,
  parameter  int unsigned SHIFT_W   = 5,
  localparam int unsigned DATA_W    = (DATA_TYPE == 0) ? 8  : 16,
  localparam int unsigned ACC_W     = (DATA_TYPE == 0) ? 32 :
                                      (DATA_TYPE == 1) ? 48 : 32,
  localparam int unsigned TILE_ELS  = N * N,
  localparam int unsigned CNT_W     = $clog2(TILE_ELS) + 1
) (
  input  logic clk,
  input  logic rst_n,

  // ── Tile control ─────────────────────────────────────────────────────────
  input  logic start,
  output logic done,

  // ── SRAM base addresses (must be stable while busy) ──────────────────────
  input  logic [ADDR_W-1:0] weight_base_addr,
  input  logic [ADDR_W-1:0] act_base_addr,
  input  logic [ADDR_W-1:0] out_base_addr,

  // ── Quantization parameters (must be stable while busy) ──────────────────
  input  logic        [SCALE_W-1:0] scale     [N],
  input  logic        [SHIFT_W-1:0] shift_amt [N],   // renamed: 'shift' is SV keyword
  input  logic signed [7:0]         zero_pt   [N],

  // ── Shared SRAM (pass-through from mem_arbiter) ───────────────────────────
  output logic               sram_ce,
  output logic               sram_we,
  output logic [ADDR_W-1:0]  sram_addr,
  output logic [DATA_W-1:0]  sram_wdata,
  input  logic [DATA_W-1:0]  sram_rdata,

  // ── Observability ─────────────────────────────────────────────────────────
  output logic               bank_conflict
);

  // ── Elaboration-time checks ───────────────────────────────────────────────
  initial begin
    if (N < 1)
      $fatal(1, "control_fsm: N must be >= 1 (got %0d).", N);
    if (DATA_TYPE > 2)
      $fatal(1, "control_fsm: DATA_TYPE=%0d unsupported (0=INT8, 1=INT16, 2=FP16).",
             DATA_TYPE);
    if (ADDR_W < 1)
      $fatal(1, "control_fsm: ADDR_W must be >= 1 (got %0d).", ADDR_W);
    if (NUM_BANKS < 1)
      $fatal(1, "control_fsm: NUM_BANKS must be >= 1 (got %0d).", NUM_BANKS);
  end

  // ── FSM state encoding ────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    IDLE     = 3'b000,
    LOAD     = 3'b001,
    COMPUTE  = 3'b010,
    QUANTIZE = 3'b011,
    STORE    = 3'b100,
    DONE     = 3'b101
  } state_t;

  state_t state;

  // ── Tile data registers ───────────────────────────────────────────────────
  logic signed [DATA_W-1:0] weight_tile  [N][N];
  logic signed [DATA_W-1:0] act_tile     [N][N];
  logic [7:0]        quant_result [N][N];   // latched from quant_unit.out_q

  // ── Element counters ──────────────────────────────────────────────────────
  logic [CNT_W-1:0] w_cnt, a_cnt, s_cnt;
  logic             w_done, a_done;

  // ── 2-D index decode (widened to 32 bits to satisfy DIV/MODDIV width rules)
  logic [31:0] wi, wj, ai, aj, si, sj;
  assign wi = 32'(w_cnt) / N;   assign wj = 32'(w_cnt) % N;
  assign ai = 32'(a_cnt) / N;   assign aj = 32'(a_cnt) % N;
  assign si = 32'(s_cnt) / N;   assign sj = 32'(s_cnt) % N;

  // ── mem_arbiter requestor wires ───────────────────────────────────────────
  logic               req_a_valid, req_a_ready;
  logic [ADDR_W-1:0]  req_a_addr;
  logic               req_a_we;
  logic [DATA_W-1:0]  req_a_wdata;
  logic [DATA_W-1:0]  req_a_rdata;
  logic               req_a_rvld;

  logic               req_b_valid, req_b_ready;
  logic [ADDR_W-1:0]  req_b_addr;
  logic               req_b_we;
  logic [DATA_W-1:0]  req_b_wdata;
  logic [DATA_W-1:0]  req_b_rdata;
  logic               req_b_rvld;

  // ── mac_array interface wires ─────────────────────────────────────────────
  logic                    mac_in_valid,  mac_in_ready;
  logic                    mac_out_valid, mac_out_ready;
  logic signed [ACC_W-1:0] mac_out_c [N][N];

  // ── quant_unit interface wires ────────────────────────────────────────────
  logic               quant_in_valid,  quant_in_ready;
  logic               quant_out_valid, quant_out_ready;
  logic signed [7:0]  quant_out_q [N][N];

  // ── mac_array instantiation ───────────────────────────────────────────────
  // in_a = activations, in_b = weights  →  C = activations × weights
  mac_array #(
    .N        (N),
    .DATA_TYPE(DATA_TYPE)
  ) mac_inst (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_a     (act_tile),
    .in_b     (weight_tile),
    .in_valid (mac_in_valid),
    .in_ready (mac_in_ready),
    .out_c    (mac_out_c),
    .out_valid(mac_out_valid),
    .out_ready(mac_out_ready)
  );

  // ── quant_unit instantiation ──────────────────────────────────────────────
  // in_acc is wired directly to mac_out_c; quant computes combinationally
  // from its latched acc_reg so there is no extra pipeline stage here.
  quant_unit #(
    .N      (N),
    .ACC_W  (ACC_W),
    .SCALE_W(SCALE_W),
    .SHIFT_W(SHIFT_W)
  ) quant_inst (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_acc   (mac_out_c),
    .in_valid (quant_in_valid),
    .in_ready (quant_in_ready),
    .scale    (scale),
    .shift    (shift_amt),
    .zero_pt  (zero_pt),
    .out_q    (quant_out_q),
    .out_valid(quant_out_valid),
    .out_ready(quant_out_ready)
  );

  // ── mem_arbiter instantiation ─────────────────────────────────────────────
  mem_arbiter #(
    .ADDR_W   (ADDR_W),
    .DATA_W   (DATA_W),
    .NUM_BANKS(NUM_BANKS)
  ) arb_inst (
    .clk          (clk),
    .rst_n        (rst_n),
    .req_a_valid  (req_a_valid),
    .req_a_ready  (req_a_ready),
    .req_a_addr   (req_a_addr),
    .req_a_we     (req_a_we),
    .req_a_wdata  (req_a_wdata),
    .req_a_rdata  (req_a_rdata),
    .req_a_rvld   (req_a_rvld),
    .req_b_valid  (req_b_valid),
    .req_b_ready  (req_b_ready),
    .req_b_addr   (req_b_addr),
    .req_b_we     (req_b_we),
    .req_b_wdata  (req_b_wdata),
    .req_b_rdata  (req_b_rdata),
    .req_b_rvld   (req_b_rvld),
    .sram_ce      (sram_ce),
    .sram_we      (sram_we),
    .sram_addr    (sram_addr),
    .sram_wdata   (sram_wdata),
    .sram_rdata   (sram_rdata),
    .bank_conflict(bank_conflict)
  );

  // ── Combinational control for sub-module interfaces ───────────────────────
  always_comb begin
    // Inactive defaults for all requestor and sub-module controls.
    req_a_valid     = 1'b0;
    req_a_addr      = '0;
    req_a_we        = 1'b0;
    req_a_wdata     = '0;
    req_b_valid     = 1'b0;
    req_b_addr      = '0;
    req_b_we        = 1'b0;
    req_b_wdata     = '0;
    mac_in_valid    = 1'b0;
    mac_out_ready   = 1'b0;
    quant_in_valid  = 1'b0;
    quant_out_ready = 1'b0;

    case (state)

      LOAD: begin
        req_a_valid = !w_done;
        req_a_addr  = weight_base_addr + ADDR_W'(w_cnt);
        req_b_valid = !a_done;
        req_b_addr  = act_base_addr + ADDR_W'(a_cnt);
      end

      COMPUTE: begin
        // Keep in_valid asserted until mac_array accepts (in_ready goes low).
        // mac_out_ready is forwarded from quant's in_ready so the mac→quant
        // handshake collapses into a single cycle when quant is idle.
        mac_in_valid   = 1'b1;
        mac_out_ready  = quant_in_ready;
        quant_in_valid = mac_out_valid;
      end

      QUANTIZE: begin
        // Release quant_unit as soon as it is done; result is latched this cycle.
        quant_out_ready = quant_out_valid;
      end

      STORE: begin
        req_a_valid = 1'b1;
        req_a_we    = 1'b1;
        req_a_addr  = out_base_addr + ADDR_W'(s_cnt);
        // Zero-extend INT8 result to DATA_W; $unsigned strips sign to silence lint.
        req_a_wdata = DATA_W'($unsigned(quant_result[si][sj]));
      end

      default: ;

    endcase
  end

  // ── Sequential FSM + datapath ─────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state  <= IDLE;
      done   <= 1'b0;
      w_cnt  <= '0;
      a_cnt  <= '0;
      s_cnt  <= '0;
      w_done <= 1'b0;
      a_done <= 1'b0;
      for (int i = 0; i < int'(N); i++)
        for (int j = 0; j < int'(N); j++) begin
          weight_tile [i][j] <= '0;
          act_tile    [i][j] <= '0;
          quant_result[i][j] <= '0;
        end
    end else begin
      done <= 1'b0;    // pulse: cleared every cycle; set only in DONE state

      case (state)

        // ── IDLE: wait for start pulse ──────────────────────────────────────
        IDLE: begin
          if (start) begin
            w_cnt  <= '0;
            a_cnt  <= '0;
            s_cnt  <= '0;
            w_done <= 1'b0;
            a_done <= 1'b0;
            state  <= LOAD;
          end
        end

        // ── LOAD: fetch N×N weights via req_a and activations via req_b ─────
        // Both requestors run concurrently; mem_arbiter arbitrates.
        // Transition once the last element of each tile has been accepted.
        LOAD: begin
          if (req_a_rvld && !w_done) begin
            weight_tile[wi][wj] <= req_a_rdata;
            if (w_cnt == CNT_W'(TILE_ELS - 1))
              w_done <= 1'b1;
            else
              w_cnt <= w_cnt + 1'b1;
          end

          if (req_b_rvld && !a_done) begin
            act_tile[ai][aj] <= req_b_rdata;
            if (a_cnt == CNT_W'(TILE_ELS - 1))
              a_done <= 1'b1;
            else
              a_cnt <= a_cnt + 1'b1;
          end

          // Check for completion: either already done from a prior cycle,
          // or completing this cycle on the last accepted element.
          if ((w_done || (req_a_rvld && !w_done && w_cnt == CNT_W'(TILE_ELS - 1))) &&
              (a_done || (req_b_rvld && !a_done && a_cnt == CNT_W'(TILE_ELS - 1))))
            state <= COMPUTE;
        end

        // ── COMPUTE: mac_array runs N cycles; forward result to quant_unit ──
        // in_valid stays asserted until mac accepts (in_ready drops).
        // Transition to QUANTIZE the cycle the mac→quant handshake fires.
        COMPUTE: begin
          if (mac_out_valid && quant_in_ready)
            state <= QUANTIZE;
        end

        // ── QUANTIZE: wait one cycle for quant_unit output ──────────────────
        // out_ready is combinationally tied to out_valid, so quant_unit is
        // released the same cycle we latch quant_result.
        QUANTIZE: begin
          if (quant_out_valid) begin
            for (int i = 0; i < int'(N); i++)
              for (int j = 0; j < int'(N); j++)
                quant_result[i][j] <= quant_out_q[i][j];
            s_cnt <= '0;
            state <= STORE;
          end
        end

        // ── STORE: write N×N INT8 results back to SRAM via req_a ────────────
        STORE: begin
          if (req_a_ready) begin
            if (s_cnt == CNT_W'(TILE_ELS - 1))
              state <= DONE;
            else
              s_cnt <= s_cnt + 1'b1;
          end
        end

        // ── DONE: pulse done for one cycle then return to idle ───────────────
        DONE: begin
          done  <= 1'b1;
          state <= IDLE;
        end

        default: state <= IDLE;

      endcase
    end
  end

endmodule

`default_nettype wire
