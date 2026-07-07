// rtl/quant_unit.sv
//
// Per-channel INT8 asymmetric quantization unit.
//
// Converts an N×N tile of signed accumulators (from mac_array) to INT8 via
// per-channel (per-output-row) asymmetric quantization:
//
//   q[i][j] = clamp( ((acc[i][j] * scale[i]) >>> shift[i]) + zero_pt[i],
//                    -128, 127 )
//
// "Channel" = output row index i; each row corresponds to one output neuron.
// scale[i] is unsigned; shift[i] is an arithmetic right-shift; zero_pt[i] is
// a signed INT8 bias.  All three must remain stable while state != IDLE.
//
// Handshake (AXI-style valid/ready):
//   Input  : in_acc held stable while in_valid=1 until in_ready=1.
//   Output : out_q  held stable while out_valid=1 until out_ready=1.
//
// Latency:    1 clock cycle (accepted input → out_valid asserted)
// Throughput: 1 tile per 2 cycles minimum (IDLE→DONE→IDLE)
//
// Parameters
//   N       – tile dimension; must match the connected mac_array N
//   ACC_W   – accumulator width in bits (32 for INT8 mac mode, 48 for INT16)
//   SCALE_W – per-channel scale multiplier width, unsigned (default 16)
//   SHIFT_W – right-shift amount width; range 0..2^SHIFT_W-1 (default 5 → 0..31)

`default_nettype none

module quant_unit #(
  parameter  int unsigned N       = 4,
  parameter  int unsigned ACC_W   = 32,
  parameter  int unsigned SCALE_W = 16,
  parameter  int unsigned SHIFT_W = 5
) (
  input  logic                      clk,
  input  logic                      rst_n,       // synchronous reset, active-low

  // ── Input tile (from mac_array) ────────────────────────────────────────────
  input  logic signed [ACC_W-1:0]   in_acc  [N][N],
  input  logic                      in_valid,
  output logic                      in_ready,

  // ── Per-channel quantization parameters ────────────────────────────────────
  input  logic        [SCALE_W-1:0] scale   [N],   // unsigned fixed-point multiplier
  input  logic        [SHIFT_W-1:0] shift   [N],   // arithmetic right-shift amount
  input  logic signed [7:0]         zero_pt [N],   // INT8 zero-point (bias)

  // ── Quantized output tile ───────────────────────────────────────────────────
  output logic signed [7:0]         out_q   [N][N],
  output logic                      out_valid,
  input  logic                      out_ready
);

  // ── Elaboration-time checks ─────────────────────────────────────────────────
  // These fire during simulation start-up before time 0.
  initial begin
    if (N < 1)       $fatal(1, "quant_unit: N must be >= 1 (got %0d).",       N);
    if (SCALE_W < 1) $fatal(1, "quant_unit: SCALE_W must be >= 1 (got %0d).", SCALE_W);
    if (SHIFT_W < 1) $fatal(1, "quant_unit: SHIFT_W must be >= 1 (got %0d).", SHIFT_W);
  end

  // ── FSM state encoding ───────────────────────────────────────────────────────
  typedef enum logic { IDLE = 1'b0, DONE = 1'b1 } state_t;
  state_t state;

  // ── Accumulator latch ────────────────────────────────────────────────────────
  logic signed [ACC_W-1:0] acc_reg [N][N];

  // ── Combinational quantization ───────────────────────────────────────────────
  // Product needs ACC_W + SCALE_W + 1 signed bits for full precision:
  //   max |acc| = 2^(ACC_W-1), max scale = 2^SCALE_W - 1
  //   max |product| < 2^(ACC_W-1 + SCALE_W) which fits in ACC_W+SCALE_W signed bits;
  //   the +1 guard bit ensures the sign is unambiguous after zero-extending scale.
  localparam int unsigned PROD_W = ACC_W + SCALE_W + 1;

  // Signed comparison constants — sign-extended to PROD_W width in relational ops.
  localparam int INT8_MAX =  127;
  localparam int INT8_MIN = -128;

  logic signed [PROD_W-1:0] q_prod  [N][N];   // acc × scale  (full precision)
  logic signed [PROD_W-1:0] q_shift [N][N];   // >>> shift[i] (arithmetic)
  logic signed [PROD_W-1:0] q_bias  [N][N];   // + zero_pt[i]

  always_comb begin
    for (int i = 0; i < int'(N); i++) begin
      for (int j = 0; j < int'(N); j++) begin
        // Sign × unsigned: zero-extend scale by one bit so the leading 0
        // prevents $signed from treating the MSB as a negative sign.
        q_prod[i][j]  = PROD_W'($signed(acc_reg[i][j])) *
                        PROD_W'($signed({1'b0, scale[i]}));
        // Arithmetic right-shift (>>> on a signed operand preserves sign).
        q_shift[i][j] = $signed(q_prod[i][j]) >>> 1;
        // Add per-channel zero-point (INT8 value, sign-extended to PROD_W).
        q_bias[i][j]  = q_shift[i][j] + PROD_W'($signed(zero_pt[i]));
        // Saturate to INT8 range [-128, 127].
        if      (q_bias[i][j] > $signed(PROD_W'(INT8_MAX))) out_q[i][j] = 8'h7F;  // +127
        else if (q_bias[i][j] < $signed(PROD_W'(INT8_MIN))) out_q[i][j] = 8'h80; // -128
        else                              out_q[i][j] = $signed(q_bias[i][j][7:0]);
      end
    end
  end

  // ── Handshake control ────────────────────────────────────────────────────────
  // Accept new data only when idle.
  assign in_ready = (state == IDLE);

  // ── Main FSM + datapath ───────────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      out_valid <= 1'b0;
      for (int i = 0; i < int'(N); i++)
        for (int j = 0; j < int'(N); j++)
          acc_reg[i][j] <= '0;
    end else begin
      case (state)

        // ── IDLE: wait for a new accumulator tile ─────────────────────────────
        IDLE: begin
          if (in_valid) begin
            for (int i = 0; i < int'(N); i++)
              for (int j = 0; j < int'(N); j++)
                acc_reg[i][j] <= in_acc[i][j];
            out_valid <= 1'b1;
            state     <= DONE;
          end
        end

        // ── DONE: hold quantized tile until downstream accepts ─────────────────
        DONE: begin
          if (out_ready) begin
            out_valid <= 1'b0;
            state     <= IDLE;
          end
        end

        default: state <= IDLE;

      endcase
    end
  end

endmodule

`default_nettype wire
