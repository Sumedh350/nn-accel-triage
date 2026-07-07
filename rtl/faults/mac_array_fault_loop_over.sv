// rtl/mac_array.sv
//
// Parameterized N×N MAC array — computes one tile of C = A × B per transaction.
//
// DATA_TYPE selects the arithmetic mode:
//   0 = INT8   : 8-bit signed inputs,  32-bit signed accumulator
//   1 = INT16  : 16-bit signed inputs, 48-bit signed accumulator
//   2 = FP16   : port widths match INT16, computation stubbed (out_c = 0)
//
// Handshake (AXI-style valid/ready):
//   Input  : in_a and in_b held stable while in_valid=1 until in_ready=1.
//   Output : out_c held stable while out_valid=1 until out_ready=1.
//
// Latency: N clock cycles from the accepted input handshake to out_valid.
// Throughput: one N×N tile per N+1 cycles (COMPUTE + one IDLE gap).

`default_nettype none

module mac_array #(
  parameter  int unsigned N         = 4,    // array dimension (N×N PEs)
  parameter  int unsigned DATA_TYPE = 0,    // 0=INT8, 1=INT16, 2=FP16(stub)
  // Derived widths — declared as localparams to prevent external override.
  localparam int unsigned DATA_W    = (DATA_TYPE == 0) ? 8  : 16,
  localparam int unsigned ACC_W     = (DATA_TYPE == 0) ? 32 :
                                      (DATA_TYPE == 1) ? 48 : 32
) (
  input  logic                      clk,
  input  logic                      rst_n,    // synchronous reset, active-low

  // Activation tile A and weight tile B — must arrive in the same handshake.
  input  logic signed [DATA_W-1:0]  in_a [N][N],
  input  logic signed [DATA_W-1:0]  in_b [N][N],
  input  logic                      in_valid,
  output logic                      in_ready,

  // Result tile C = A × B.
  output logic signed [ACC_W-1:0]   out_c [N][N],
  output logic                      out_valid,
  input  logic                      out_ready
);

  // ── Elaboration-time checks ──────────────────────────────────────────────
  // These fire once during simulation start-up before time 0.
  initial begin
    if (N < 1)
      $fatal(1, "mac_array: N must be >= 1, got %0d.", N);
    if (DATA_TYPE > 2)
      $fatal(1, "mac_array: DATA_TYPE=%0d is unsupported (0=INT8, 1=INT16, 2=FP16).",
             DATA_TYPE);
    if (DATA_TYPE == 2)
      $warning("mac_array: FP16 (DATA_TYPE=2) is stubbed — out_c will always be zero.");
  end

  // ── FSM state encoding ───────────────────────────────────────────────────
  typedef enum logic [1:0] {
    IDLE    = 2'b00,
    COMPUTE = 2'b01,
    DONE    = 2'b10
  } state_t;

  state_t state;

  // ── Storage registers ────────────────────────────────────────────────────
  localparam int unsigned STEP_W = (N > 1) ? $clog2(N) : 1;

  logic signed [DATA_W-1:0]  A_reg [N][N];  // latched activation tile
  logic signed [DATA_W-1:0]  B_reg [N][N];  // latched weight tile
  logic signed [ACC_W-1:0]   acc   [N][N];  // per-PE signed accumulator
  logic [STEP_W-1:0]          step;          // inner-product index: 0 .. N-1

  // ── Combinational products for the current step ──────────────────────────
  // Both operands are sign-extended to ACC_W before the multiply so the full
  // product fits without overflow and the arithmetic stays unambiguously signed.
  //   INT8 : sign-extend 8→32, multiply in 32-bit signed → fits easily
  //   INT16: sign-extend 16→48, multiply in 48-bit signed → fits easily
  logic signed [ACC_W-1:0] pe_prod [N][N];

  always_comb begin
    for (int i = 0; i < int'(N); i++)
      for (int j = 0; j < int'(N); j++)
        pe_prod[i][j] = ACC_W'(A_reg[i][step]) * ACC_W'(B_reg[step][j]);
  end

  // ── Handshake control ────────────────────────────────────────────────────
  // Accept new data only when idle.
  assign in_ready = (state == IDLE);

  // ── Main FSM + datapath ──────────────────────────────────────────────────
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      step      <= 0;
      out_valid <= 1'b0;
      for (int i = 0; i < int'(N); i++)
        for (int j = 0; j < int'(N); j++) begin
          A_reg[i][j] <= '0;
          B_reg[i][j] <= '0;
          acc[i][j]   <= '0;
        end
    end else begin
      case (state)

        // ── IDLE: wait for a new tile pair ─────────────────────────────────
        IDLE: begin
          out_valid <= 1'b0;
          if (in_valid) begin
            for (int i = 0; i < int'(N); i++)
              for (int j = 0; j < int'(N); j++) begin
                A_reg[i][j] <= in_a[i][j];
                B_reg[i][j] <= in_b[i][j];
                acc[i][j]   <= '0;
              end
            step  <= 0;
            state <= COMPUTE;
          end
        end

        // ── COMPUTE: N cycles of parallel MAC across all N×N PEs ───────────
        COMPUTE: begin
          for (int i = 0; i <= int'(N); i++)
            for (int j = 0; j < int'(N); j++)
              acc[i][j] <= acc[i][j] + pe_prod[i][j];
          if (step == STEP_W'(N - 1)) begin
            out_valid <= 1'b1;
            state     <= DONE;
          end else
            step <= step + 1;
        end

        // ── DONE: hold result until downstream accepts ──────────────────────
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

  // ── Output mux ───────────────────────────────────────────────────────────
  generate
    if (DATA_TYPE == 2) begin : g_fp16_stub
      // FP16 arithmetic is not yet implemented; drive zero so the
      // valid/ready handshake remains functional for stub testing.
      always_comb begin
        for (int i = 0; i < int'(N); i++)
          for (int j = 0; j < int'(N); j++)
            out_c[i][j] = '0;
      end
    end else begin : g_int_out
      // Integer modes: expose the signed accumulator directly.
      always_comb begin
        for (int i = 0; i < int'(N); i++)
          for (int j = 0; j < int'(N); j++)
            out_c[i][j] = acc[i][j];
      end
    end
  endgenerate

endmodule

`default_nettype wire
