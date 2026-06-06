// tb/cocotb/quant_unit_wrap.sv
//
// Thin wrapper that flattens quant_unit's unpacked ports to packed flat buses
// so CocoTB/Verilator can drive and sample them from Python.
//
// Packing conventions:
//   2-D (acc, out_q): element [i][j] at bits [(i*N+j)*W +: W]  (row-major)
//   1-D (scale, shift, zero_pt): element [i] at bits [i*W +: W]

`default_nettype none

module quant_unit_wrap #(
  parameter  int unsigned N       = 4,
  parameter  int unsigned ACC_W   = 32,
  parameter  int unsigned SCALE_W = 16,
  parameter  int unsigned SHIFT_W = 5
) (
  input  logic                      clk,
  input  logic                      rst_n,

  // Flat accumulator input (row-major; element [i][j] at [(i*N+j)*ACC_W +: ACC_W])
  input  logic [N*N*ACC_W-1:0]      flat_acc,
  input  logic                      in_valid,
  output logic                      in_ready,

  // Flat per-channel quantization parameters (element [i] at [i*W +: W])
  input  logic [N*SCALE_W-1:0]      flat_scale,
  input  logic [N*SHIFT_W-1:0]      flat_shift,
  input  logic [N*8-1:0]            flat_zero_pt,

  // Flat quantized output (row-major; element [i][j] at [(i*N+j)*8 +: 8])
  output logic [N*N*8-1:0]          flat_q,
  output logic                      out_valid,
  input  logic                      out_ready
);

  // ── Internal unpacked signals ──────────────────────────────────────────────
  logic signed [ACC_W-1:0]   in_acc  [N][N];
  logic        [SCALE_W-1:0] scale   [N];
  logic        [SHIFT_W-1:0] shift   [N];
  logic signed [7:0]         zero_pt [N];
  logic signed [7:0]         out_q   [N][N];

  // ── Unpack flat → 2-D / 1-D (inputs) ──────────────────────────────────────
  always_comb begin
    for (int i = 0; i < int'(N); i++) begin
      for (int j = 0; j < int'(N); j++)
        in_acc[i][j] = signed'(flat_acc[(i * int'(N) + j) * int'(ACC_W) +: ACC_W]);
      scale  [i] = flat_scale  [i * int'(SCALE_W) +: SCALE_W];
      shift  [i] = flat_shift  [i * int'(SHIFT_W) +: SHIFT_W];
      zero_pt[i] = signed'(flat_zero_pt[i * 8 +: 8]);
    end
  end

  // ── Pack 2-D → flat (output) ───────────────────────────────────────────────
  always_comb begin
    for (int i = 0; i < int'(N); i++)
      for (int j = 0; j < int'(N); j++)
        flat_q[(i * int'(N) + j) * 8 +: 8] = out_q[i][j];
  end

  // ── DUT instantiation ──────────────────────────────────────────────────────
  quant_unit #(
    .N      (N),
    .ACC_W  (ACC_W),
    .SCALE_W(SCALE_W),
    .SHIFT_W(SHIFT_W)
  ) u_quant_unit (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_acc   (in_acc),
    .in_valid (in_valid),
    .in_ready (in_ready),
    .scale    (scale),
    .shift    (shift),
    .zero_pt  (zero_pt),
    .out_q    (out_q),
    .out_valid(out_valid),
    .out_ready(out_ready)
  );

endmodule

`default_nettype wire
