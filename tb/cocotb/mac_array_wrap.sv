// tb/cocotb/mac_array_wrap.sv
//
// Thin wrapper that flattens mac_array's unpacked 2-D ports to packed flat buses
// so CocoTB/Verilator can drive and sample them from Python.
//
// Packing convention (row-major):
//   element [i][j] sits at bits [(i*N+j)*W +: W]
//   [0][0] at LSBs, [N-1][N-1] at MSBs.

`default_nettype none

module mac_array_wrap #(
  parameter  int unsigned N         = 4,
  parameter  int unsigned DATA_TYPE = 0,
  localparam int unsigned DATA_W    = (DATA_TYPE == 0) ? 8  : 16,
  localparam int unsigned ACC_W     = (DATA_TYPE == 0) ? 32 :
                                      (DATA_TYPE == 1) ? 48 : 32
) (
  input  logic                        clk,
  input  logic                        rst_n,

  // Flat packed inputs (row-major; element [i][j] at [(i*N+j)*DATA_W +: DATA_W])
  input  logic [N*N*DATA_W-1:0]       flat_a,
  input  logic [N*N*DATA_W-1:0]       flat_b,
  input  logic                        in_valid,
  output logic                        in_ready,

  // Flat packed output (row-major; element [i][j] at [(i*N+j)*ACC_W +: ACC_W])
  output logic [N*N*ACC_W-1:0]        flat_c,
  output logic                        out_valid,
  input  logic                        out_ready
);

  // ── Internal unpacked 2-D signals ─────────────────────────────────────────
  logic signed [DATA_W-1:0] in_a  [N][N];
  logic signed [DATA_W-1:0] in_b  [N][N];
  logic signed [ACC_W-1:0]  out_c [N][N];

  // ── Unpack flat → 2-D (inputs) ────────────────────────────────────────────
  always_comb begin
    for (int i = 0; i < int'(N); i++)
      for (int j = 0; j < int'(N); j++) begin
        in_a[i][j] = signed'(flat_a[(i * int'(N) + j) * int'(DATA_W) +: DATA_W]);
        in_b[i][j] = signed'(flat_b[(i * int'(N) + j) * int'(DATA_W) +: DATA_W]);
      end
  end

  // ── Pack 2-D → flat (output) ──────────────────────────────────────────────
  always_comb begin
    for (int i = 0; i < int'(N); i++)
      for (int j = 0; j < int'(N); j++)
        flat_c[(i * int'(N) + j) * int'(ACC_W) +: ACC_W] = out_c[i][j];
  end

  // ── DUT instantiation ─────────────────────────────────────────────────────
  mac_array #(
    .N        (N),
    .DATA_TYPE(DATA_TYPE)
  ) u_mac_array (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_a     (in_a),
    .in_b     (in_b),
    .in_valid (in_valid),
    .in_ready (in_ready),
    .out_c    (out_c),
    .out_valid(out_valid),
    .out_ready(out_ready)
  );

endmodule

`default_nettype wire
