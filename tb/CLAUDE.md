# Testbench Directory
- Framework: CocoTB with Verilator backend
- Reference model is pure NumPy (no PyTorch dependency for basic runs)
- Scoreboard outputs mismatch records as dicts matching /docs/mismatch_schema.json
- One test file per accelerator operation (test_conv2d.py, test_matmul.py, etc.)