import sys
from pathlib import Path

# Ensure tb/reference_model/ is on sys.path so test files can import mac_ref / quant_ref
# directly regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).parent))
