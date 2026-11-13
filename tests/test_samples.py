import subprocess
import sys
from pathlib import Path


def test_demo_samples_match_prediction_file():
    """ONNX on every docs/samples/* file reproduces v2/results/review/test_predictions.csv (|dp| <= 0.01, same uncertain flag)."""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "v2/scripts/check_samples.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
