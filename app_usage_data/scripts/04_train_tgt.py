"""Thin wrapper that calls scripts/03 with model=tgt."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], "--model", "tgt"] + sys.argv[1:]
exec((ROOT / "03_train_gru.py").read_text())
