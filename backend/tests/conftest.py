"""Pytest configuration: make the ``app`` package importable no matter where
pytest is invoked from (root or backend/)."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
