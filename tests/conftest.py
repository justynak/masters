import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The legacy code lives in test/ (rTransform extension, imageshow.py) and
# imports `ui` from the repo root; make both importable in tests.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "test"))
