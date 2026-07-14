"""Make the repo root importable so ``src`` / ``ui`` resolve when running pytest.

The project uses absolute imports from the repo root (no installed package), so
tests need the root on ``sys.path`` regardless of the working directory pytest is
invoked from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
