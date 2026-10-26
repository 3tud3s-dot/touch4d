"""Legacy import compatibility; implementation: scripts.experiments.oracle.oracle_fit."""

import sys
from scripts.experiments.oracle import oracle_fit as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
