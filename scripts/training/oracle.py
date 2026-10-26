"""Legacy import compatibility; implementation: scripts.experiments.oracle.oracle."""

import sys
from scripts.experiments.oracle import oracle as _implementation

sys.modules[__name__] = _implementation
