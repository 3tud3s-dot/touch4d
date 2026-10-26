"""Legacy import compatibility; implementation: scripts.experiments.fixed_window.fixed_fit."""

import sys
from scripts.experiments.fixed_window import fixed_fit as _implementation

sys.modules[__name__] = _implementation
