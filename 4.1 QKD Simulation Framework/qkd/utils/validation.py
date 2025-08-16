# -*- coding: utf-8 -*-
"""
Parameter validation helper functions.

Moved from the monolithic script. No logic changes.
"""
from typing import Any
import numpy as np
from ..exceptions import ParameterValidationError

__all__ = ["_parse_bool"]


def _parse_bool(x: Any) -> bool:
    """Strictly parses a value to a boolean, handling common string representations."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        if x in (0, 1):
            return bool(x)
        raise ParameterValidationError(f"Invalid boolean int value: {x}")
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
        raise ParameterValidationError(f"Invalid boolean string value: {x!r}")
    raise ParameterValidationError(f"Cannot coerce parameter to bool: {x!r}")
