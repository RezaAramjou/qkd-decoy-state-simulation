# -*- coding: utf-8 -*-
"""
Parameter validation helper functions.

This module provides a robust boolean parsing utility.
"""
from typing import Any
import numbers
import numpy as np

# A custom exception should be defined in your project's exception hierarchy.
# For demonstration purposes, we'll define a placeholder here.
class ParameterValidationError(ValueError):
    """Raised when a parameter has an invalid value."""
    pass

__all__ = ["parse_bool"]

# Canonical true/false string tokens (lowercased for case-insensitive matching)
_TRUE_TOKENS = {"true", "1", "yes", "y", "on"}
_FALSE_TOKENS = {"false", "0", "no", "n", "off"}


def parse_bool(x: Any) -> bool:
    """
    Strictly parses a value to a boolean, handling common representations.

    This function is designed to be explicit and defensive, raising an error
    for any input that is not clearly a boolean equivalent.

    Args:
        x: The value to parse.

    Returns:
        The boolean representation of the input value.

    Raises:
        ParameterValidationError: If the input value is not a recognized
            boolean representation or cannot be safely coerced.

    Accepted inputs:
      - bool and numpy.bool_ (returned as-is)
      - Integer-like types (e.g., int, numpy.integer) with values 0 or 1.
      - Float-like types (e.g., float, numpy.floating) with values 0.0 or 1.0.
      - str or bytes representing common true/false tokens (case-insensitive).
    """
    # 1. Handle native bool and numpy.bool_ types directly for performance.
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    # 2. Handle integer-like types (covers Python int and NumPy integer scalars).
    if isinstance(x, numbers.Integral):
        if x == 1:
            return True
        if x == 0:
            return False
        raise ParameterValidationError(
            f"Invalid integer for boolean parsing: {x!r}. Only 0 or 1 are accepted."
        )

    # 3. Handle float-like types, accepting only exact 0.0 or 1.0.
    # This prevents ambiguity with values like 0.1 or 2.0.
    if isinstance(x, numbers.Real) and not isinstance(x, numbers.Integral):
        if x == 1.0:
            return True
        if x == 0.0:
            return False
        raise ParameterValidationError(
            f"Invalid float for boolean parsing: {x!r}. Only exact 0.0 or 1.0 are accepted."
        )

    # 4. Handle bytes by decoding them to a string.
    # Assumes UTF-8, which is a standard for config and env vars.
    if isinstance(x, bytes):
        try:
            s = x.decode("utf-8")
        except UnicodeDecodeError:
            raise ParameterValidationError(
                f"Cannot parse boolean from bytes with invalid UTF-8 sequence: {x!r}"
            )
    elif isinstance(x, str):
        s = x
    else:
        # 6. Reject all other types.
        raise ParameterValidationError(
            f"Cannot coerce type '{type(x).__name__}' to bool: {x!r}"
        )

    # 5. Handle strings after normalizing (strip whitespace, convert to lowercase).
    s_normalized = s.strip().lower()
    if s_normalized in _TRUE_TOKENS:
        return True
    if s_normalized in _FALSE_TOKENS:
        return False

    raise ParameterValidationError(
        f"Invalid string for boolean parsing: {s!r}. "
        f"Recognized values are (case-insensitive): {sorted(_TRUE_TOKENS | _FALSE_TOKENS)}"
    )

