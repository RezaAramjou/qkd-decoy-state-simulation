# -*- coding: utf-8 -*-
"""
Shared constants, numerical tolerances, and utility functions for the QKD
simulation framework.

This module provides a centralized, well-documented, and robust set of default
values critical for the simulation's numerical stability, security proof
implementations, and reproducibility.

The constants are grouped into the following categories:
- RNG & Solver: For random number generation and linear programming.
- Security & Proof: Parameters directly influencing security calculations.
- Numeric Tolerances: Epsilon-scaled values for safe floating-point comparisons.

This module also provides helper functions to apply these constants correctly,
for instance, by validating probability vectors or clamping probabilities to
avoid numerical errors in entropy calculations.

:version: 2.0.0
:tested-on: Python 3.9+, numpy 1.21+, scipy 1.7+
:provenance: Values are based on common practices in QKD literature and are
             chosen for double-precision (64-bit float) computations. See
             individual constant docstrings for specific rationale.
"""

__author__ = "Quantum Communications Research Group"
__version__ = "2.0.0"

# --- Public API ---
__all__ = [
    # --- Constants ---
    "MAX_SEED_INT_PCG64",
    "MAX_SEED_INT_UINT32",
    "LP_SOLVER_METHODS",
    "MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST",
    "EPS",
    "NUMERIC_ABS_TOL",
    "NUMERIC_REL_TOL",
    "Y1_SAFE_THRESHOLD",
    "ENTROPY_PROB_CLAMP",
    "PROB_SUM_TOL",
    "DEFAULT_POISSON_TAIL_THRESHOLD",
    "LP_CONSTRAINT_VIOLATION_TOL",
    # --- Helper Functions ---
    "is_close",
    "clamp_probability",
    "clamp_probabilities",
    "is_prob_vector",
    "validate_lp_solver",
    "as_dict",
]

import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
# Defined before constants so they can be used in potential validations.

def is_close(a: float, b: float) -> bool:
    """
    Checks if two floats `a` and `b` are close, using the module's default
    relative and absolute tolerances.

    This provides a consistent comparison method across the simulation,
    analogous to `math.isclose`.

    :param a: The first float.
    :param b: The second float.
    :return: True if the values are close, False otherwise.
    """
    return abs(a - b) <= max(NUMERIC_REL_TOL * max(abs(a), abs(b)), NUMERIC_ABS_TOL)


def clamp_probability(p: float) -> float:
    """
    Clamps a single probability value to the safe range [clamp, 1-clamp].

    This is crucial for preventing `log(0)` errors in entropy calculations
    without introducing significant bias.

    :param p: The probability value to clamp.
    :return: The clamped probability.
    """
    return max(ENTROPY_PROB_CLAMP, min(p, 1.0 - ENTROPY_PROB_CLAMP))


def clamp_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Vectorized version of `clamp_probability` for NumPy arrays.

    :param probs: A NumPy array of probability values.
    :return: A new NumPy array with clamped probabilities.
    """
    return np.clip(probs, ENTROPY_PROB_CLAMP, 1.0 - ENTROPY_PROB_CLAMP)


def is_prob_vector(p: np.ndarray) -> bool:
    """
    Validates if a NumPy array represents a valid probability vector.

    Checks for:
    1. Non-negativity and finiteness of all elements.
    2. Sum of elements is 1.0 within the `PROB_SUM_TOL` tolerance.

    :param p: The NumPy array to validate.
    :return: True if it is a valid probability vector, False otherwise.
    """
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p) & (p >= 0.0)):
        return False
    return abs(p.sum() - 1.0) <= PROB_SUM_TOL


def validate_lp_solver(method: str) -> str:
    """
    Validates if a given LP solver method is supported by this framework.

    Note: This checks against a predefined tuple of known-good solvers. A more
    robust implementation could introspect the installed SciPy version, but
    that adds a heavier dependency check.

    :param method: The name of the LP solver (e.g., 'highs').
    :return: The method name if it is valid.
    :raises ValueError: If the method is not in the supported list.
    """
    if method not in LP_SOLVER_METHODS:
        raise ValueError(
            f"LP solver '{method}' is not supported. "
            f"Available methods: {LP_SOLVER_METHODS}"
        )
    return method


def as_dict() -> Dict[str, Any]:
    """
    Returns all public constants from this module as a dictionary.

    This is useful for logging, metadata storage, and ensuring reproducibility
    of simulation runs. It excludes functions and private members.

    :return: A dictionary mapping constant names to their values.
    """
    return {
        key: value
        for key, value in globals().items()
        if key in __all__ and isinstance(value, (int, float, str, tuple))
    }


# ==============================================================================
# --- RNG & Solver Constants ---
# ==============================================================================

MAX_SEED_INT_PCG64: int = (1 << 63) - 1
"""
The maximum integer for seeding NumPy's default PCG64 RNG. This corresponds
to the maximum positive signed 64-bit integer. Using seeds within the range
[0, 2**63 - 1] is recommended for compatibility.
"""

MAX_SEED_INT_UINT32: int = (1 << 32) - 1
"""
The maximum integer for RNGs that expect a 32-bit unsigned integer seed.
Provided for compatibility with older or alternative RNG backends.
"""

LP_SOLVER_METHODS: Tuple[str, ...] = ("highs", "highs-ds")
"""
A tuple of recommended Linear Programming solver methods available in SciPy.
The order implies preference, with 'highs' being the modern default.
Using a tuple makes this constant immutable, preventing accidental modification.
"""

# ==============================================================================
# --- Security & Proof Constants ---
# ==============================================================================

MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST: int = 10
"""
Minimum number of successful detection events in the Z-basis with intensity 1
(S_z,1) required to perform a valid phase error rate estimation. Below this
threshold, the statistics are considered too poor for the Chernoff bound-based
estimation formulas to be reliable. This is a heuristic value based on
finite-key security analysis principles.
"""

DEFAULT_POISSON_TAIL_THRESHOLD: float = 1e-9
"""
Default probability mass to leave in the tail when truncating a Poisson
distribution. For example, a value of 1e-9 means the photon number cutoff N
is chosen such that P(n > N) <= 1e-9. This ensures that the vast majority
of the distribution's probability mass is included in decoy state analysis.
"""

# --- Runtime validation for the threshold ---
if not (0.0 < DEFAULT_POISSON_TAIL_THRESHOLD < 1.0):
    raise ValueError("DEFAULT_POISSON_TAIL_THRESHOLD must be in the interval (0, 1).")


# ==============================================================================
# --- Numeric Tolerances ---
# ==============================================================================
# These tolerances are derived from or compared against machine epsilon for
# double-precision floating-point numbers to ensure they are robust and
# well-defined.

EPS: float = np.finfo(float).eps
"""
Machine epsilon for double-precision (64-bit) floats, approximately 2.22e-16.
This is the smallest number such that `1.0 + EPS != 1.0`. It serves as a
fundamental unit for other tolerance values.
"""

NUMERIC_ABS_TOL: float = 1e-12
"""
Absolute tolerance for general-purpose floating-point comparisons.
Chosen as a reasonably small number, roughly 1e4 * EPS, that is robust
against typical accumulation of floating-point errors in QKD calculations.
"""

NUMERIC_REL_TOL: float = 1e-9
"""
Relative tolerance for general-purpose floating-point comparisons.
Used when comparing numbers of large magnitude, where absolute tolerance
would be inappropriate.
"""

_Y1_EPS_MULTIPLIER: float = 100.0
Y1_SAFE_THRESHOLD: float = max(1e-12, _Y1_EPS_MULTIPLIER * EPS)
"""
A safe, small, positive threshold for single-photon yields (Y1). It is used
to prevent division-by-zero or other numerical instabilities when Y1 is
estimated to be zero or negative (an unphysical result of statistical
fluctuations). The value is set to be robustly above machine epsilon.
The multiplier `_Y1_EPS_MULTIPLIER` provides a safety margin.
"""

ENTROPY_PROB_CLAMP: float = 10.0 * EPS
"""
A small positive value used to clamp probabilities away from exactly 0 or 1
before calculating entropy (e.g., `p*log2(p)`). This prevents `log(0)` which
results in NaN. The value is chosen to be a small multiple of machine epsilon
to have a negligible impact on the final entropy calculation while ensuring
numerical stability.
"""

PROB_SUM_TOL: float = 1e-8
"""
Tolerance for checking if a vector of probabilities sums to 1.0.
This value is larger than `NUMERIC_ABS_TOL` because summing many small,
imprecise floating-point numbers can lead to a larger accumulated error.
It must be chosen carefully to catch legitimate normalization errors without
failing due to benign floating-point inaccuracies.
"""

LP_CONSTRAINT_VIOLATION_TOL: float = 1e-9
"""
Tolerance for checking violations of Linear Programming constraints after a
solution is found. This should typically be aligned with the tolerance
parameter of the LP solver itself (e.g., SciPy's `linprog` `tol` parameter)
to ensure consistency.
"""
