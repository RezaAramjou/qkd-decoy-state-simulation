# -*- coding: utf-8 -*-
"""
Numerical and statistical helper functions for QKD simulations.

This module provides robust, validated, and well-documented implementations of
common mathematical functions used in quantum information and statistics,
particularly for analyzing experimental results from QKD systems.

Functions are designed to be scalar, pure, and thread-safe. For performance-
critical applications involving large datasets (e.g., decoy-state analysis),
callers should consider implementing vectorized versions of these functions.

To see detailed warnings and debug information from this module, configure the
Python logger. For example:
    import logging
    logging.basicConfig(level=logging.DEBUG)

.. warning::
    This module's filename (`math.py`) can shadow the built-in Python `math`
    module if imported improperly. It is strongly recommended to use an absolute
    import path, e.g., `from qkd.utils.math import binary_entropy`.

This code requires Python >= 3.8 for typing features like `Literal`.
"""
from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Tuple, Any

# Set up a logger for this module to report numeric warnings.
logger = logging.getLogger(__name__)

# A very small probability, used to guard against floating point errors in logs.
_MIN_ALPHA = 1e-300

# Declare np with type Any to accommodate the case where it's not installed.
np: Any

# Attempt to import numpy for type hinting and robust type checking.
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None
    _NUMPY_AVAILABLE = False

# Lazy import of scipy only where needed.
_SCIPY_AVAILABLE = False
_SCIPY_VERSION = "N/A"

# Declare _scipy_beta with type Any to handle the optional import.
_scipy_beta: Any

try:
    from scipy import __version__ as _scipy_version_str
    from scipy.stats import beta as _scipy_beta  # type: ignore
    _SCIPY_AVAILABLE = True
    _SCIPY_VERSION = _scipy_version_str
except ImportError:  # pragma: no cover
    _scipy_beta = None

from ..constants import ENTROPY_PROB_CLAMP

__all__ = [
    "binary_entropy",
    "hoeffding_bounds",
    "clopper_pearson_bounds",
    "ConfidenceInterval",
    "IntervalMethod",
]


class IntervalMethod(Enum):
    """Enumeration for the statistical methods used to compute intervals."""
    HOEFFDING = "hoeffding"
    CLOPPER_PEARSON = "clopper-pearson"


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Represents a confidence interval with metadata.

    This structure provides a clear, self-documenting return type.

    Attributes:
        lower: The lower bound of the interval.
        upper: The upper bound of the interval.
        alpha: The total significance level (failure probability) supplied by the user.
        method: The name of the method used to compute the interval.
    """
    lower: float
    upper: float
    alpha: float
    method: IntervalMethod

    def as_tuple(self) -> Tuple[float, float]:
        """Returns the interval as a standard (lower, upper) tuple."""
        return self.lower, self.upper


def _validate_k_n(k: int | float, n: int | float) -> Tuple[int, int]:
    """
    Internal helper to validate and coerce k and n for binomial trials.
    It accepts integer-like floats and returns integer types.
    """
    try:
        k_int, n_int = int(k), int(n)
        if k_int != k or n_int != n:
            raise ValueError("Non-integer values provided for k or n.")
    except (ValueError, TypeError):
        raise ValueError(f"Inputs 'k' and 'n' must be integers or integer-like, but got k={k}, n={n}.")

    if not isinstance(n_int, numbers.Integral) or n_int < 0:
        raise ValueError(f"Number of trials 'n' must be a non-negative integer, but got {n}.")
    if not isinstance(k_int, numbers.Integral) or not (0 <= k_int <= n_int):
        raise ValueError(f"Number of successes 'k' must be an integer satisfying 0 <= k <= n, but got k={k}, n={n}.")
    return k_int, n_int


def binary_entropy(p_err: float, clamp: bool = True) -> float:
    """
    Calculates the binary Shannon entropy h(p) in bits.

    The formula is: h(p) = -p * log2(p) - (1-p) * log2(1-p).
    This function is pure, thread-safe, and has O(1) complexity.

    Args:
        p_err: The probability of error, a float in the range [0.0, 1.0].
        clamp: If True (default), clamps interior probabilities to the interval
               `[ENTROPY_PROB_CLAMP, 1 - ENTROPY_PROB_CLAMP]` before the
               logarithm to prevent `math.log(0)` errors and ensure numerical
               stability. This does not affect exact 0.0 or 1.0 inputs.

    Returns:
        The binary entropy in bits. The result is not rounded.

    Raises:
        ValueError: If `p_err` is not a finite number or is outside [0, 1].
    """
    p = float(p_err)
    if not math.isfinite(p):
        raise ValueError("Input probability 'p_err' must be a finite number.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("Input probability 'p_err' must be in [0, 1].")

    if p == 0.0 or p == 1.0:
        return 0.0

    if clamp:
        p = max(ENTROPY_PROB_CLAMP, min(p, 1.0 - ENTROPY_PROB_CLAMP))

    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def hoeffding_bounds(
    k: int, n: int, alpha: float, side: Literal["two-sided", "upper", "lower"] = "two-sided"
) -> ConfidenceInterval:
    """
    Calculates the Hoeffding (additive Chernoff) confidence interval.

    Reference: Hoeffding, W. (1963). "Probability inequalities for sums of
    bounded random variables." Journal of the American Statistical Association.
    DOI: 10.1080/01621459.1963.10500830

    Args:
        k: The number of successes (a non-negative integer).
        n: The total number of trials (a positive integer).
        alpha: The total failure probability (significance level), a float.
        side: The type of interval: "two-sided", "upper", or "lower".

    Returns:
        A `ConfidenceInterval` object.

    Raises:
        ValueError: For invalid inputs.
    """
    if not (_MIN_ALPHA < alpha < 1.0):
        raise ValueError(f"Significance level 'alpha' must be in ({_MIN_ALPHA}, 1), but got {alpha}.")
    if side not in ("two-sided", "upper", "lower"):
        raise ValueError("Parameter 'side' must be one of 'two-sided', 'upper', or 'lower'.")
    
    k, n = _validate_k_n(k, n)
    if n == 0:
        # This check is specific to Hoeffding, as n=0 is valid for Clopper-Pearson.
        raise ValueError("Number of trials 'n' must be positive for Hoeffding bounds.")

    p_hat = float(k) / n
    log_arg = (2.0 if side == "two-sided" else 1.0) / alpha
    delta = math.sqrt(math.log(log_arg) / (2.0 * n))

    lower = max(0.0, p_hat - delta) if side != "upper" else 0.0
    upper = min(1.0, p_hat + delta) if side != "lower" else 1.0

    return ConfidenceInterval(lower=lower, upper=upper, alpha=alpha, method=IntervalMethod.HOEFFDING)


def clopper_pearson_bounds(
    k: int, n: int, alpha: float, side: Literal["two-sided", "upper", "lower"] = "two-sided"
) -> ConfidenceInterval:
    """
    Calculates the exact Clopper-Pearson confidence interval using the Beta distribution.

    Reference: Clopper, C. J., & Pearson, E. S. (1934). "The use of confidence
    or fiducial limits illustrated in the case of the binomial." Biometrika.
    DOI: 10.2307/2331986

    For one-sided intervals ('upper' or 'lower'), `alpha` is the one-sided
    significance level. For 'two-sided', `alpha` is the total significance,
    which is split between the two tails.

    Note: Example outputs may vary slightly across different `scipy` versions.

    Args:
        k: The number of successes (non-negative integer).
        n: The total number of trials (non-negative integer).
        alpha: The total failure probability (significance level), a float.
        side: The type of interval: "two-sided", "upper", or "lower".

    Returns:
        A `ConfidenceInterval` object.

    Raises:
        ValueError: For invalid inputs.
        ImportError: If `scipy` is not installed.
    """
    if not (_MIN_ALPHA < alpha < 1.0):
        raise ValueError(f"Significance level 'alpha' must be in ({_MIN_ALPHA}, 1), but got {alpha}.")
    if side not in ("two-sided", "upper", "lower"):
        raise ValueError("Parameter 'side' must be one of 'two-sided', 'upper', or 'lower'.")
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "The 'scipy' library is required for clopper_pearson_bounds. "
            "Please install it (`pip install scipy`) or use hoeffding_bounds."
        )
    k, n = _validate_k_n(k, n)

    if n == 0:
        return ConfidenceInterval(0.0, 1.0, alpha=alpha, method=IntervalMethod.CLOPPER_PEARSON)

    ppf_alpha = alpha if side != "two-sided" else alpha / 2.0

    # Lower bound calculation
    if k == 0 or side == "upper":
        lower = 0.0
    else:
        lower = _scipy_beta.ppf(ppf_alpha, k, n - k + 1)

    # Upper bound calculation
    if k == n or side == "lower":
        upper = 1.0
    else:
        upper = _scipy_beta.ppf(1.0 - ppf_alpha, k + 1, n - k)

    # Defensively sanitize results from scipy, ensuring they are scalar floats.
    try:
        lower_f = float(lower)
        if math.isnan(lower_f):
            logger.debug(f"scipy.beta.ppf returned NaN for lower bound (k={k}, n={n}). Defaulting to 0.0.")
            lower_f = 0.0
    except (TypeError, ValueError):
        logger.warning(f"scipy.beta.ppf returned a non-scalar or non-numeric value for lower bound: {lower}")
        lower_f = 0.0
        
    try:
        upper_f = float(upper)
        if math.isnan(upper_f):
            logger.debug(f"scipy.beta.ppf returned NaN for upper bound (k={k}, n={n}). Defaulting to 1.0.")
            upper_f = 1.0
    except (TypeError, ValueError):
        logger.warning(f"scipy.beta.ppf returned a non-scalar or non-numeric value for upper bound: {upper}")
        upper_f = 1.0

    # Final clamping and safety check
    lower_f = max(0.0, min(1.0, lower_f))
    upper_f = max(0.0, min(1.0, upper_f))

    if lower_f > upper_f:
        logger.warning(
            "Clopper-Pearson calculation resulted in lower > upper "
            f"(k={k}, n={n}, alpha={alpha}, scipy_version={_SCIPY_VERSION}). "
            "This indicates a numerical precision issue. Falling back to the trivial [0, 1] interval."
        )
        lower_f, upper_f = 0.0, 1.0

    return ConfidenceInterval(lower=lower_f, upper=upper_f, alpha=alpha, method=IntervalMethod.CLOPPER_PEARSON)

