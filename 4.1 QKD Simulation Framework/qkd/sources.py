# -*- coding: utf-8 -*-
"""
Models for quantum light sources, incorporating comprehensive validation,
numerical stability enhancements, and API improvements based on expert review.

This module provides a high-performance, validated `PoissonSource` class and a
numerically stable `p_n_mu_vector` function for QKD simulations.

NOTE:
- This module has a hard dependency on NumPy (>=1.20) and SciPy (>=1.6).
- It is intended for use on 64-bit platforms for full integer range support.
- Importing this module may raise ImportError if dependencies are not met.
- For reproducible results, the caller must seed the random number generator (RNG).
- Fallback types are exported in __all__ only when the canonical `qkd` package is not found.
"""
# --- Module Metadata ---
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("qkd_simulation_package")  # Replace with your actual package name
except (ImportError, PackageNotFoundError):
    __version__ = "7.0.0"  # Fallback for standalone use

__author__ = "Quantum Optics Simulation Group"


# --- Imports ---
import logging
import math
import numbers
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, ClassVar, Dict, List, Optional, Set, Union

import numpy as np

# --- Logger Definition ---
logger = logging.getLogger(__name__)


# --- Dependency Version Check and Imports ---
try:
    from packaging.version import Version  # type: ignore
    _PACKAGING_AVAILABLE = True
except ImportError:
    _PACKAGING_AVAILABLE = False
    logger.warning("`packaging` library not found. Cannot perform version checks for NumPy and SciPy.")

_NUMPY_MIN_VERSION = "1.20.0"
_SCIPY_MIN_VERSION = "1.6.0"

if _PACKAGING_AVAILABLE:
    try:
        if Version(np.__version__) < Version(_NUMPY_MIN_VERSION):
            raise ImportError(f"NumPy version >= {_NUMPY_MIN_VERSION} required, but found {np.__version__}.")
    except Exception as e:
        logger.warning(f"Could not parse NumPy version string '{np.__version__}': {e}")
    try:
        import scipy
        try:
            if Version(scipy.__version__) < Version(_SCIPY_MIN_VERSION):
                raise ImportError(f"SciPy version >= {_SCIPY_MIN_VERSION} required, but found {scipy.__version__}.")
        except Exception as e:
            logger.warning(f"Could not parse SciPy version string '{scipy.__version__}': {e}")
    except ImportError:
        raise ImportError("This module requires SciPy. Please install it with 'pip install scipy'.")
else:
    try:
        import scipy
    except ImportError:
        raise ImportError("This module requires SciPy. Please install it with 'pip install scipy'.")

from scipy.special import gammaincc, gammaln, logsumexp


# --- Canonical Type Imports with Fallback ---
_STANDALONE_MODE = False
try:
    from qkd.exceptions import ParameterValidationError, QKDSimulationError
    from qkd.datatypes import PulseTypeConfig
except (ImportError, ModuleNotFoundError) as e:
    _STANDALONE_MODE = True
    logger.debug(f"qkd package not found (reason: {e}); using local fallback types for standalone use.")
    class ParameterValidationError(ValueError): # type: ignore [no-redef]
        """(Standalone) Custom exception for invalid simulation parameters."""
        pass
    class QKDSimulationError(RuntimeError): # type: ignore [no-redef]
        """(Standalone) Custom exception for errors during simulation runtime."""
        pass
    @dataclass(frozen=True, slots=True)
    class PulseTypeConfig: # type: ignore [no-redef]
        """(Standalone) Configuration for a single pulse type."""
        name: str
        mean_photon_number: float
        probability: float
        def __post_init__(self) -> None:
            if not isinstance(self.name, str) or not self.name: raise ParameterValidationError("Pulse name must be a non-empty string.")
            mpn = float(self.mean_photon_number)
            if not math.isfinite(mpn) or mpn < 0: raise ParameterValidationError("Pulse mean_photon_number must be a non-negative real number.")
            prob = float(self.probability)
            if not (0.0 <= prob <= 1.0): raise ParameterValidationError("Pulse probability must be a real number in [0, 1].")
except Exception:
    logger.exception("Unexpected error while importing from the `qkd` package.")
    raise


# --- Module Constants ---
PROB_SUM_ABS_TOL: float = 1e-9
PROB_SUM_REL_TOL: float = 1e-9
PMF_SUM_TOL: float = 1e-12
MAX_N_CAP_DEFAULT: int = 100_000

__all__ = ["PoissonSource", "p_n_mu_vector"]
if _STANDALONE_MODE:
    __all__.extend(["PulseTypeConfig", "ParameterValidationError", "QKDSimulationError"])


@dataclass(frozen=True, slots=True)
class PoissonSource:
    """Models a light source with Poissonian photon number statistics."""
    pulse_configs: List[PulseTypeConfig]
    _mus_cache: np.ndarray = field(init=False, repr=False, compare=False)
    _PROB_SUM_TOL_INFO: ClassVar[str] = (f"(abs_tol={PROB_SUM_ABS_TOL}, rel_tol={PROB_SUM_REL_TOL})")

    def __post_init__(self) -> None:
        if not self.pulse_configs: raise ParameterValidationError("pulse_configs must be a non-empty list.")
        names_in_order = [pc.name for pc in self.pulse_configs]
        seen_names: Set[str] = set()
        for i, pc in enumerate(self.pulse_configs):
            if not all(hasattr(pc, attr) for attr in ['name', 'probability', 'mean_photon_number']): raise ParameterValidationError(f"Item at index {i} is not a valid PulseTypeConfig.")
            if pc.name in seen_names: raise ParameterValidationError(f"Duplicate pulse name found: '{pc.name}'.")
            seen_names.add(pc.name)
            if not (0.0 <= pc.probability <= 1.0): raise ParameterValidationError(f"Pulse '{pc.name}' (index {i}) has invalid probability {pc.probability:.6g}.")
            if not (pc.mean_photon_number >= 0 and math.isfinite(pc.mean_photon_number)): raise ParameterValidationError(f"Pulse '{pc.name}' (index {i}) has non-finite mean_photon_number: {pc.mean_photon_number:.6g}.")

        prob_sum: float = math.fsum(float(pc.probability) for pc in self.pulse_configs)
        if not math.isclose(prob_sum, 1.0, rel_tol=PROB_SUM_REL_TOL, abs_tol=PROB_SUM_ABS_TOL): raise ParameterValidationError(f"Sum of probabilities must be ~1.0, but got {prob_sum:.6g}. Tolerance: {self._PROB_SUM_TOL_INFO}.")

        mus = np.array([pc.mean_photon_number for pc in self.pulse_configs], dtype=np.float64, copy=True)
        if not np.all(np.isfinite(mus)):
            bad_indices = np.where(~np.isfinite(mus))[0]
            raise ParameterValidationError(f"Found non-finite values in mean_photon_numbers at indices: {bad_indices[:5]}.")
        mus.setflags(write=False)
        object.__setattr__(self, "_mus_cache", mus)
        names_str = ", ".join(names_in_order[:5]) + ("..." if len(names_in_order) > 5 else "")
        logger.debug("PoissonSource initialized for %d pulses: %s", len(names_in_order), names_str)

    def to_config_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dictionary of the source configuration."""
        return {
            "pulse_configs": [asdict(pc) for pc in self.pulse_configs]
        }

    @property
    def mus(self) -> np.ndarray:
        """Returns a copy of the mean photon number array."""
        return self._mus_cache.copy()

    def generate_photons(self, alice_pulse_indices: Any, rng: Any) -> Union[np.ndarray, np.int64]:
        """Generates photon numbers for a sequence of pulses."""
        if not hasattr(rng, "poisson") or not callable(getattr(rng, "poisson")): raise TypeError(f"rng of type {type(rng).__name__} must have a callable 'poisson' method.")
        
        is_scalar = np.isscalar(alice_pulse_indices)
        indices = np.atleast_1d(alice_pulse_indices)
        if not np.issubdtype(indices.dtype, np.integer):
            if np.issubdtype(indices.dtype, np.floating) and np.all(np.isclose(indices, np.rint(indices))):
                if np.any((indices < np.iinfo(np.intp).min) | (indices > np.iinfo(np.intp).max)): raise ParameterValidationError("Float indices are too large to safely cast to platform integer.")
                indices = indices.astype(np.intp)
            else:
                raise ParameterValidationError("alice_pulse_indices must contain integer or integer-valued float values.")
        if indices.ndim != 1: raise ParameterValidationError("alice_pulse_indices must be a 1D array or scalar.")

        if indices.size > 0:
            min_idx, max_idx = indices.min(), indices.max()
            if min_idx < 0 or max_idx >= len(self.pulse_configs): raise ParameterValidationError(f"Out-of-range indices. Valid: [0, {len(self.pulse_configs) - 1}], got min={min_idx}, max={max_idx}.")

        photons = rng.poisson(self._mus_cache[indices])
        result = photons.astype(np.int64, copy=False)
        return result.item() if is_scalar else result

    def get_pulse_config_by_name(self, name: str) -> PulseTypeConfig:
        """Retrieves a pulse configuration by its unique name."""
        try:
            return next(c for c in self.pulse_configs if c.name == name)
        except StopIteration:
            raise ParameterValidationError(f"No pulse configuration found with name: '{name}'.")

    def __repr__(self) -> str:
        num_pulses = len(self.pulse_configs)
        names = [pc.name for pc in self.pulse_configs]
        names_str = ", ".join(f"'{n}'" for n in names[:3]) + ("..." if num_pulses > 3 else "")
        return f"{self.__class__.__name__}(num_pulses={num_pulses}, names=[{names_str}])"


@lru_cache(maxsize=128)
def p_n_mu_vector(
    mu: float, n_cap: int, *,
    tail_threshold: Optional[float] = None, return_log: bool = False,
    hard_cap_limit: bool = False, max_n_cap: int = MAX_N_CAP_DEFAULT,
    min_mu_threshold: float = np.finfo(float).tiny
) -> np.ndarray:
    """Calculates the Poisson PMF vector robustly using log-space computation."""
    try: mu = float(mu)
    except (TypeError, ValueError): raise ParameterValidationError(f"Mean `mu` must be a float, got type {type(mu).__name__}.")
    if not math.isfinite(mu) or mu < 0: raise ParameterValidationError(f"Mean `mu` must be a non-negative finite number, got {mu:.6g}.")
    try: n_cap = int(n_cap)
    except (TypeError, ValueError): raise ParameterValidationError(f"n_cap must be an integer, got type {type(n_cap).__name__}.")
    if n_cap < 1: raise ParameterValidationError(f"n_cap must be a positive integer, got {n_cap}.")

    if n_cap > max_n_cap:
        mem_mib = (n_cap + 1) * 8 / (1024**2)
        msg = f"n_cap={n_cap} exceeds max_n_cap={max_n_cap}. This will allocate ~{mem_mib:.2f} MiB (approx. data buffer size)."
        if hard_cap_limit: raise ParameterValidationError(msg)
        logger.warning(msg)

    if mu < min_mu_threshold:
        if return_log:
            vec = np.full(n_cap + 1, -np.inf, dtype=np.float64); vec[0] = 0.0
        else:
            vec = np.zeros(n_cap + 1, dtype=np.float64); vec[0] = 1.0
        return vec

    log_vec = np.empty(n_cap + 1, dtype=np.float64)
    ns = np.arange(n_cap, dtype=np.int64)
    with np.errstate(all='ignore'):
        log_pmf_body = ns * np.log(mu) - mu - gammaln(ns + 1.0)
    if not np.all(np.isfinite(log_pmf_body)):
        bad_indices = np.where(~np.isfinite(log_pmf_body))[0]
        raise QKDSimulationError(f"Numeric overflow in PMF calc for mu={mu:.6g}, n_cap={n_cap} at indices {bad_indices[:5]}. Consider increasing n_cap.")
    log_vec[:n_cap] = log_pmf_body

    with np.errstate(divide='ignore', invalid='ignore'):
        tail_val = gammaincc(float(n_cap), mu)
    if np.isnan(tail_val): raise QKDSimulationError(f"Tail calculation resulted in NaN for mu={mu:.6g}, n_cap={n_cap}.")
    if tail_val <= 0.0: logger.debug("Tail probability underflowed to %.2e for mu=%.2e, n_cap=%d", tail_val, mu, n_cap)
    log_tail = -np.inf if tail_val <= 0.0 else math.log(tail_val)
    log_vec[-1] = log_tail

    if np.all(np.isneginf(log_vec)): raise QKDSimulationError(f"All probabilities underflowed to zero for mu={mu:.6g}, n_cap={n_cap}. Try increasing n_cap.")

    log_sum = logsumexp(log_vec)
    if not np.isfinite(log_sum): raise QKDSimulationError(f"Log-sum is not finite during normalization for mu={mu:.6g}, n_cap={n_cap}.")

    log_vec_normalized = log_vec - log_sum
    if return_log: return log_vec_normalized

    vec = np.exp(log_vec_normalized)
    vec = np.clip(vec, 0.0, None)
    tail_prob = vec[-1]

    if tail_threshold is not None and tail_prob > tail_threshold: raise ParameterValidationError(f"Tail P(n>={n_cap}) = {tail_prob:.3e} for mu={mu:.3f} exceeds threshold of {tail_threshold:.3e}. Consider increasing n_cap.")

    s = np.sum(vec, dtype=np.float64)
    if not math.isclose(s, 1.0, rel_tol=PMF_SUM_TOL, abs_tol=PMF_SUM_TOL): raise QKDSimulationError(f"Final PMF sum is not 1.0 (sum={s:.18f}).")

    return vec

