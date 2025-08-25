# -*- coding: utf-8 -*-
"""
Abstract base class for finite-key security proofs in quantum key distribution.

This module defines the core interface for all security proof implementations,
ensuring a standardized structure for epsilon allocation, parameter estimation,
and secure key length calculation. It is designed with scientific rigor,
numerical stability, and extensibility in mind, reflecting best practices
from academic literature and software engineering.

This file is a self-contained version that includes necessary data structures
and utilities.

Formal Assumptions & Usage
--------------------------
- **Assumptions**: Proofs typically assume Poissonian sources, memoryless channels,
  independent pulses, and trusted local devices. Subclasses must document any
  deviations.
- **Numeric Precision**: Calculations use float64. For extreme precision needs
  (e.g., eps < 1e-50), an optional high-precision mode can be enabled via
  parameters (`use_mpmath=True`).
- **Clamping**: Clamping small negative numbers from solvers is a conservative
  measure to handle numerical noise. This may result in a key length of zero.
- **Concurrency**: This class is designed to be thread-safe. For multiprocessing,
  it is recommended to instantiate a new proof object in each worker process and
  use `multiprocessing.set_start_method('spawn')` to avoid fork-safety issues.

Recommended Settings
--------------------
- **Production Runs**: `mode=ProofMode.PRODUCTION` for performance.
- **Formal Audits**: `mode=ProofMode.AUDIT` for maximum strictness and diagnostics.
- **Debugging**: `mode=ProofMode.DEBUG` for verbose logging.

How to Read Diagnostics
-----------------------
- **SolverDiagnostics**: Contains details from numerical solvers (e.g., LP),
  including success status, timings, and residuals.
- **ErrorCode**: Provides machine-readable reasons for outcomes, such as
  `INSUFFICIENT_STATISTICS` when yields are too low.

Example `audit_dump` Schema
---------------------------
```json
{
    "params": {"run_id": "abc", ...},
    "mode": "AUDIT",
    "proof_implementation": "Lim2014Proof v1.0.0",
    "decoy_estimates": {
        "yield_1_lower_bound": 0.05, ...,
        "metadata": {"schema_version": "1.2", ...}
    },
    ...
}
```
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (Any, Callable, Dict, Generator, Iterable, Literal,
                    Protocol, Tuple, Type, TypeVar)

import numpy as np

# --- Safe SciPy Import with Version Check ---
try:
    import scipy
    from scipy import special
    from scipy.stats import beta
    if tuple(map(int, scipy.__version__.split('.')[:2])) < (1, 8):
        raise RuntimeError("SciPy >= 1.8 is required for reliable beta.ppf behavior.")
except ImportError as exc:
    raise RuntimeError(
        "SciPy is required for Clopper-Pearson confidence intervals "
        "(run `pip install 'scipy>=1.8'`)."
    ) from exc

# --- High-Precision Math (Optional) ---
try:
    import mpmath  # type: ignore
except ImportError:
    mpmath = None

# --- Constants and Type Variables ---
__version__ = "4.0.0"
__all__ = [
    'FiniteKeyProof', 'DecoyEstimates', 'KeyCalculationResult', 'TallyCounts',
    'EpsilonAllocation', 'SolverDiagnostics', 'ProofMode', 'ErrorCode',
    'ConfidenceBoundMethod', 'QKDParamsProto', 'ProofException',
    'ParameterValidationError', 'UnsupportedCIError', 'CIComputationError',
    'TALLY_KEY_Z_SIGNAL', 'TALLY_KEY_X_SIGNAL', 'derive_child_seed'
]
ENTROPY_PROB_CLAMP = 1e-15
NUMERIC_TOL = 1e-12
TINY_THRESHOLD = 1e-12
T = TypeVar("T")

# --- Stable Symbol Names for Tally Keys ---
TALLY_KEY_Z_SIGNAL = "Z_signal"
TALLY_KEY_X_SIGNAL = "X_signal"


# --- Type Protocol for QKDParams ---
class QKDParamsProto(Protocol):
    """
    A protocol for QKDParams to avoid circular imports while retaining type hints.
    The `sanitized_params` method must redact any secrets like master seeds.
    """
    ci_method: ConfidenceBoundMethod
    entropy_clamp: float
    run_id: str
    security_model: str
    solver_tol: float | None
    use_mpmath: bool | None
    export_seeds: bool | None
    hmac_key: bytes | None
    num_pulses: int | None

    def sanitized_params(self) -> Dict[str, Any]: ...


# --- Structured Exception Hierarchy ---
class ProofException(Exception): ...
class ParameterValidationError(ProofException, ValueError): ...
class UnsupportedCIError(ProofException, NotImplementedError): ...
class CIComputationError(ProofException, RuntimeError): ...


# --- Enums for Configuration ---
class ConfidenceBoundMethod(Enum):
    CLOPPER_PEARSON = auto()
    HOEFFDING = auto()

class ProofMode(Enum):
    PRODUCTION = auto()
    DEBUG = auto()
    AUDIT = auto()

class ErrorCode(Enum):
    INSUFFICIENT_STATISTICS = auto()
    LP_INFEASIBLE = auto()
    NUMERICAL_CLAMP_APPLIED = auto()
    LEAKAGE_EXCEEDS_ENTROPY = auto()
    HIGH_SOLVER_RESIDUAL = auto()
    SAFE_DIVIDE_FALLBACK = auto()


# --- Data Structures for API Contracts ---
def _json_safe_dict(data: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Recursively converts numpy/enum types to native Python types for JSON."""
    d: Dict[str, Any] = {}
    for key, value in data:
        if isinstance(value, (np.integer, np.int64)):
            d[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            d[key] = float(value)
        elif isinstance(value, np.ndarray):
            if value.size > 1000:
                d[key] = {'data': value[:1000].tolist(), 'truncated': True}
            else:
                d[key] = value.tolist()
        elif isinstance(value, np.bool_):
            d[key] = bool(value)
        elif isinstance(value, Enum):
            d[key] = value.name
        elif isinstance(value, dict):
            d[key] = _json_safe_dict(value.items())
        else:
            d[key] = value
    return d

@dataclass(frozen=True)
class TallyCounts:
    detections: int = 0
    errors: int = 0

    def __post_init__(self):
        if self.detections < 0 or self.errors < 0:
            raise ParameterValidationError("Tally counts cannot be negative.")
        if self.errors > self.detections:
            raise ParameterValidationError(f"Error count ({self.errors}) cannot exceed detection count ({self.detections}).")

@dataclass
class EpsilonAllocation:
    eps_sec: float
    eps_pe: float
    eps_smooth: float
    eps_pa: float

    def validate(self, policy: Callable[["EpsilonAllocation"], float]) -> None:
        consumed = policy(self)
        if consumed > self.eps_sec:
            raise ValueError(f"Epsilon allocation exceeds budget: {consumed:.2e} > {self.eps_sec:.2e}")

@dataclass(frozen=True)
class SolverDiagnostics:
    solver_name: str
    is_success: bool
    status_message: str # e.g., SUCCESS, TIMEOUT, NUMERIC_FAILURE
    residual_norm: float | None = None
    timings: Dict[str, float] = field(default_factory=dict)
    numeric_diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DecoyEstimates:
    yield_1_lower_bound: float
    error_rate_1_upper_bound: float
    is_feasible: bool
    failure_prob_used: float
    diagnostics: SolverDiagnostics

    def __repr__(self) -> str:
        return (f"DecoyEstimates(Y1_L={self.yield_1_lower_bound:.2e}, "
                f"e1_U={self.error_rate_1_upper_bound:.3f}, feasible={self.is_feasible})")

    def as_serializable(self) -> Dict[str, Any]:
        data = asdict(self, dict_factory=_json_safe_dict)
        data['metadata'] = {
            'schema_version': '1.2', 'lib_version': __version__,
            'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        }
        return data

@dataclass(frozen=True)
class KeyCalculationResult:
    secure_key_length: int
    privacy_amplification_term: float
    error_correction_leakage: float
    phase_error_rate_upper_bound: float
    diagnostics: list[str] = field(default_factory=list, repr=False)
    error_codes: list[ErrorCode] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"KeyCalculationResult(len={self.secure_key_length}, e_ph={self.phase_error_rate_upper_bound:.3f})"

    def as_serializable(self) -> Dict[str, Any]:
        data = asdict(self, dict_factory=_json_safe_dict)
        data['error_codes'] = [e.name for e in self.error_codes]
        data['metadata'] = {'schema_version': '1.2', 'lib_version': __version__}
        return data


# --- Abstract Base Class for Security Proofs ---
class FiniteKeyProof(ABC):
    """
    An abstract base class for implementing finite-key QKD security proofs.

    Subclasses **must** override this docstring to provide:
    1.  **DOI/arXiv**: A link to the primary scientific paper.
    2.  **Equation Mapping**: A list mapping key formulas to their implementation.
    3.  **Assumptions**: Any deviations from the module-level assumptions.
    """
    __implementation_version__: str = "0.0.0"
    allowed_ci_methods: frozenset[ConfidenceBoundMethod] = frozenset(ConfidenceBoundMethod)
    min_y1_threshold: float = 1e-9

    def __init__(self, params: QKDParamsProto, mode: ProofMode = ProofMode.PRODUCTION):
        self.p = params
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_id = getattr(params, 'run_id', 'no-run-id')
        self.min_prob_clamp = getattr(params, 'entropy_clamp', ENTROPY_PROB_CLAMP)
        self.use_mpmath = getattr(params, 'use_mpmath', False) and mpmath is not None

        self.logger.info("event=init run_id=%s proof=%s version=%s numpy=%s scipy=%s",
                         self.run_id, self.__class__.__name__, self.__implementation_version__,
                         np.__version__, scipy.__version__)

        self.validate_physical_params()
        self._check_notation_map()

        if self.p.ci_method not in self.allowed_ci_methods:
            msg = f"CI method '{self.p.ci_method.name}' is not recommended for this proof."
            if self.is_audit_mode(): raise ParameterValidationError(msg)
            self.logger.warning("[%s] %s", self.run_id, msg)

        self.eps_alloc = self.allocate_epsilons()
        self.eps_alloc.validate(policy=self.get_epsilon_policy())

    def is_audit_mode(self) -> bool: return self.mode == ProofMode.AUDIT
    def is_debug_mode(self) -> bool: return self.mode in (ProofMode.DEBUG, ProofMode.AUDIT)

    @abstractmethod
    def notation_map(self) -> Dict[str, str]: ...
    @abstractmethod
    def allocate_epsilons(self) -> EpsilonAllocation: ...
    @abstractmethod
    def get_epsilon_policy(self) -> Callable[[EpsilonAllocation], float]: ...
    @abstractmethod
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> DecoyEstimates: ...
    @abstractmethod
    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, TallyCounts]) -> KeyCalculationResult: ...

    def validate_physical_params(self) -> None:
        """Base-level sanity checks. Subclasses should call `super().`"""
        num_pulses = getattr(self.p, 'num_pulses', None)
        if num_pulses is not None and num_pulses < 100:
            msg = f"Number of pulses ({num_pulses}) is very low for a meaningful run."
            if self.is_audit_mode(): raise ParameterValidationError(msg)
            self.logger.warning(msg)

    def validate_stats_map(self, stats_map: Dict[str, TallyCounts], required_keys: list[str]):
        for key in required_keys:
            if key not in stats_map: raise ParameterValidationError(f"Missing required key in stats_map: '{key}'")
            if not isinstance(stats_map[key], TallyCounts): raise ParameterValidationError(f"Value for key '{key}' must be a TallyCounts instance.")

    def get_bounds(self, k: int, n: int, failure_prob: float, sided: Literal["two", "one_upper", "one_lower"] = "two", diagnostics: Dict | None = None) -> Tuple[float, float]:
        k, n = int(k), int(n)
        if not (0 <= k <= n): raise ParameterValidationError(f"k must be in [0, n], got k={k}, n={n}.")
        if not (1e-300 < failure_prob < 1): raise ParameterValidationError(f"Failure probability must be in (1e-300, 1), got {failure_prob}.")
        if n == 0: return (0.0, 1.0)
        alpha = float(failure_prob if sided != "two" else failure_prob / 2.0)

        p_obs = k / n
        if diagnostics is not None:
            diagnostics['alpha_used'] = alpha
            if p_obs == 0.0 or p_obs == 1.0: diagnostics['boundary_p_obs'] = True

        if self.p.ci_method == ConfidenceBoundMethod.CLOPPER_PEARSON:
            lower = self._safe_beta_ppf(alpha, k, n - k + 1, diagnostics) if k > 0 else 0.0
            upper = self._safe_beta_ppf(1 - alpha, k + 1, n - k, diagnostics) if k < n else 1.0
        elif self.p.ci_method == ConfidenceBoundMethod.HOEFFDING:
            if self.use_mpmath:
                with mpmath.workdps(100): delta = mpmath.sqrt(mpmath.log(1 / mpmath.mpf(alpha)) / (2 * mpmath.mpf(n)))
            else:
                log_term = self._safe_log(1.0 / alpha); delta = math.sqrt(log_term / (2.0 * n))
            self._assert_finite("Hoeffding delta", delta)
            lower, upper = max(0.0, p_obs - float(delta)), min(1.0, p_obs + float(delta))
        else:
            raise UnsupportedCIError(f"CI method '{self.p.ci_method.name}' not implemented.")

        self._assert_finite("lower bound", lower); self._assert_finite("upper bound", upper)

        if sided == "one_lower": return (lower, 1.0)
        if sided == "one_upper": return (0.0, upper)
        return (lower, upper)

    def binary_entropy(self, p_err: float | np.ndarray | list) -> float | np.ndarray:
        p = np.asarray(p_err, dtype=np.float64)
        p = np.clip(p, self.min_prob_clamp, 1.0 - self.min_prob_clamp)
        with np.errstate(divide='ignore', invalid='ignore', over='raise' if self.is_audit_mode() else 'warn'):
            h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        h = np.nan_to_num(h, nan=0.0)
        return h.item() if isinstance(p_err, (float, int)) else h

    def _safe_beta_ppf(self, q: float, a: float, b: float, diagnostics: Dict | None = None) -> float:
        try:
            val = beta.ppf(q, a, b)
            if not np.isfinite(val): raise ValueError("Non-finite result")
            return float(val)
        except (ValueError, RuntimeError) as e:
            msg = f"beta.ppf failed (scipy={scipy.__version__}), trying fallback. Error: {e}"
            if self.is_audit_mode(): raise CIComputationError(msg) from e
            self.logger.warning(msg)
            try:
                fallback_val = float(special.betaincinv(a, b, q))
                if diagnostics is not None:
                    diagnostics['beta_ppf_fallback_used'] = True
                    diagnostics['beta_ppf_original_error'] = str(e)
                    diagnostics['last_exception'] = traceback.format_exc()
                return fallback_val
            except Exception as fallback_exc:
                raise CIComputationError("beta.ppf fallback betaincinv also failed.") from fallback_exc

    def _check_notation_map(self):
        required = {"Y_1^L", "e_ph", "s_z_1^L"}
        if not required.issubset(self.notation_map().keys()):
            raise NotImplementedError(f"Subclass must implement notation_map with keys: {required}")

    def _assert_finite(self, name: str, value: Any):
        if not math.isfinite(value):
            msg = f"Computation of '{name}' resulted in non-finite value: {value}"
            if self.is_audit_mode(): raise CIComputationError(msg)
            self.logger.error(msg)

    def _safe_log(self, x: float) -> float:
        if x < 1e-300:
            if self.is_audit_mode(): raise CIComputationError("Log argument too small in AUDIT mode.")
            self.logger.warning("Log argument %e is smaller than safe limit, clamping.", x)
            x = 1e-300
        return math.log(x)

    def _safe_divide(self, num: float, den: float, default: float = 0.0, name: str = '', error_codes: list | None = None) -> float:
        if abs(den) < TINY_THRESHOLD:
            if error_codes is not None: error_codes.append(ErrorCode.SAFE_DIVIDE_FALLBACK)
            self.logger.warning("Denominator for '%s' is near zero (%e). Returning default value %f.", name, den, default)
            return default
        return num / den

    def _clamp_nonneg(self, x: float, name: str, error_codes: list | None = None) -> float:
        if x < 0:
            if error_codes is not None: error_codes.append(ErrorCode.NUMERICAL_CLAMP_APPLIED)
            if x < -NUMERIC_TOL and self.is_audit_mode():
                raise CIComputationError(f"Negative value for {name} ({x}) exceeded tolerance in AUDIT mode.")
            self.logger.debug("Clamped negative value for %s: %e -> 0.0", name, x)
            return 0.0
        return x

    def _clamp_key_length(self, l_float: float) -> int:
        return int(max(0.0, math.floor(l_float)))

    @contextmanager
    def _timed(self, timings: Dict, key: str) -> Generator:
        t0 = time.perf_counter(); yield; timings[key] = time.perf_counter() - t0

    def _conservative_decoy_estimate(self, failure_prob: float, diagnostics: SolverDiagnostics) -> DecoyEstimates:
        return DecoyEstimates(0.0, 0.5, False, failure_prob, diagnostics)

    def audit_dump(self, result: KeyCalculationResult, decoy: DecoyEstimates, stats: Dict, start_time: float) -> Dict:
        dump = {
            "params": self.p.sanitized_params(), "mode": self.mode.name,
            "proof_implementation": f"{self.__class__.__name__} v{self.__implementation_version__}",
            "stats_map": {k: asdict(v) for k, v in stats.items()},
            "decoy_estimates": decoy.as_serializable(),
            "key_calculation_result": result.as_serializable(),
            "timestamps": {'start': start_time, 'end': time.time()},
        }
        if self.is_audit_mode():
             # In audit mode, add more detailed diagnostics like sensitivity
             y1_margin = decoy.yield_1_lower_bound - self.min_y1_threshold
             e1_margin = 0.5 - decoy.error_rate_1_upper_bound # Assuming max error is 0.5
             dump['sensitivity_summary'] = {'y1_margin': y1_margin, 'e1_margin': e1_margin}
        return dump

    def explain_key_decision(self, result: KeyCalculationResult, decoy: DecoyEstimates) -> str:
        if result.secure_key_length > 0:
            return f"Secure key generated. Final length: {result.secure_key_length} bits."
        reasons = []
        if ErrorCode.INSUFFICIENT_STATISTICS in result.error_codes:
            reasons.append(f"insufficient single-photon yield (Y1_L={decoy.yield_1_lower_bound:.2e} < threshold={self.min_y1_threshold:.2e})")
        if ErrorCode.LP_INFEASIBLE in result.error_codes:
            reasons.append("decoy-state analysis was infeasible")
        if ErrorCode.LEAKAGE_EXCEEDS_ENTROPY in result.error_codes:
            reasons.append(f"information leakage ({result.error_correction_leakage + result.privacy_amplification_term:.2f}) exceeded available entropy")
        if not reasons: reasons.append("an unspecified condition led to a non-positive key length")
        return f"No secure key generated because {', and '.join(reasons)}."


# --- Utility Functions ---
@lru_cache(maxsize=128)
def derive_child_seed(master_seed: int, index: int) -> int:
    """Deterministically derives a child seed. Not for cryptographic use."""
    key = str(master_seed).encode()
    msg = str(index).encode()
    digest = hmac.new(key, msg, digestmod=hashlib.sha256).digest()
    return int.from_bytes(digest[:8], 'big') % (2**63 - 1) or 1

