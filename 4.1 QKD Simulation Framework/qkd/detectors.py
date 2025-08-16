# -*- coding: utf-8 -*-
"""
Models for a two-detector quantum receiver system.

This module provides a `ThresholdDetector` class that simulates the behavior of
a pair of single-photon detectors. It is designed for high-fidelity numerical
simulations in quantum communication, emphasizing correctness, numerical stability,
security, and a robust, well-documented API.

The model incorporates critical physical effects:
- Per-detector efficiency and channel transmittance.
- Dark counts (uncorrelated between detectors).
- Optical misalignment (configurable to affect signal-only or all events).
- Intrinsic QBER (detection result flips).
- Detector dead time.
- Time-correlated, stateful after-pulsing.
- Configurable policies for handling double-click events.

Key Features:
- **Immutable Configuration**: The detector's physical parameters are frozen on
  instantiation to prevent accidental modification.
- **Secure, Versioned State**: The mutable state is managed separately and can be
  safely serialized to/from a versioned dictionary, avoiding the security risks
  of pickle. JSON serialization handles non-standard values like infinity safely.
- **Dual Simulation Paths**: A highly optimized vectorized path is used when
  stateful effects are disabled, while a performant sequential path with a
  circular buffer correctly models time-dependent behavior.
- **Memory Efficiency**: Supports chunked processing for simulating very large
  datasets without exhausting memory.
- **Reproducibility**: The API supports reproducible simulations through controlled
  RNG usage and an option to provide pre-generated random numbers. A helper
  function is provided to generate these random sets.

Dependencies:
- Python 3.10+
- NumPy 1.22+

Example:
    >>> import logging
    >>> import numpy as np
    >>> from numpy.random import default_rng
    >>>
    >>> logging.basicConfig(level=logging.INFO)
    >>>
    >>> detector = ThresholdDetector(
    ...     det_eff_d0=0.85, det_eff_d1=0.75, dark_rate=1e-7,
    ...     qber_intrinsic=0.001, misalignment=0.005,
    ...     double_click_policy=DoubleClickPolicy.TOSS,
    ...     afterpulse_prob=0.01, afterpulse_memory=10, dead_time_ns=50.0
    ... )
    >>>
    >>> rng = default_rng(seed=42)
    >>> num_pulses = 1000
    >>> photon_numbers = rng.poisson(0.5, size=num_pulses)
    >>> ideal_outcomes_d0 = rng.random(num_pulses) < 0.5
    >>>
    >>> results = detector.simulate_detection(
    ...     channel_transmittance=0.1, photon_numbers=photon_numbers, rng=rng,
    ...     ideal_outcomes_d0=ideal_outcomes_d0, pulse_period_ns=10.0,
    ...     return_diagnostics=True
    ... )
    >>>
    >>> print(f"Detector 0 clicks: {np.sum(results.click0)}")
    >>> print(f"Detector 1 clicks: {np.sum(results.click1)}")
    >>> print(f"Diagnostics: {results.diagnostics}")
"""
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# --- Fallback definitions for local imports ---
try:
    from .exceptions import ParameterValidationError
except ImportError:
    class ParameterValidationError(ValueError): pass

try:
    from .datatypes import DoubleClickPolicy
except ImportError:
    class DoubleClickPolicy(Enum):
        TOSS, RANDOM, D0_WINS, D1_WINS = 1, 2, 3, 4

# --- Structured return types ---
@dataclass(frozen=True)
class DetectionDiagnostics:
    """Diagnostic information from a detection simulation run (integer counts)."""
    double_clicks_resolved: int = 0
    resolved_to_d0: int = 0
    resolved_to_d1: int = 0
    tossed_events: int = 0
    misalignment_flips: int = 0
    qber_flips: int = 0
    dead_time_suppressions: int = 0

class DetectionResult(NamedTuple):
    """Structured output for a detection simulation."""
    click0: NDArray[np.bool_]
    click1: NDArray[np.bool_]
    diagnostics: Optional[DetectionDiagnostics] = None
    final_state: Optional[str] = None

@dataclass
class _DetectorState:
    """Internal mutable state for the detector."""
    ap_state_d0: Tuple[NDArray[np.bool_], int, int]
    ap_state_d1: Tuple[NDArray[np.bool_], int, int]
    time_since_last_click_ns_d0: Optional[float]
    time_since_last_click_ns_d1: Optional[float]

    def __repr__(self) -> str:
        return (f"_DetectorState(ap_sum0={self.ap_state_d0[1]}, ap_sum1={self.ap_state_d1[1]}, "
                f"time_d0={self.time_since_last_click_ns_d0}, time_d1={self.time_since_last_click_ns_d1})")

__all__ = [
    "ThresholdDetector", "DoubleClickPolicy", "DetectionResult",
    "DetectionDiagnostics", "ParameterValidationError", "generate_precomputed_randoms"
]

@dataclass(frozen=True, slots=True)
class ThresholdDetector:
    """
    Models a stateful two-detector system with an immutable configuration.

    This class is NOT thread-safe. For multi-threaded simulations, use one
    instance per thread, created via `clone()`.

    Attributes:
        det_eff_d0, det_eff_d1: Intrinsic efficiency for each detector [0,1].
        dark_rate: Probability of a dark count per detector per pulse [0,1).
        qber_intrinsic: Probability a signal-induced click is assigned to the wrong detector.
        misalignment: Probability an incoming photon is routed to the wrong detector.
        misalignment_affects_dark_counts: If True, misalignment can swap dark counts.
        double_click_policy: Policy for resolving double-click events.
        afterpulse_prob: Probability of a single preceding click causing an after-pulse.
        afterpulse_memory: Number of recent pulses to consider for after-pulsing.
        dead_time_ns: Duration in ns after a click during which a detector cannot fire.
        strict_mode: If True, raise errors on numerical anomalies (e.g., NaNs).
    """
    det_eff_d0: float
    det_eff_d1: float
    dark_rate: float
    qber_intrinsic: float
    misalignment: float
    misalignment_affects_dark_counts: bool = True
    double_click_policy: DoubleClickPolicy
    afterpulse_prob: float = 0.0
    afterpulse_memory: int = 0
    dead_time_ns: float = 0.0
    strict_mode: bool = False

    _internal_state: _DetectorState = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        self._validate_params()
        object.__setattr__(self, '_internal_state', self._create_initial_state())

    def __repr__(self) -> str:
        params = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in self.__dataclass_fields__)
        return f"{self.__class__.__name__}({params})"

    def _validate_params(self):
        """Complete validation of all configuration parameters."""
        params_to_check = {
            "det_eff_d0": (0.0, 1.0, True), "det_eff_d1": (0.0, 1.0, True),
            "dark_rate": (0.0, 1.0, False), "qber_intrinsic": (0.0, 1.0, False),
            "misalignment": (0.0, 1.0, False), "afterpulse_prob": (0.0, 1.0, True)
        }
        for name, (low, high, high_incl) in params_to_check.items():
            val = getattr(self, name)
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                raise ParameterValidationError(f"{name} must be a finite number, but got {val}.")
            if not (low <= val and (val <= high if high_incl else val < high)):
                interval = f"[{low}, {high}]" if high_incl else f"[{low}, {high})"
                raise ParameterValidationError(f"{name} must be in {interval}, but got {val}.")

        if not isinstance(self.double_click_policy, DoubleClickPolicy):
            raise ParameterValidationError(f"double_click_policy must be a DoubleClickPolicy enum.")
        if not (isinstance(self.afterpulse_memory, int) and 0 <= self.afterpulse_memory <= 100_000):
            raise ParameterValidationError(f"afterpulse_memory must be an integer in [0, 100000].")
        if not (isinstance(self.dead_time_ns, (int, float)) and math.isfinite(self.dead_time_ns) and self.dead_time_ns >= 0):
            raise ParameterValidationError("dead_time_ns must be a non-negative finite number.")
        if not isinstance(self.misalignment_affects_dark_counts, bool):
            raise ParameterValidationError("misalignment_affects_dark_counts must be a boolean.")
        if not isinstance(self.strict_mode, bool):
            raise ParameterValidationError("strict_mode must be a boolean.")

    def _create_initial_state(self) -> _DetectorState:
        if self.afterpulse_memory > 0:
            state_d0 = (np.zeros(self.afterpulse_memory, dtype=bool), 0, 0)
            state_d1 = (np.zeros(self.afterpulse_memory, dtype=bool), 0, 0)
        else:
            state_d0 = (np.array([], dtype=bool), 0, 0)
            state_d1 = (np.array([], dtype=bool), 0, 0)
        return _DetectorState(
            ap_state_d0=state_d0, ap_state_d1=state_d1,
            time_since_last_click_ns_d0=None, time_since_last_click_ns_d1=None
        )

    def reset_state(self):
        object.__setattr__(self, '_internal_state', self._create_initial_state())

    def get_state(self) -> str:
        state = self._internal_state
        payload = {
            'version': 3,
            'config_hash': self._config_hash(),
            'ap_buf0': state.ap_state_d0[0].tolist(), 'ap_sum0': state.ap_state_d0[1], 'ap_idx0': state.ap_state_d0[2],
            'ap_buf1': state.ap_state_d1[0].tolist(), 'ap_sum1': state.ap_state_d1[1], 'ap_idx1': state.ap_state_d1[2],
            'time_d0': state.time_since_last_click_ns_d0, 'time_d1': state.time_since_last_click_ns_d1,
        }
        return json.dumps(payload)

    def set_state(self, state_json: str):
        try:
            d = json.loads(state_json)
            if d.get('version') != 3: raise ValueError("State version is incompatible.")
            if d.get('config_hash') != self._config_hash():
                logger.warning("Loading state from a different detector configuration.")

            ap_buf0 = np.array(d.get('ap_buf0', []), dtype=bool)
            if len(ap_buf0) != self.afterpulse_memory: raise ValueError("State `afterpulse_memory` mismatch.")
            
            ap_sum0, ap_idx0 = int(d['ap_sum0']), int(d['ap_idx0'])
            if not (0 <= ap_sum0 <= self.afterpulse_memory and (self.afterpulse_memory == 0 or 0 <= ap_idx0 < self.afterpulse_memory)):
                raise ValueError("Invalid after-pulsing state values for detector 0.")

            new_state = _DetectorState(
                ap_state_d0=(ap_buf0, ap_sum0, ap_idx0),
                ap_state_d1=(np.array(d['ap_buf1'], dtype=bool), int(d['ap_sum1']), int(d['ap_idx1'])),
                time_since_last_click_ns_d0=d.get('time_d0'), time_since_last_click_ns_d1=d.get('time_d1'),
            )
            object.__setattr__(self, '_internal_state', new_state)
        except Exception as e:
            raise ValueError(f"Failed to load state from JSON: {e}") from e

    def clone(self) -> "ThresholdDetector":
        new_detector = replace(self)
        # The new object gets a fresh state from __post_init__, so we set from the old one
        new_detector.set_state(self.get_state())
        return new_detector

    def _config_hash(self) -> str:
        """Creates a hash of the immutable configuration for state validation."""
        config_str = (f"{self.det_eff_d0}{self.det_eff_d1}{self.dark_rate}"
                      f"{self.afterpulse_memory}{self.dead_time_ns}")
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def simulate_detection(
        self,
        channel_transmittance: float | NDArray[np.float64],
        photon_numbers: NDArray[np.int_],
        rng: Generator,
        ideal_outcomes_d0: NDArray[np.bool_],
        pulse_period_ns: float = 1.0,
        return_diagnostics: bool = False,
        precomputed_randoms: Optional[Dict[str, NDArray]] = None,
        chunk_size: Optional[int] = None,
        initial_state: Optional[str] = None
    ) -> DetectionResult:
        if initial_state:
            self.set_state(initial_state)

        photon_numbers = np.asarray(photon_numbers, dtype=np.int64)
        num_pulses = photon_numbers.shape[0]

        if chunk_size is None or chunk_size >= num_pulses:
            return self._simulate_single_batch(
                channel_transmittance, photon_numbers, rng, ideal_outcomes_d0,
                pulse_period_ns, return_diagnostics, precomputed_randoms
            )

        all_clicks0, all_clicks1 = [], []
        agg_diag = DetectionDiagnostics()

        for i in range(0, num_pulses, chunk_size):
            end = i + chunk_size
            chunk_trans = channel_transmittance if np.isscalar(channel_transmittance) else channel_transmittance[i:end]
            chunk_photons = photon_numbers[i:end]
            chunk_ideal = ideal_outcomes_d0[i:end]
            
            chunk_randoms = None
            if precomputed_randoms:
                chunk_randoms = {k: v[i:end] for k, v in precomputed_randoms.items()}

            result = self._simulate_single_batch(
                chunk_trans, chunk_photons, rng, chunk_ideal,
                pulse_period_ns, return_diagnostics, chunk_randoms
            )
            all_clicks0.append(result.click0)
            all_clicks1.append(result.click1)
            if return_diagnostics and result.diagnostics:
                agg_diag = DetectionDiagnostics(**{k: getattr(agg_diag, k) + getattr(result.diagnostics, k) for k in agg_diag._fields})

        return DetectionResult(
            click0=np.concatenate(all_clicks0),
            click1=np.concatenate(all_clicks1),
            diagnostics=agg_diag if return_diagnostics else None,
            final_state=self.get_state()
        )

    def _simulate_single_batch(
        self, channel_transmittance, photon_numbers, rng, ideal_outcomes_d0,
        pulse_period_ns, return_diagnostics, precomputed_randoms
    ) -> DetectionResult:
        use_sequential = self.afterpulse_memory > 0 or self.dead_time_ns > 0
        if not use_sequential:
            res = self._simulate_vectorized(
                channel_transmittance, photon_numbers, rng, ideal_outcomes_d0,
                return_diagnostics, precomputed_randoms
            )
        else:
            res = self._simulate_sequential(
                channel_transmittance, photon_numbers, rng, ideal_outcomes_d0,
                pulse_period_ns, return_diagnostics, precomputed_randoms
            )
        return res._replace(final_state=self.get_state())

    def _calculate_p_click_signal(
        self, channel_transmittance: NDArray[np.float64], photon_numbers: NDArray[np.int64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        eta_d0 = np.clip(np.asarray(channel_transmittance) * self.det_eff_d0, 0.0, 1.0)
        eta_d1 = np.clip(np.asarray(channel_transmittance) * self.det_eff_d1, 0.0, 1.0)
        p_click_d0 = self._compute_prob_from_eta(eta_d0, photon_numbers)
        p_click_d1 = self._compute_prob_from_eta(eta_d1, photon_numbers)
        return p_click_d0, p_click_d1

    def _compute_prob_from_eta(self, eta, photon_numbers):
        p_click = np.zeros_like(photon_numbers, dtype=np.float64)
        eta_is_one_mask = np.isclose(eta, 1.0)
        p_click[eta_is_one_mask & (photon_numbers > 0)] = 1.0
        valid_eta_mask = (eta > 0.0) & ~eta_is_one_mask
        if np.any(valid_eta_mask):
            n_sub = photon_numbers[valid_eta_mask]
            eta_sub = eta if np.isscalar(eta) else eta[valid_eta_mask]
            log1m_eta = np.log1p(-eta_sub)
            with np.errstate(over='ignore', invalid='ignore'):
                prob = -np.expm1(n_sub.astype(np.float64) * log1m_eta)
            p_click[valid_eta_mask] = prob
        if self.strict_mode and np.any(np.isnan(p_click)):
            raise ValueError("NaNs detected in signal click probability calculation.")
        np.nan_to_num(p_click, copy=False, nan=0.0)
        return np.clip(p_click, 0.0, 1.0)

    def _apply_errors_and_clicks(
        self, p_click_d0, p_click_d1, ideal_outcomes_d0, rng, precomputed_randoms
    ) -> Dict[str, Any]:
        num_pulses = len(p_click_d0)
        
        def get_rand(key, size):
            if precomputed_randoms and key in precomputed_randoms:
                return precomputed_randoms[key]
            return rng.random(size=size)

        misalignment_flips = get_rand("misalignment", num_pulses) < self.misalignment
        effective_ideal_d0 = np.where(misalignment_flips, ~ideal_outcomes_d0, ideal_outcomes_d0)

        click0_base_signal = get_rand("signal_d0", num_pulses) < p_click_d0
        click1_base_signal = get_rand("signal_d1", num_pulses) < p_click_d1
        
        click0_signal_path = click0_base_signal & effective_ideal_d0
        click1_signal_path = click1_base_signal & ~effective_ideal_d0

        qber_flips_potential = get_rand("qber", num_pulses) < self.qber_intrinsic
        qber_flips_on_d0 = qber_flips_potential & click0_signal_path
        qber_flips_on_d1 = qber_flips_potential & click1_signal_path

        click0_signal = (click0_signal_path ^ qber_flips_on_d0) | qber_flips_on_d1
        click1_signal = (click1_signal_path ^ qber_flips_on_d1) | qber_flips_on_d0

        click0_dark = get_rand("dark_d0", num_pulses) < self.dark_rate
        click1_dark = get_rand("dark_d1", num_pulses) < self.dark_rate
        
        if not self.misalignment_affects_dark_counts:
             swapped_dark_d0 = click0_dark & ~ideal_outcomes_d0 & misalignment_flips
             swapped_dark_d1 = click1_dark & ideal_outcomes_d0 & misalignment_flips
             click0_dark = (click0_dark ^ swapped_dark_d0) | swapped_dark_d1
             click1_dark = (click1_dark ^ swapped_dark_d1) | swapped_dark_d0

        return {
            "click0_signal": click0_signal, "click1_signal": click1_signal,
            "click0_dark": click0_dark, "click1_dark": click1_dark,
            "misalignment_flips": int(np.sum(misalignment_flips)),
            "qber_flips": int(np.sum(qber_flips_on_d0 | qber_flips_on_d1))
        }

    def _resolve_double_clicks(
        self, clicks_d0, clicks_d1, rng
    ) -> Tuple[NDArray[np.bool_], NDArray[np.bool_], DetectionDiagnostics]:
        double_click_mask = clicks_d0 & clicks_d1
        num_double_clicks = int(np.sum(double_click_mask))
        diag = DetectionDiagnostics(double_clicks_resolved=num_double_clicks)
        if num_double_clicks > 0:
            policy = self.double_click_policy
            if policy is DoubleClickPolicy.TOSS:
                clicks_d0[double_click_mask] = False
                clicks_d1[double_click_mask] = False
                diag = replace(diag, tossed_events=num_double_clicks)
            elif policy is DoubleClickPolicy.D0_WINS:
                clicks_d1[double_click_mask] = False
                diag = replace(diag, resolved_to_d0=num_double_clicks)
            elif policy is DoubleClickPolicy.D1_WINS:
                clicks_d0[double_click_mask] = False
                diag = replace(diag, resolved_to_d1=num_double_clicks)
            elif policy is DoubleClickPolicy.RANDOM:
                choices = rng.integers(0, 2, size=num_double_clicks, dtype=bool)
                clicks_d0[double_click_mask] = choices
                clicks_d1[double_click_mask] = ~choices
                diag = replace(diag, resolved_to_d0=int(np.sum(choices)),
                                     resolved_to_d1=int(np.sum(~choices)))
        return clicks_d0, clicks_d1, diag

    def _simulate_vectorized(
        self, transmittance, photon_numbers, rng, ideal_outcomes_d0,
        return_diagnostics, precomputed_randoms
    ) -> DetectionResult:
        p_click_d0, p_click_d1 = self._calculate_p_click_signal(transmittance, photon_numbers)
        components = self._apply_errors_and_clicks(p_click_d0, p_click_d1, ideal_outcomes_d0, rng, precomputed_randoms)
        
        clicks_d0 = components["click0_signal"] | components["click0_dark"]
        clicks_d1 = components["click1_signal"] | components["click1_dark"]

        clicks_d0, clicks_d1, diag = self._resolve_double_clicks(clicks_d0, clicks_d1, rng)
        
        final_diag = replace(diag, 
            misalignment_flips=components["misalignment_flips"], 
            qber_flips=components["qber_flips"]
        )
        return DetectionResult(
            click0=clicks_d0, click1=clicks_d1,
            diagnostics=final_diag if return_diagnostics else None
        )

    def _simulate_sequential(
        self, transmittance, photon_numbers, rng, ideal_outcomes_d0,
        pulse_period_ns, return_diagnostics, precomputed_randoms
    ) -> DetectionResult:
        num_pulses = len(photon_numbers)
        p_click_d0, p_click_d1 = self._calculate_p_click_signal(transmittance, photon_numbers)
        components = self._apply_errors_and_clicks(p_click_d0, p_click_d1, ideal_outcomes_d0, rng, precomputed_randoms)
        
        clicks_d0, clicks_d1 = np.zeros(num_pulses, dtype=bool), np.zeros(num_pulses, dtype=bool)
        dead_time_suppressions = 0
        
        state = self._internal_state
        ap_buf0, ap_sum0, ap_idx0 = state.ap_state_d0
        ap_buf1, ap_sum1, ap_idx1 = state.ap_state_d1
        time_d0 = state.time_since_last_click_ns_d0 if state.time_since_last_click_ns_d0 is not None else float('inf')
        time_d1 = state.time_since_last_click_ns_d1 if state.time_since_last_click_ns_d1 is not None else float('inf')
        p_ap = self.afterpulse_prob
        log1m_pap = np.log1p(-p_ap) if 0 < p_ap < 1 else 0.0

        for i in range(num_pulses):
            time_d0 += pulse_period_ns
            time_d1 += pulse_period_ns
            dead0 = time_d0 < self.dead_time_ns
            dead1 = time_d1 < self.dead_time_ns
            
            click0_ap = False
            if ap_sum0 > 0 and not dead0:
                prob_ap = 1.0 if np.isclose(p_ap, 1.0) else -np.expm1(float(ap_sum0) * log1m_pap)
                click0_ap = rng.random() < prob_ap

            click1_ap = False
            if ap_sum1 > 0 and not dead1:
                prob_ap = 1.0 if np.isclose(p_ap, 1.0) else -np.expm1(float(ap_sum1) * log1m_pap)
                click1_ap = rng.random() < prob_ap

            raw_click_d0 = components["click0_signal"][i] | components["click0_dark"][i] | click0_ap
            raw_click_d1 = components["click1_signal"][i] | components["click1_dark"][i] | click1_ap
            
            final_click_d0 = raw_click_d0 and not dead0
            final_click_d1 = raw_click_d1 and not dead1
            if raw_click_d0 and dead0: dead_time_suppressions += 1
            if raw_click_d1 and dead1: dead_time_suppressions += 1

            clicks_d0[i], clicks_d1[i] = final_click_d0, final_click_d1
            
            if final_click_d0: time_d0 = 0.0
            if final_click_d1: time_d1 = 0.0
            
            if self.afterpulse_memory > 0:
                ap_sum0 += final_click_d0 - ap_buf0[ap_idx0]
                ap_sum1 += final_click_d1 - ap_buf1[ap_idx1]
                ap_buf0[ap_idx0], ap_buf1[ap_idx1] = final_click_d0, final_click_d1
                ap_idx0 = (ap_idx0 + 1) % self.afterpulse_memory
                ap_idx1 = (ap_idx1 + 1) % self.afterpulse_memory

        state.time_since_last_click_ns_d0 = time_d0 if np.isfinite(time_d0) else None
        state.time_since_last_click_ns_d1 = time_d1 if np.isfinite(time_d1) else None
        if self.afterpulse_memory > 0:
            state.ap_state_d0 = (ap_buf0, ap_sum0, ap_idx0)
            state.ap_state_d1 = (ap_buf1, ap_sum1, ap_idx1)

        clicks_d0, clicks_d1, diag = self._resolve_double_clicks(clicks_d0, clicks_d1, rng)
        final_diag = replace(diag, 
            misalignment_flips=components["misalignment_flips"], 
            qber_flips=components["qber_flips"],
            dead_time_suppressions=dead_time_suppressions
        )

        return DetectionResult(
            click0=clicks_d0, click1=clicks_d1,
            diagnostics=final_diag if return_diagnostics else None
        )

def generate_precomputed_randoms(
    num_pulses: int, rng: Generator, sequential_path: bool = False
) -> Dict[str, NDArray[np.float64]]:
    """Helper to generate a reproducible set of random numbers for simulation."""
    keys = ["misalignment", "signal_d0", "signal_d1", "qber", "dark_d0", "dark_d1"]
    randoms = {key: rng.random(size=num_pulses) for key in keys}
    if sequential_path:
        # For sequential mode, after-pulsing RNG is drawn inside the loop
        # This helper provides the vectorized part only.
        pass
    return randoms
