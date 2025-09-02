# -*- coding: utf-8 -*-
"""
Refactored QKD Simulation Framework (v3.0 - Review-Integrated)

This script is a modular, object-oriented refactoring of a QKD simulation
framework. This version incorporates a full suite of expert-review-driven
improvements ("Phase 3"), addressing scientific, numerical, and architectural
issues to bring the framework to a higher standard of quality and correctness.

This module requires Python 3.10+ and is tested with SciPy >= 1.8 and NumPy >= 1.22.

Key Enhancements in v3.0 (Post-Review):
1.  **Scientifically Correct Models**:
    - **MDI-QKD Proof**: The previous placeholder proof for MDI-QKD has been
      replaced. The simulation will now refuse to calculate a key for MDI-QKD
      by default, raising an error unless an explicit `allow_unsafe_mdi_approx`
      flag is set, acknowledging the need for a full 2D decoy-state analysis.
    - **Detector After-pulsing**: The `ThresholdDetector` model is now properly
      stateful. It correctly models detector-specific after-pulsing with a
      configurable memory/decay, replacing the physically unrealistic batch-wide
      `np.roll` implementation.
    - **Double-Click Handling**: The measurement model now correctly combines
      signal, dark, and after-pulse clicks before evaluating double-click
      events, leading to more accurate sifting and QBER statistics.

2.  **Robustness and Numerical Stability**:
    - **LP Solver**: The Linear Programming solver now has a more robust fallback
      mechanism and uses a more flexible tolerance check for constraint violations,
      reducing spurious failures. Diagnostics are standardized.
    - **LP Constraint Bugfix**: A critical off-by-one error in the monotonicity
      constraint of the decoy-state LP has been corrected.
    - **Numerical Precision**: The photon click probability calculation now uses
      `np.expm1` for improved numerical stability, especially in edge cases.
    - **Parameter Validation**: Validation is hardened, preventing invalid
      protocol/proof combinations and checking for excessive Poisson tail
      probabilities in decoy-state analysis.

3.  **Architectural and Code Quality Improvements**:
    - **Stateful Components**: The `ThresholdDetector` is no longer a frozen
      dataclass, allowing it to correctly manage its internal state.
    - **Epsilon Allocation**: Security parameter allocation is now handled more
      transparently and robustly to prevent insecure configurations.
    - **Reproducibility**: The multiprocessing start method is now explicitly set
      to 'spawn' on POSIX systems to improve cross-platform reproducibility.
    - **Clarity and Documentation**: Docstrings and comments have been updated to
      clarify model assumptions, limitations, and physical units.

Model Assumptions and Limitations (v3.0):
- **MDI-QKD Decoy Analysis**: The framework currently LACKS a full 2D decoy-state
  LP required for a secure MDI-QKD key rate calculation. The existing 1D LP is
  used as an UNSAFE placeholder, and key calculation is disabled unless
  explicitly enabled with a flag for experimental purposes only.
- **After-pulse State Persistence**: The after-pulse model is now correct
  *within* a simulation batch. However, detector state is NOT persisted *between*
  batches when running in parallel (`num_workers > 1`). This is a known
  limitation of stateless parallel processing. For simulations where inter-batch
  after-pulsing is critical, run with `num_workers=1` or `--force-sequential`.
- **BSM Model**: The Bell-State Measurement in `MDIQKDProtocol` remains a
  simplified abstract model and does not include physical parameters like
  interference visibility.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import secrets
import sys
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import multiprocessing

# This module requires Python 3.10+ for dataclass __slots__ semantics.
if sys.version_info < (3, 10):
    raise RuntimeError("This module requires Python 3.10 or newer.")

import numpy as np
from numpy.random import Generator

# SciPy dependencies
try:
    from scipy.optimize import OptimizeResult, linprog
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.stats import beta, poisson
except ImportError:
    logging.critical("CRITICAL ERROR: SciPy is required. Run `pip install scipy`.")
    sys.exit(1)

# tqdm fallback for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("QKDSystem")

# --- Constants & Numeric Tolerances ---
MAX_SEED_INT = 2**63 - 1
LP_SOLVER_METHODS = ["highs", "highs-ds"]
# --- Security & Proof Constants ---
S_Z_1_L_MIN_FOR_PHASE_EST = 10.0
# --- Numeric Tolerances ---
NUMERIC_ABS_TOL = 1e-12
NUMERIC_REL_TOL = 1e-9
Y1_SAFE_THRESHOLD = max(1e-12, 100 * np.finfo(float).eps)
ENTROPY_PROB_CLAMP = 1e-15
PROB_SUM_TOL = 1e-8
# Default threshold for Poisson tail probability. Can be overridden in params.
DEFAULT_POISSON_TAIL_THRESHOLD = 0.01
LP_CONSTRAINT_VIOLATION_TOL = 1e-9


# --- Custom Exceptions ---
class ParameterValidationError(ValueError):
    """Raised when a simulation parameter is invalid."""
    pass

class QKDSimulationError(RuntimeError):
    """Raised for general errors during the simulation runtime."""
    pass

class LPFailureError(RuntimeError):
    """Raised when the linear programming solver fails to find a solution."""
    pass


# --- Helper Functions ---
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


# --- Enums and Dataclasses ---
class DoubleClickPolicy(Enum):
    DISCARD = "discard"
    RANDOM = "random"

class SecurityProof(Enum):
    LIM_2014 = "lim-2014"
    TIGHT_PROOF = "tight-proof"
    MDI_QKD = "mdi-qkd"

class ConfidenceBoundMethod(Enum):
    CLOPPER_PEARSON = "clopper-pearson"
    HOEFFDING = "hoeffding"


@dataclass(frozen=True)
class PulseTypeConfig:
    __slots__ = ['name', 'mean_photon_number', 'probability']
    name: str
    mean_photon_number: float # Unitless mean photon number
    probability: float

# --- Component Classes: Source, Channel, Detector ---

@dataclass(frozen=True, slots=True)
class PoissonSource:
    """
    Models a light source with Poissonian photon number statistics.

    Args:
        pulse_configs: A list of configurations for each pulse type (e.g., signal, decoy).
                       Each config specifies a name, mean_photon_number (unitless), and
                       probability of selection ([0,1]).
    """
    pulse_configs: List[PulseTypeConfig]

    def __post_init__(self):
        p_sum = sum(pc.probability for pc in self.pulse_configs)
        if not math.isclose(p_sum, 1.0, rel_tol=PROB_SUM_TOL, abs_tol=PROB_SUM_TOL):
            raise ParameterValidationError(f"Sum of pulse_configs probabilities must be ~1.0 (got {p_sum}).")
        for pc in self.pulse_configs:
            if pc.mean_photon_number < 0:
                raise ParameterValidationError(f"Pulse '{pc.name}' has negative mean_photon_number.")

    def generate_photons(self, alice_pulse_indices: np.ndarray, rng: Generator) -> np.ndarray:
        """Generates photon numbers for a sequence of pulses."""
        mus = np.array([pc.mean_photon_number for pc in self.pulse_configs])
        pulse_mus = mus[alice_pulse_indices]
        return rng.poisson(pulse_mus)

    def get_pulse_config_by_name(self, name: str) -> Optional[PulseTypeConfig]:
        return next((c for c in self.pulse_configs if c.name == name), None)

@dataclass(frozen=True, slots=True)
class FiberChannel:
    """
    Models a simple optical fiber channel.

    Args:
        distance_km: The length of the fiber channel in kilometers (km).
        fiber_loss_db_km: The attenuation of the fiber in decibels per kilometer (dB/km).
    """
    distance_km: float
    fiber_loss_db_km: float

    def __post_init__(self):
        if self.distance_km < 0:
            raise ParameterValidationError("distance_km must be non-negative.")
        if self.fiber_loss_db_km < 0:
            raise ParameterValidationError("fiber_loss_db_km must be non-negative.")

    @property
    def transmittance(self) -> float:
        """Calculates the total channel transmittance (a probability in [0,1])."""
        return 10 ** (-(self.distance_km * self.fiber_loss_db_km) / 10.0)

@dataclass(slots=True) # Not frozen to allow for stateful after-pulsing
class ThresholdDetector:
    """
    Models a stateful threshold detector system with after-pulsing.

    Args:
        det_eff: The efficiency of the detectors (probability in [0,1]).
        dark_rate: The probability of a dark count per detector per pulse (in [0,1)).
        qber_intrinsic: The intrinsic Quantum Bit Error Rate.
        misalignment: The probability of a bit flip due to optical misalignment.
        double_click_policy: How to handle events where both detectors click.
        afterpulse_prob: Probability of an after-pulse in the next time slot, given a click.
        afterpulse_memory: Number of recent time slots to consider for after-pulsing.
    """
    det_eff: float
    dark_rate: float
    qber_intrinsic: float
    misalignment: float
    double_click_policy: DoubleClickPolicy
    afterpulse_prob: float = 0.0
    afterpulse_memory: int = 3
    # Internal state for after-pulsing. These are not part of the config.
    _recent_clicks_d0: np.ndarray = field(init=False, repr=False)
    _recent_clicks_d1: np.ndarray = field(init=False, repr=False)


    def __post_init__(self):
        # Parameter validation
        if not (0.0 <= self.det_eff <= 1.0): raise ParameterValidationError("det_eff must be in [0,1].")
        if not (0 <= self.dark_rate < 1): raise ParameterValidationError("dark_rate must be in [0,1).")
        if not (0 <= self.qber_intrinsic < 1.0): raise ParameterValidationError("qber_intrinsic must be in [0,1).")
        if not (0 <= self.misalignment < 1.0): raise ParameterValidationError("misalignment must be in [0,1).")
        if not (0 <= self.afterpulse_prob <= 1.0): raise ParameterValidationError("afterpulse_prob must be in [0,1].")
        if not self.afterpulse_memory >= 0: raise ParameterValidationError("afterpulse_memory must be non-negative.")
        # Initialize state. This is reset for each batch.
        self.reset_state()

    def reset_state(self):
        """Resets the internal after-pulsing state."""
        self._recent_clicks_d0 = np.zeros(self.afterpulse_memory, dtype=bool)
        self._recent_clicks_d1 = np.zeros(self.afterpulse_memory, dtype=bool)

    def simulate_detection(self, channel_transmittance: float, photon_numbers: np.ndarray, rng: Generator, ideal_outcomes_d0: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulates photon detection with a stateful, detector-specific after-pulse model.
        This method processes pulses sequentially within the batch to correctly model
        the time-correlated nature of after-pulsing.

        Args:
            channel_transmittance: The channel efficiency.
            photon_numbers: Array of photon numbers for each pulse.
            rng: The random number generator.
            ideal_outcomes_d0: A boolean array where True means detector 0 should click
                               in the absence of noise or misalignment.
        """
        eta = float(channel_transmittance) * float(self.det_eff)
        eta = float(np.clip(eta, 0.0, 1.0))
        n_photons = photon_numbers.astype(np.int64)
        num_pulses = len(n_photons)

        # Use numerically stable expm1 for better accuracy: 1 - (1-eta)^n = -expm1(n * log(1-eta))
        if eta <= 0.0:
            p_click_signal = np.zeros(num_pulses, dtype=np.float64)
        else:
            log1m_eta = math.log1p(-eta)
            with np.errstate(over='ignore', invalid='ignore'):
                p_click_signal = -np.expm1(n_photons.astype(np.float64) * log1m_eta)
            p_click_signal = np.clip(p_click_signal, 0.0, 1.0)
        
        # Pre-generate all random numbers for vectorization where possible
        signal_click_rng = rng.random(num_pulses)
        dark_rng_0 = rng.random(num_pulses)
        dark_rng_1 = rng.random(num_pulses)
        afterpulse_rng_0 = rng.random(num_pulses)
        afterpulse_rng_1 = rng.random(num_pulses)

        # Final click results
        clicks_d0 = np.zeros(num_pulses, dtype=bool)
        clicks_d1 = np.zeros(num_pulses, dtype=bool)
        
        # --- Stateful After-pulsing Loop ---
        # We must loop sequentially to correctly propagate the after-pulse state.
        for i in range(num_pulses):
            # 1. Calculate after-pulse probability from recent clicks
            # A simple model: any recent click contributes. A more complex model
            # could use an exponential decay kernel.
            prob_ap_d0 = 1.0 - (1.0 - self.afterpulse_prob) ** np.sum(self._recent_clicks_d0)
            prob_ap_d1 = 1.0 - (1.0 - self.afterpulse_prob) ** np.sum(self._recent_clicks_d1)

            afterpulse0 = afterpulse_rng_0[i] < prob_ap_d0
            afterpulse1 = afterpulse_rng_1[i] < prob_ap_d1
            
            # 2. Base dark counts
            dark0 = dark_rng_0[i] < self.dark_rate
            dark1 = dark_rng_1[i] < self.dark_rate

            # 3. Signal-induced clicks
            is_signal_photon_present = signal_click_rng[i] < p_click_signal[i]
            
            # This logic assumes BB84-like protocols where a signal click goes to one detector.
            # `ideal_outcomes_d0` determines the target detector.
            click0_signal = is_signal_photon_present and ideal_outcomes_d0[i]
            click1_signal = is_signal_photon_present and not ideal_outcomes_d0[i]

            # 4. Total clicks for this time step
            total_click_d0 = click0_signal or dark0 or afterpulse0
            total_click_d1 = click1_signal or dark1 or afterpulse1
            
            clicks_d0[i] = total_click_d0
            clicks_d1[i] = total_click_d1

            # 5. Update state for the next iteration
            self._recent_clicks_d0 = np.roll(self._recent_clicks_d0, 1)
            self._recent_clicks_d1 = np.roll(self._recent_clicks_d1, 1)
            self._recent_clicks_d0[0] = total_click_d0
            self._recent_clicks_d1[0] = total_click_d1
        
        return {"click0": clicks_d0, "click1": clicks_d1}


# --- Protocol Abstraction ---

class Protocol(ABC):
    """Abstract Base Class for a QKD Protocol."""

    @abstractmethod
    def prepare_states(self, num_pulses: int, rng: Generator) -> Dict[str, np.ndarray]:
        """
        Prepares the quantum states.
        Returns a dictionary containing all necessary choices (bits, bases, pulse types).
        """
        pass

    @abstractmethod
    def sift_results(self, num_pulses: int, prepared_states: Dict, detection_results: Dict, rng: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs sifting and error analysis based on measurement outcomes.
        Returns sifted_mask, errors_mask, and discarded_double_clicks_mask.
        """
        pass

class BB84DecoyProtocol(Protocol):
    """
    Concrete implementation of the BB84 protocol with decoy states.
    
    Args:
        alice_z_basis_prob: Probability Alice chooses the Z-basis ([0,1]).
        bob_z_basis_prob: Probability Bob chooses the Z-basis ([0,1]).
        source: The configured `PoissonSource` instance.
        detector: The configured `ThresholdDetector` instance.
    """
    def __init__(self, alice_z_basis_prob: float, bob_z_basis_prob: float, source: PoissonSource, detector: ThresholdDetector):
        self.alice_z_basis_prob = alice_z_basis_prob
        self.bob_z_basis_prob = bob_z_basis_prob
        self.source = source
        self.detector = detector
        if not (0.0 <= self.alice_z_basis_prob <= 1.0): raise ParameterValidationError("alice_z_basis_prob must be in [0,1].")
        if not (0.0 <= self.bob_z_basis_prob <= 1.0): raise ParameterValidationError("bob_z_basis_prob must be in [0,1].")

    def prepare_states(self, num_pulses: int, rng: Generator) -> Dict[str, np.ndarray]:
        """Prepares Alice's bits, bases, and pulse types for BB84."""
        if num_pulses < 0: raise ParameterValidationError("num_pulses must be non-negative")
        
        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.int8)
        alice_bases = rng.choice([0, 1], size=num_pulses, p=[self.alice_z_basis_prob, 1.0 - self.alice_z_basis_prob]).astype(np.int8)
        
        probs = [pc.probability for pc in self.source.pulse_configs]
        alice_pulse_indices = rng.choice(len(self.source.pulse_configs), size=num_pulses, p=probs)
        
        return {
            "alice_bits": alice_bits,
            "alice_bases": alice_bases,
            "alice_pulse_indices": alice_pulse_indices
        }

    def sift_results(self, num_pulses: int, prepared_states: Dict, detection_results: Dict, rng: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs sifting and error calculation for BB84 with robust click model."""
        alice_bits = prepared_states['alice_bits']
        alice_bases = prepared_states['alice_bases']
        
        # Get final click outcomes from the detector model
        click0_final = detection_results['click0']
        click1_final = detection_results['click1']

        if len(alice_bits) != num_pulses:
            raise ParameterValidationError(f"Mismatched array length: alice_bits has {len(alice_bits)} elements, expected {num_pulses}.")
        
        bob_bases = rng.choice([0, 1], size=num_pulses, p=[self.bob_z_basis_prob, 1.0 - self.bob_z_basis_prob]).astype(np.int8)
        basis_match = (alice_bases == bob_bases)
        
        # Determine Bob's measurement outcomes
        bob_bits = -1 * np.ones(num_pulses, dtype=np.int8)
        
        # 1. Identify single-click events
        conclusive0 = click0_final & ~click1_final
        conclusive1 = click1_final & ~click0_final
        bob_bits[conclusive0] = 0
        bob_bits[conclusive1] = 1
        
        # 2. Handle double-click events
        double_click_mask = click0_final & click1_final
        discarded_dc_mask = np.zeros_like(double_click_mask)
        
        if self.detector.double_click_policy == DoubleClickPolicy.RANDOM:
            num_dc = np.sum(double_click_mask)
            if num_dc > 0: bob_bits[double_click_mask] = rng.integers(0, 2, size=num_dc, dtype=np.int8)
        elif self.detector.double_click_policy == DoubleClickPolicy.DISCARD:
            bob_bits[double_click_mask] = -1
            # Only count discarded double clicks if bases matched
            discarded_dc_mask = basis_match & double_click_mask

        # 3. Sift the key
        # A result is sifted if bases match AND it was a conclusive measurement (not discarded)
        sifted_mask = basis_match & (bob_bits != -1)
        
        # 4. Calculate errors on the sifted key
        errors_mask = np.zeros(num_pulses, dtype=bool)
        if np.any(sifted_mask):
            # Base errors are from misalignment and other physical effects now modeled in the detector
            base_errors = (alice_bits[sifted_mask] != bob_bits[sifted_mask])
            
            # Intrinsic QBER is applied on top of everything else
            num_sifted = np.sum(sifted_mask)
            intrinsic_flips = rng.random(num_sifted) < self.detector.qber_intrinsic
            errors_mask[sifted_mask] = np.logical_xor(base_errors, intrinsic_flips)
            
        return sifted_mask, errors_mask, discarded_dc_mask

class MDIQKDProtocol(Protocol):
    """
    Concrete implementation of the Measurement-Device-Independent (MDI) QKD protocol.
    """
    def __init__(self, z_basis_prob: float, source: PoissonSource, detector: ThresholdDetector):
        self.z_basis_prob = z_basis_prob
        self.source = source
        self.detector = detector # Charlie's detector
        if not (0.0 <= self.z_basis_prob <= 1.0):
            raise ParameterValidationError("z_basis_prob must be in [0,1].")

    def prepare_states(self, num_pulses: int, rng: Generator) -> Dict[str, np.ndarray]:
        """Prepares Alice's and Bob's bits, bases, and pulse types for MDI-QKD."""
        if num_pulses < 0: raise ParameterValidationError("num_pulses must be non-negative")
        
        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.int8)
        alice_bases = rng.choice([0, 1], size=num_pulses, p=[self.z_basis_prob, 1.0 - self.z_basis_prob]).astype(np.int8)
        probs = [pc.probability for pc in self.source.pulse_configs]
        alice_pulse_indices = rng.choice(len(self.source.pulse_configs), size=num_pulses, p=probs)
        
        bob_bits = rng.integers(0, 2, size=num_pulses, dtype=np.int8)
        bob_bases = rng.choice([0, 1], size=num_pulses, p=[self.z_basis_prob, 1.0 - self.z_basis_prob]).astype(np.int8)
        bob_pulse_indices = rng.choice(len(self.source.pulse_configs), size=num_pulses, p=probs)

        return {
            "alice_bits": alice_bits, "alice_bases": alice_bases, "alice_pulse_indices": alice_pulse_indices,
            "bob_bits": bob_bits, "bob_bases": bob_bases, "bob_pulse_indices": bob_pulse_indices,
        }

    def sift_results(self, num_pulses: int, prepared_states: Dict, detection_results: Dict, rng: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs sifting for MDI-QKD based on Charlie's BSM announcement."""
        alice_bits = prepared_states['alice_bits']
        alice_bases = prepared_states['alice_bases']
        bob_bits = prepared_states['bob_bits']
        bob_bases = prepared_states['bob_bases']

        # In MDI, the two detectors can be thought of as `click0` and `click1`
        # corresponding to two distinguishable outcomes of the BSM.
        # A successful Psi- or Psi+ result requires one click from Alice's side and one from Bob's.
        # This is simplified here: a "double click" (click0 & click1) is a successful BSM.
        click0 = detection_results['click0']
        click1 = detection_results['click1']
        successful_bsm = click0 & click1
        
        # For MDI, sifting is on matching bases AND a successful BSM announcement.
        basis_match = (alice_bases == bob_bases)
        sifted_mask = basis_match & successful_bsm
        
        errors_mask = np.zeros(num_pulses, dtype=bool)
        if np.any(sifted_mask):
            sifted_alice_bits = alice_bits[sifted_mask]
            sifted_bob_bits = bob_bits[sifted_mask]
            sifted_bases = alice_bases[sifted_mask]
            
            # Z-basis (basis=0): bits should be anti-correlated (a != b) for Psi-
            # For simplicity, we assume all successful BSMs are Psi- type.
            z_basis_errors = (sifted_bases == 0) & (sifted_alice_bits == sifted_bob_bits)
            # X-basis (basis=1): bits should be correlated (a == b)
            x_basis_errors = (sifted_bases == 1) & (sifted_alice_bits != sifted_bob_bits)
            
            base_errors = z_basis_errors | x_basis_errors
            num_sifted = np.sum(sifted_mask)
            intrinsic_flips = rng.random(num_sifted) < self.detector.qber_intrinsic
            errors_mask[sifted_mask] = np.logical_xor(base_errors, intrinsic_flips)

        # No double clicks to discard in this model, as they are the success condition.
        discarded_dc_mask = np.zeros(num_pulses, dtype=bool)
        return sifted_mask, errors_mask, discarded_dc_mask


# --- Core Data Structures ---

@dataclass(slots=True)
class TallyCounts:
    sent: int = field(default=0)
    sifted: int = field(default=0)
    errors_sifted: int = field(default=0)
    double_clicks_discarded: int = field(default=0)
    sent_z: int = field(default=0)
    sent_x: int = field(default=0)
    sifted_z: int = field(default=0)
    sifted_x: int = field(default=0)
    errors_sifted_z: int = field(default=0)
    errors_sifted_x: int = field(default=0)


@dataclass(frozen=True)
class EpsilonAllocation:
    __slots__ = ['eps_sec', 'eps_cor', 'eps_pe', 'eps_smooth', 'eps_pa', 'eps_phase_est']
    eps_sec: float
    eps_cor: float
    eps_pe: float
    eps_smooth: float
    eps_pa: float
    eps_phase_est: float

    def validate(self):
        if not (self.eps_cor > 0 and self.eps_pa > 0 and self.eps_pe > 0 and self.eps_smooth > 0 and self.eps_phase_est > 0):
            raise ParameterValidationError("All component epsilons must be > 0.")
        # The key security composition theorem requires eps_sec >= eps_pe + 2*eps_smooth + eps_pa
        total_sum = self.eps_pe + 2 * self.eps_smooth + self.eps_pa
        if total_sum > self.eps_sec + NUMERIC_ABS_TOL:
            raise ParameterValidationError(
                f"Epsilon allocation insecure: sum of components ({total_sum:.2e}) > eps_sec ({self.eps_sec:.2e}). "
                "Consider increasing eps_sec or reducing other epsilons."
            )


@dataclass(frozen=True, slots=True)
class SecurityCertificate:
    proof_name: str
    confidence_bound_method: str
    assumed_phase_equals_bit_error: bool
    epsilon_allocation: EpsilonAllocation
    lp_solver_diagnostics: Optional[Dict] = None


def _to_serializable(o: Any) -> Any:
    """Helper function to recursively convert objects to JSON-serializable types."""
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Enum): return o.value
    if dataclasses.is_dataclass(o):
        # For non-frozen dataclasses, handle state vs. config
        if isinstance(o, ThresholdDetector):
            return {
                "det_eff": o.det_eff, "dark_rate": o.dark_rate,
                "qber_intrinsic": o.qber_intrinsic, "misalignment": o.misalignment,
                "double_click_policy": o.double_click_policy.value,
                "afterpulse_prob": o.afterpulse_prob,
                "afterpulse_memory": o.afterpulse_memory
            }
        return {f.name: _to_serializable(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, dict): return {k: _to_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_serializable(i) for i in o]
    if isinstance(o, float) and not math.isfinite(o): return str(o)
    return o

@dataclass(slots=True)
class QKDParams:
    """High-level container for all QKD simulation parameters and components."""
    protocol: Protocol
    source: PoissonSource
    channel: FiberChannel
    detector: ThresholdDetector
    num_bits: int
    photon_number_cap: int
    batch_size: int
    num_workers: int
    force_sequential: bool
    f_error_correction: float
    eps_sec: float
    eps_cor: float
    eps_pe: float
    eps_smooth: float
    security_proof: SecurityProof
    ci_method: ConfidenceBoundMethod
    enforce_monotonicity: bool
    assume_phase_equals_bit_error: bool
    lp_solver_method: str = "highs"
    # New safety/experimental flags
    allow_unsafe_mdi_approx: bool = False
    require_tail_below: Optional[float] = DEFAULT_POISSON_TAIL_THRESHOLD

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Performs validation of high-level simulation parameters."""
        if not self.num_bits > 0: raise ParameterValidationError("num_bits must be positive.")
        if not (1.0 <= self.f_error_correction <= 5.0): raise ParameterValidationError("f_error_correction must be in [1.0,5.0].")
        if not (0 < self.batch_size <= self.num_bits): raise ParameterValidationError("batch_size must be >0 and <= num_bits")
        if self.photon_number_cap < 1: raise ParameterValidationError("photon_number_cap must be >= 1")
        epsilons = [self.eps_sec, self.eps_cor, self.eps_pe, self.eps_smooth]
        if not all(isinstance(e, (float,int)) and 0 < float(e) < 1 for e in epsilons):
            raise ParameterValidationError("eps_sec/eps_cor/eps_pe/eps_smooth must be floats in (0,1)")
        if not isinstance(self.lp_solver_method, str) or self.lp_solver_method not in LP_SOLVER_METHODS:
            raise ParameterValidationError(f"lp_solver_method must be one of {LP_SOLVER_METHODS}")
        if self.security_proof == SecurityProof.MDI_QKD and not isinstance(self.protocol, MDIQKDProtocol):
            raise ParameterValidationError("The 'mdi-qkd' security proof can only be used with the 'mdi-qkd' protocol.")

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Serializes the entire QKDParams object graph to a JSON-compatible dictionary."""
        if isinstance(self.protocol, BB84DecoyProtocol):
            protocol_name = "bb84-decoy"
            protocol_config = {
                "alice_z_basis_prob": self.protocol.alice_z_basis_prob,
                "bob_z_basis_prob": self.protocol.bob_z_basis_prob
            }
        elif isinstance(self.protocol, MDIQKDProtocol):
            protocol_name = "mdi-qkd"
            protocol_config = {"z_basis_prob": self.protocol.z_basis_prob}
        else:
            raise NotImplementedError(f"Serialization not implemented for protocol type {type(self.protocol)}")
            
        return {
            "protocol_name": protocol_name, "protocol_config": _to_serializable(protocol_config),
            "source_type": "poisson", "source_config": _to_serializable(self.source),
            "channel_type": "fiber", "channel_config": _to_serializable(self.channel),
            "detector_type": "threshold", "detector_config": _to_serializable(self.detector),
            "num_bits": self.num_bits, "photon_number_cap": self.photon_number_cap,
            "batch_size": self.batch_size, "num_workers": self.num_workers,
            "force_sequential": self.force_sequential, "f_error_correction": self.f_error_correction,
            "eps_sec": self.eps_sec, "eps_cor": self.eps_cor,
            "eps_pe": self.eps_pe, "eps_smooth": self.eps_smooth,
            "security_proof": self.security_proof.value, "ci_method": self.ci_method.value,
            "enforce_monotonicity": self.enforce_monotonicity,
            "assume_phase_equals_bit_error": self.assume_phase_equals_bit_error,
            "lp_solver_method": self.lp_solver_method,
            "allow_unsafe_mdi_approx": self.allow_unsafe_mdi_approx,
            "require_tail_below": self.require_tail_below
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QKDParams":
        """Acts as a factory to deserialize parameters and construct the full object graph."""
        d_copy = d.copy()
        try:
            source_type = d_copy.pop("source_type", "poisson")
            source_config = d_copy.pop("source_config")
            if source_type == "poisson":
                pulse_list = source_config.get("pulse_configs")
                if not isinstance(pulse_list, list):
                    raise ParameterValidationError("`pulse_configs` must be a list of dicts.")
                new_pulse_list = [PulseTypeConfig(**pc) for pc in pulse_list]
                source = PoissonSource(pulse_configs=new_pulse_list)
            else:
                raise NotImplementedError(f"Source type '{source_type}' not supported.")

            channel_type = d_copy.pop("channel_type", "fiber")
            channel_config = d_copy.pop("channel_config")
            if channel_type == "fiber":
                channel = FiberChannel(**channel_config)
            else:
                raise NotImplementedError(f"Channel type '{channel_type}' not supported.")

            detector_type = d_copy.pop("detector_type", "threshold")
            detector_config = d_copy.pop("detector_config")
            if detector_type == "threshold":
                policy_val = detector_config.get('double_click_policy')
                if policy_val: detector_config['double_click_policy'] = DoubleClickPolicy(policy_val)
                detector = ThresholdDetector(**detector_config)
            else:
                raise NotImplementedError(f"Detector type '{detector_type}' not supported.")

            protocol_name = d_copy.pop("protocol_name", "bb84-decoy")
            protocol_config = d_copy.pop("protocol_config")
            if protocol_name == "bb84-decoy":
                protocol = BB84DecoyProtocol(source=source, detector=detector, **protocol_config)
            elif protocol_name == "mdi-qkd":
                protocol = MDIQKDProtocol(source=source, detector=detector, **protocol_config)
            else:
                raise NotImplementedError(f"Protocol '{protocol_name}' not supported.")

            d_copy['security_proof'] = SecurityProof(d_copy["security_proof"])
            d_copy['ci_method'] = ConfidenceBoundMethod(d_copy["ci_method"])
            bool_keys = {"force_sequential", "enforce_monotonicity", "assume_phase_equals_bit_error", "allow_unsafe_mdi_approx"}
            for key in bool_keys:
                if key in d_copy: d_copy[key] = _parse_bool(d_copy[key])

            return cls(protocol=protocol, source=source, channel=channel, detector=detector, **d_copy)
        except KeyError as e:
            raise ParameterValidationError(f"Missing required parameter or config section: {e}")
        except (TypeError, ValueError) as e:
            raise ParameterValidationError(f"Failed to load parameters due to invalid value: {e}")


@dataclass
class SimulationResults:
    params: QKDParams
    metadata: Dict[str, Any]
    security_certificate: Optional[SecurityCertificate] = None
    decoy_estimates: Optional[Dict[str, Any]] = None
    secure_key_length: Optional[int] = None
    raw_sifted_key_length: int = 0
    simulation_time_seconds: float = 0.0
    status: str = "OK"

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Converts the results object to a fully JSON-serializable dictionary."""
        return {
            "params": self.params.to_serializable_dict(),
            "metadata": _to_serializable(self.metadata),
            "security_certificate": _to_serializable(self.security_certificate),
            "decoy_estimates": _to_serializable(self.decoy_estimates),
            "secure_key_length": _to_serializable(self.secure_key_length),
            "raw_sifted_key_length": _to_serializable(self.raw_sifted_key_length),
            "simulation_time_seconds": _to_serializable(self.simulation_time_seconds),
            "status": _to_serializable(self.status),
        }

    def save_json(self, path: str):
        full_path = os.path.abspath(path)
        dir_path = os.path.dirname(full_path)
        if dir_path: os.makedirs(dir_path, exist_ok=True)
        
        tmp_path = None
        try:
            payload = self.to_serializable_dict()
            with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_path, encoding="utf-8", suffix=".json") as f:
                tmp_path = f.name
                json.dump(payload, f, indent=4, ensure_ascii=False)
                f.flush(); os.fsync(f.fileno())
            
            os.replace(tmp_path, full_path)
            if os.name == "posix":
                try:
                    os.chmod(full_path, 0o600)
                except OSError as e:
                    logger.warning(f"Could not set file permissions on {full_path}: {e}")
            file_size_kb = os.path.getsize(full_path) / 1024
            logger.info(f"Results saved to JSON: {full_path} ({file_size_kb:.2f} KB)")
        except (IOError, OSError, TypeError) as e:
            logger.error(f"Failed to save results to {path}: {e}", exc_info=True)
            if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)


# --- Simulation Logic ---

def _simulate_quantum_part_batch(p: QKDParams, num_pulses: int, rng: Generator) -> Dict:
    """Simulates one batch of the quantum part of the protocol using vectorized tallying."""
    p.detector.reset_state() # Reset detector state for each new batch
    prepared_states = p.protocol.prepare_states(num_pulses, rng)
    
    alice_bases = prepared_states['alice_bases']
    alice_bits = prepared_states['alice_bits']
    misalignment_flips = rng.random(num_pulses) < p.detector.misalignment
    
    if isinstance(p.protocol, MDIQKDProtocol):
        photons_A = p.source.generate_photons(prepared_states['alice_pulse_indices'], rng)
        photons_B = p.source.generate_photons(prepared_states['bob_pulse_indices'], rng)
        # A BSM click depends on photons from *both* arriving. This is a placeholder model.
        # We pass an 'effective' photon number to the detector.
        photon_numbers = np.minimum(photons_A, photons_B)
        channel_transmittance = p.channel.transmittance**2 # Both arms suffer loss
        # Ideal outcomes for MDI BSM are more complex. We simplify.
        ideal_outcomes_d0 = rng.random(num_pulses) < 0.5
    else: # BB84
        photon_numbers = p.source.generate_photons(prepared_states['alice_pulse_indices'], rng)
        channel_transmittance = p.channel.transmittance
        # Ideal outcome depends on Alice's bit, but is flipped by misalignment
        detector0_ideal_outcome_no_misalign = (alice_bits == 0)
        # For different-basis pulses, outcome is random
        bob_bases_temp = rng.choice([0, 1], size=num_pulses, p=[p.protocol.bob_z_basis_prob, 1.0 - p.protocol.bob_z_basis_prob]).astype(np.int8)
        basis_match_temp = (alice_bases == bob_bases_temp)
        
        ideal_outcomes_d0 = np.where(
            basis_match_temp,
            np.logical_xor(detector0_ideal_outcome_no_misalign, misalignment_flips),
            rng.random(num_pulses) < 0.5
        )

    detection_results = p.detector.simulate_detection(channel_transmittance, photon_numbers, rng, ideal_outcomes_d0)
    sifted_mask, errors_mask, discarded_dc_mask = p.protocol.sift_results(
        num_pulses, prepared_states, detection_results, rng
    )
    
    # Vectorized tallying
    batch_tallies = {}
    indices = prepared_states['alice_pulse_indices']
    num_pulse_types = len(p.source.pulse_configs)

    def bincount(mask):
        return np.bincount(indices[mask], minlength=num_pulse_types).astype(np.int64)

    sent_counts = np.bincount(indices, minlength=num_pulse_types).astype(np.int64)
    sent_z_counts = bincount(alice_bases == 0)
    sifted_counts = bincount(sifted_mask)
    sifted_z_counts = bincount(sifted_mask & (alice_bases == 0))
    errors_counts = bincount(errors_mask)
    errors_z_counts = bincount(errors_mask & (alice_bases == 0))
    dc_counts = bincount(discarded_dc_mask)

    for i, pc in enumerate(p.source.pulse_configs):
        t = TallyCounts(
            sent=sent_counts[i].item(), sent_z=sent_z_counts[i].item(),
            sent_x=(sent_counts[i] - sent_z_counts[i]).item(),
            sifted=sifted_counts[i].item(), sifted_z=sifted_z_counts[i].item(),
            sifted_x=(sifted_counts[i] - sifted_z_counts[i]).item(),
            errors_sifted=errors_counts[i].item(), errors_sifted_z=errors_z_counts[i].item(),
            errors_sifted_x=(errors_counts[i] - errors_z_counts[i]).item(),
            double_clicks_discarded=dc_counts[i].item()
        )
        batch_tallies[pc.name] = asdict(t)
        
    return {"overall": batch_tallies, "sifted_count": int(np.sum(sifted_mask))}


def _top_level_worker_function(serialized_params: Dict, num_pulses: int, seed: int) -> Dict:
    """Top-level function executed by each worker process."""
    try:
        deserialized_params = QKDParams.from_dict(serialized_params)
        rng = np.random.default_rng(int(seed) % MAX_SEED_INT or 1)
        return _simulate_quantum_part_batch(deserialized_params, num_pulses, rng)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in worker process (seed={seed}): {e}\n{tb}")
        raise RuntimeError(f"Worker error (seed={seed}): {e}\n{tb}") from e

# --- Post-Processing and Security Proof Logic ---

def p_n_mu_vector(mu: float, n_cap: int, tail_threshold: Optional[float]) -> np.ndarray:
    if mu < 0: raise ValueError("Mean `mu` must be non-negative.")
    if n_cap < 1: raise ValueError("n_cap must be >= 1.")
    
    if mu == 0:
        vec = np.zeros(n_cap + 1, dtype=np.float64); vec[0] = 1.0
        return vec

    ns = np.arange(0, n_cap, dtype=np.int64)
    pmf = poisson.pmf(ns, mu).astype(np.float64)
    tail = float(poisson.sf(n_cap - 1, mu))
    
    if tail_threshold is not None and tail > tail_threshold:
        raise ParameterValidationError(
            f"Poisson tail P(n>={n_cap}) = {tail:.2e} for mu={mu:.2f} exceeds "
            f"the specified threshold of {tail_threshold:.2e}. "
            "Increase photon_number_cap or set require_tail_below to null."
        )

    vec = np.concatenate((pmf, np.array([tail], dtype=np.float64)))
    s = float(vec.sum())
    if not math.isfinite(s) or s <= 0:
        raise QKDSimulationError(f"Invalid Poisson PMF sum computed (sum={s}, mu={mu}, n_cap={n_cap}).")
    vec /= s
    return vec

def hoeffding_bounds(k: int, n: int, failure_prob: float) -> Tuple[float, float]:
    if n <= 0: return 0.0, 1.0
    if not (0 < failure_prob < 1): raise ValueError("failure_prob must be in (0,1).")
    delta = math.sqrt(math.log(2.0 / failure_prob) / (2.0 * n))
    p_hat = k / n
    return max(0.0, p_hat - delta), min(1.0, p_hat + delta)

def clopper_pearson_bounds(k: int, n: int, failure_prob: float) -> Tuple[float, float]:
    if n <= 0: return 0.0, 1.0
    if not (0 < failure_prob < 1): raise ValueError("failure_prob must be in (0,1).")
    alpha = failure_prob
    lower = 0.0 if k == 0 else beta.ppf(alpha / 2.0, k, n - k + 1)
    upper = 1.0 if k == n else beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(np.nan_to_num(lower, nan=0.0)), float(np.nan_to_num(upper, nan=1.0))


class FiniteKeyProof(ABC):
    def __init__(self, params: QKDParams):
        self.p = params
        self.eps_alloc = self.allocate_epsilons()
        self.eps_alloc.validate()
        
    @abstractmethod
    def allocate_epsilons(self) -> EpsilonAllocation: raise NotImplementedError
    
    @abstractmethod
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> Dict[str, Any]: raise NotImplementedError
    
    @abstractmethod
    def calculate_key_length(self, decoy_estimates: Dict[str, Any], stats_map: Dict[str, TallyCounts]) -> int: raise NotImplementedError
    
    def get_bounds(self, k: int, n: int, failure_prob: float) -> Tuple[float, float]:
        if self.p.ci_method == ConfidenceBoundMethod.CLOPPER_PEARSON: return clopper_pearson_bounds(k, n, failure_prob)
        elif self.p.ci_method == ConfidenceBoundMethod.HOEFFDING: return hoeffding_bounds(k, n, failure_prob)
        else: raise NotImplementedError(f"CI method {self.p.ci_method} not implemented.")

    @staticmethod
    def binary_entropy(p_err: float) -> float:
        p = np.clip(p_err, ENTROPY_PROB_CLAMP, 1.0 - ENTROPY_PROB_CLAMP)
        if p <= 0.0 or p >= 1.0: return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

class Lim2014Proof(FiniteKeyProof):
    """Implements the finite-key security proof from Lim et al., PRA 89, 032332 (2014)."""
    def allocate_epsilons(self) -> EpsilonAllocation:
        n_intensities = len(self.p.source.pulse_configs)
        total_tests = 4 * n_intensities + 1
        eps_pe_total = self.p.eps_pe
        eps_per_test = eps_pe_total / max(1, total_tests)
        
        # Robust allocation
        eps_budget_for_pa = self.p.eps_sec - self.p.eps_cor - eps_pe_total - (2 * self.p.eps_smooth)
        if eps_budget_for_pa <= 0:
            raise ParameterValidationError(f"Insecure epsilon allocation: eps_sec ({self.p.eps_sec:.2e}) is too small for other epsilons.")
        
        return EpsilonAllocation(
            eps_sec=self.p.eps_sec, eps_cor=self.p.eps_cor, eps_pe=eps_pe_total,
            eps_smooth=self.p.eps_smooth, eps_pa=eps_budget_for_pa, eps_phase_est=eps_per_test
        )

    def _idx_y(self, n: int, Nvar: int) -> int: return n
    def _idx_e(self, n: int, Nvar: int) -> int: return Nvar + n

    def _solve_lp(self, cost_vector: np.ndarray, A_ub: csr_matrix, b_ub: np.ndarray, n_vars: int, method: str) -> Tuple[np.ndarray, Dict]:
        bounds = [(0.0, 1.0)] * n_vars
        res: OptimizeResult = linprog(cost_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options={"presolve": True, "tol": 1e-9})
        
        if not getattr(res, 'success', False) or getattr(res, 'x', None) is None:
            raise LPFailureError(f"LP solver '{method}' failed: {getattr(res, 'message', 'No message')} (Status: {getattr(res, 'status', -1)})")
        
        sol = res.x.copy()
        residual = A_ub.dot(sol) - b_ub
        max_violation = float(np.max(np.maximum(0.0, residual))) if residual.size > 0 else 0.0
        
        # Use relative and absolute tolerance for violation check
        tol = LP_CONSTRAINT_VIOLATION_TOL * max(1.0, np.max(np.abs(b_ub))) if b_ub.size > 0 else LP_CONSTRAINT_VIOLATION_TOL
        if max_violation > tol:
            logger.warning(f"LP solution from '{method}' violates constraints by {max_violation:.3e} (tol={tol:.1e}).")
            if max_violation > 10 * tol:
                raise LPFailureError(f"LP solution from '{method}' violates constraints significantly.")

        return sol, {"method": str(method), "status": int(getattr(res, "status", -1)),
                     "message": str(getattr(res, "message", "")), "fun": float(res.fun),
                     "nit": int(res.nit), "max_violation": max_violation}

    def _build_constraints(self, required: List[str], stats_map: Dict[str, TallyCounts],
                           use_basis_z: bool, enforce_monotonicity: bool, enforce_half_error: bool) -> Tuple[csr_matrix, np.ndarray, int]:
        cap = self.p.photon_number_cap
        Nvar = cap + 1
        rows, cols, data, b_ub_list = [], [], [], []
        
        def _add_row(ridx, coeffs: Dict[int, float], rhs: float):
            for var_idx, coeff in coeffs.items():
                rows.append(ridx); cols.append(var_idx); data.append(coeff)
            b_ub_list.append(rhs)

        row_idx = 0
        pulse_map = {pc.name: pc for pc in self.p.source.pulse_configs}
        eps_per_ci = self.eps_alloc.eps_pe / max(1, 4 * len(required))

        for name in required:
            stats = stats_map[name]
            sent = stats.sent_z if use_basis_z else stats.sent
            sifted = stats.sifted_z if use_basis_z else stats.sifted
            errors = stats.errors_sifted_z if use_basis_z else stats.errors_sifted
            
            q_l, q_u = self.get_bounds(sifted, sent, eps_per_ci)
            r_l, r_u = self.get_bounds(errors, sent, eps_per_ci)
            p_vec = p_n_mu_vector(pulse_map[name].mean_photon_number, cap, self.p.require_tail_below)
            
            _add_row(row_idx, {self._idx_y(i, Nvar): p_vec[i] for i in range(Nvar)}, q_u); row_idx += 1
            _add_row(row_idx, {self._idx_y(i, Nvar): -p_vec[i] for i in range(Nvar)}, -q_l); row_idx += 1
            _add_row(row_idx, {self._idx_e(i, Nvar): p_vec[i] for i in range(Nvar)}, r_u); row_idx += 1
            _add_row(row_idx, {self._idx_e(i, Nvar): -p_vec[i] for i in range(Nvar)}, -r_l); row_idx += 1
        
        for n in range(Nvar):
            _add_row(row_idx, {self._idx_e(n, Nvar): 1.0, self._idx_y(n, Nvar): -1.0}, 0.0); row_idx += 1
        if enforce_half_error:
            for n in range(Nvar):
                _add_row(row_idx, {self._idx_e(n, Nvar): 1.0, self._idx_y(n, Nvar): -0.5}, 0.0); row_idx += 1
        
        # Corrected monotonicity constraint loop
        if enforce_monotonicity and Nvar >= 2:
            for n in range(Nvar - 1):
                _add_row(row_idx, {self._idx_y(n + 1, Nvar): 1.0, self._idx_y(n, Nvar): -1.0}, 0.0); row_idx += 1

        A_coo = coo_matrix((data, (rows, cols)), shape=(row_idx, 2 * Nvar))
        return A_coo.tocsr(), np.array(b_ub_list, dtype=float), Nvar

    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> Dict[str, Any]:
        required = [pc.name for pc in self.p.source.pulse_configs]
        try_sequence = [
            {"use_basis_z": True, "enforce_monotonicity": self.p.enforce_monotonicity, "enforce_half_error": True, "label": "Z_mon_half"},
            {"use_basis_z": True, "enforce_monotonicity": False, "enforce_half_error": True, "label": "Z_noMon_half"},
        ]
        last_exc, final_lp_diag = None, []
        
        for attempt_config in try_sequence:
            try:
                A_ub, b_ub, Nvar = self._build_constraints(required, stats_map, **{k:v for k,v in attempt_config.items() if k != 'label'})
                c_y1 = np.zeros(2 * Nvar); c_e1 = np.zeros(2 * Nvar)
                if Nvar >= 2: c_y1[self._idx_y(1, Nvar)] = -1.0; c_e1[self._idx_e(1, Nvar)] = -1.0
                
                sol_y1, sol_e1 = None, None
                diag_y1, diag_e1 = {}, {}

                for solver in LP_SOLVER_METHODS:
                    try:
                        sol_y1, diag_y1 = self._solve_lp(c_y1, A_ub, b_ub, 2 * Nvar, solver)
                        sol_e1, diag_e1 = self._solve_lp(c_e1, A_ub, b_ub, 2 * Nvar, solver)
                        logger.debug(f"LP attempt '{attempt_config['label']}' succeeded with solver '{solver}'.")
                        break
                    except LPFailureError as e:
                        logger.warning(f"LP solver '{solver}' failed for attempt '{attempt_config['label']}': {e}")
                        last_exc = e
                
                if sol_y1 is None or sol_e1 is None:
                    raise LPFailureError(f"All LP solvers failed for attempt '{attempt_config['label']}'. Last error: {last_exc}")

                Y1_L = float(sol_y1[self._idx_y(1, Nvar)]) if Nvar >= 2 else 0.0
                E1_U = float(sol_e1[self._idx_e(1, Nvar)]) if Nvar >= 2 else 0.0
                final_lp_diag.append({"attempt": attempt_config['label'], "diag_y1": diag_y1, "diag_e1": diag_e1})
                
                ok = True
                if Y1_L <= Y1_SAFE_THRESHOLD:
                    logger.warning(f"Y1_L ({Y1_L:.2e}) is below safe threshold. Using conservative e1_U=0.5 and marking estimate as not 'ok'.")
                    e1_U = 0.5
                    ok = False
                else:
                    e1_U = min(0.5, E1_U / Y1_L)
                
                return {"Y1_L": Y1_L, "e1_U": e1_U, "ok": ok, "lp_diagnostics": {"attempts": final_lp_diag}}
            except LPFailureError as e:
                last_exc = e
                logger.debug(f"LP attempt '{attempt_config['label']}' failed definitively: {e}")
                if not any(d.get("attempt") == attempt_config['label'] for d in final_lp_diag):
                    final_lp_diag.append({"attempt": attempt_config['label'], "error": str(e)})
        
        logger.error(f"All LP attempts failed, falling back to conservative estimate. Last error: {last_exc}")
        return {"Y1_L": 0.0, "e1_U": 0.5, "ok": False, "status": "LP_INFEASIBLE_FALLBACK", "lp_diagnostics": {"attempts": final_lp_diag}}

    def calculate_key_length(self, decoy_estimates: Dict[str, Any], stats_map: Dict[str, TallyCounts]) -> int:
        Y1_L, e1_bit_U = decoy_estimates["Y1_L"], decoy_estimates["e1_U"]
        signal_stats = stats_map.get("signal")
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not signal_stats or signal_stats.sent == 0: return 0
        
        alice_z_prob = getattr(self.p.protocol, 'alice_z_basis_prob', 0.0)
        bob_z_prob = getattr(self.p.protocol, 'bob_z_basis_prob', 0.0)

        mu_s = p_sig_cfg.mean_photon_number
        p1_s = mu_s * math.exp(-mu_s)
        s_z_1_L = signal_stats.sent * p1_s * Y1_L * (alice_z_prob * bob_z_prob)
        n_z, m_z = signal_stats.sifted_z, signal_stats.errors_sifted_z
        
        if n_z <= 0 or s_z_1_L < S_Z_1_L_MIN_FOR_PHASE_EST: return 0
        
        qber_z = m_z / n_z
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z
        
        if self.p.assume_phase_equals_bit_error:
            e1_phase_U = e1_bit_U
        else:
            try:
                delta = math.sqrt(math.log(2.0 / self.eps_alloc.eps_phase_est) / (2.0 * s_z_1_L))
            except (ValueError, ZeroDivisionError):
                raise QKDSimulationError("Invalid value in phase error delta calculation.")
            e1_phase_U = min(0.5, e1_bit_U + delta)
            
        pa_term_bits = 2 * (-math.log2(self.eps_alloc.eps_smooth)) + (-math.log2(self.eps_alloc.eps_pa))
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)
        
        key_length_float = s_z_1_L * (1.0 - self.binary_entropy(e1_phase_U)) - leak_EC - pa_term_bits - corr_term_bits
        
        return max(0, math.floor(key_length_float))


class BB84TightProof(Lim2014Proof):
    """Implements a tighter finite-key security proof for BB84 decoy-state QKD."""
    def allocate_epsilons(self) -> EpsilonAllocation:
        """A more balanced allocation of epsilons."""
        eps_sec = self.p.eps_sec
        eps_cor = self.p.eps_cor
        
        remaining_eps = eps_sec - eps_cor
        if remaining_eps <= 0:
            raise ParameterValidationError(f"eps_sec ({eps_sec:.2e}) must be greater than eps_cor ({eps_cor:.2e}).")
        
        eps_pe = remaining_eps / 3
        eps_smooth = remaining_eps / 6
        eps_pa = remaining_eps / 3
        eps_phase_est = eps_pe / (4 * len(self.p.source.pulse_configs) + 1)
        
        return EpsilonAllocation(
            eps_sec=eps_sec, eps_cor=eps_cor, eps_pe=eps_pe,
            eps_smooth=eps_smooth, eps_pa=eps_pa, eps_phase_est=eps_phase_est
        )

    def calculate_key_length(self, decoy_estimates: Dict[str, Any], stats_map: Dict[str, TallyCounts]) -> int:
        """Calculates the secure key length using a tighter formula for privacy amplification."""
        Y1_L, e1_bit_U = decoy_estimates["Y1_L"], decoy_estimates["e1_U"]
        signal_stats = stats_map.get("signal")
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not signal_stats or signal_stats.sent == 0: return 0
        
        alice_z_prob = getattr(self.p.protocol, 'alice_z_basis_prob', 0.0)
        bob_z_prob = getattr(self.p.protocol, 'bob_z_basis_prob', 0.0)

        mu_s = p_sig_cfg.mean_photon_number
        p1_s = mu_s * math.exp(-mu_s)
        s_z_1_L = signal_stats.sent * p1_s * Y1_L * (alice_z_prob * bob_z_prob)
        n_z, m_z = signal_stats.sifted_z, signal_stats.errors_sifted_z
        
        if n_z <= 0 or s_z_1_L < S_Z_1_L_MIN_FOR_PHASE_EST: return 0
        
        qber_z = m_z / n_z
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z
        
        if self.p.assume_phase_equals_bit_error:
            e1_phase_U = e1_bit_U
        else:
            try:
                delta = math.sqrt(math.log(1.0 / self.eps_alloc.eps_phase_est) / (s_z_1_L))
            except (ValueError, ZeroDivisionError):
                 raise QKDSimulationError("Invalid value in phase error delta calculation.")
            e1_phase_U = min(0.5, e1_bit_U + delta)
            
        pa_term_bits = 2 * math.log2(1.0 / (2 * self.eps_alloc.eps_smooth)) + math.log2(1.0 / self.eps_alloc.eps_pa)
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)
        
        key_length_float = s_z_1_L * (1.0 - self.binary_entropy(e1_phase_U)) - leak_EC - pa_term_bits - corr_term_bits
        
        return max(0, math.floor(key_length_float))


class MDIQKDProof(Lim2014Proof):
    """
    Implements a finite-key security proof for MDI-QKD.
    WARNING: This is a placeholder and is NOT SCIENTIFICALLY VALID.
    """
    def calculate_key_length(self, decoy_estimates: Dict[str, Any], stats_map: Dict[str, TallyCounts]) -> int:
        """
        Calculates the secure key length for MDI-QKD. This is an unsafe placeholder.
        """
        if not self.p.allow_unsafe_mdi_approx:
            logger.error("MDI-QKD key rate calculation aborted. The current implementation lacks a "
                         "valid 2D decoy-state analysis and is not secure. To override, set "
                         "the 'allow_unsafe_mdi_approx' parameter to true in your config file.")
            raise QKDSimulationError("Unsafe MDI-QKD approximation not allowed.")
        
        logger.warning("MDIQKDProof is using a simplified placeholder key rate formula adapted from BB84. "
                       "The results are NOT secure or scientifically valid.")

        Y11_L, e11_x_U = decoy_estimates["Y1_L"], decoy_estimates["e1_U"]
        signal_stats_A = stats_map.get("signal")
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not signal_stats_A or signal_stats_A.sent == 0: return 0

        n_z, n_x = signal_stats_A.sifted_z, signal_stats_A.sifted_x
        m_x = signal_stats_A.errors_sifted_x
        if n_z <= 0 or n_x <= 0: return 0
        
        s_z_11_L = n_z * Y11_L
        m_x_U_bound, _ = self.get_bounds(m_x, n_x, self.eps_alloc.eps_pe)
        e_ph_11_U = m_x_U_bound

        qber_z = signal_stats_A.errors_sifted_z / n_z
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z

        pa_term_bits = 7 * math.log2(21 / self.eps_alloc.eps_smooth) + 2 * math.log2(1/self.eps_alloc.eps_pa)
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)

        key_length_float = s_z_11_L * (1.0 - self.binary_entropy(e_ph_11_U)) - leak_EC - pa_term_bits - corr_term_bits
        return max(0, math.floor(key_length_float))


# --- Main System Class ---

class QKDSystem:
    def __init__(self, params: QKDParams, seed: Optional[int] = None, save_master_seed: bool = False):
        self.p = params
        self.save_master_seed = save_master_seed
        if seed is None:
            self.master_seed_int = (secrets.randbits(63) % MAX_SEED_INT) or 1
        else:
            self.master_seed_int = (int(seed) % MAX_SEED_INT) or 1
        
        self.rng = np.random.default_rng(self.master_seed_int)
        
        if self.p.security_proof == SecurityProof.LIM_2014:
            self.proof_module = Lim2014Proof(self.p)
        elif self.p.security_proof == SecurityProof.TIGHT_PROOF:
            self.proof_module = BB84TightProof(self.p)
        elif self.p.security_proof == SecurityProof.MDI_QKD:
            self.proof_module = MDIQKDProof(self.p)
        else:
            raise NotImplementedError(f"Security proof {self.p.security_proof.value} not implemented.")

    def _merge_batch_tallies(self, overall_stats: Dict[str, TallyCounts], batch_result: Dict):
        for name, tally_dict in batch_result.get("overall", {}).items():
            stats_obj = overall_stats.setdefault(name, TallyCounts())
            for key, val in tally_dict.items():
                current_val = getattr(stats_obj, key)
                new_val = current_val + int(val)
                if new_val < current_val:
                    raise QKDSimulationError("Tally count overflow or corruption detected.")
                setattr(stats_obj, key, new_val)

    def run_simulation(self) -> SimulationResults:
        start_time = time.time()
        logger.debug(f"Starting simulation run with master_seed={self.master_seed_int}")
        total_pulses, batch_size = self.p.num_bits, self.p.batch_size
        num_batches = (total_pulses + batch_size - 1) // batch_size
        child_seeds = self.rng.integers(1, MAX_SEED_INT + 1, size=num_batches, dtype=np.int64).tolist()
        
        overall_stats = {pc.name: TallyCounts() for pc in self.p.source.pulse_configs}
        total_sifted, status = 0, "OK"
        
        params_dict = self.p.to_serializable_dict()
        tasks = [(params_dict, min(batch_size, total_pulses - i * batch_size), child_seeds[i]) for i in range(num_batches)]
        metadata = {"version": "v3.0-review-integrated"}
        if self.save_master_seed: metadata["master_seed"] = self.master_seed_int
            
        try:
            use_mp = self.p.num_workers > 1 and num_batches > 1 and not self.p.force_sequential
            if use_mp:
                with ProcessPoolExecutor(max_workers=self.p.num_workers) as executor:
                    futures = [executor.submit(_top_level_worker_function, *task) for task in tasks]
                    for fut in tqdm(as_completed(futures), total=len(tasks), desc="Simulating Batches (MP)"):
                        batch_result = fut.result()
                        self._merge_batch_tallies(overall_stats, batch_result)
                        total_sifted += batch_result["sifted_count"]
            else:
                for task in tqdm(tasks, desc="Simulating Batches (Seq)"):
                    batch_result = _top_level_worker_function(*task)
                    self._merge_batch_tallies(overall_stats, batch_result)
                    total_sifted += batch_result["sifted_count"]
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user. Aborting.")
            status = "USER_ABORT"
        except Exception as e:
            logger.exception("A worker process failed, aborting simulation.")
            status = f"WORKER_ERROR: {type(e).__name__}"
            metadata['error'] = {'type': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc(limit=5)}

        elapsed_time = time.time() - start_time
        if status != "OK":
            return SimulationResults(params=self.p, metadata=metadata, status=status, simulation_time_seconds=elapsed_time)

        decoy_est, secure_len, cert = None, None, None
        try:
            logger.debug("Starting post-processing: decoy state estimation.")
            decoy_est = self.proof_module.estimate_yields_and_errors(overall_stats)
            if not decoy_est.get("ok"):
                status = f"DECOY_ESTIMATION_FAILED: {decoy_est.get('status', 'Unknown')}"
            else:
                logger.debug("Decoy estimation successful. Calculating secure key length.")
                secure_len = self.proof_module.calculate_key_length(decoy_est, overall_stats)
                cert = SecurityCertificate(
                    proof_name=self.p.security_proof.value,
                    confidence_bound_method=self.p.ci_method.value,
                    assumed_phase_equals_bit_error=self.p.assume_phase_equals_bit_error,
                    epsilon_allocation=self.proof_module.eps_alloc,
                    lp_solver_diagnostics=decoy_est.get("lp_diagnostics")
                )
        except (LPFailureError, ParameterValidationError, QKDSimulationError) as e:
            status = f"POST_PROCESSING_FAILED: {type(e).__name__} - {e}"
            logger.error(status, exc_info=True)

        return SimulationResults(
            params=self.p, metadata=metadata, security_certificate=cert,
            decoy_estimates=decoy_est, secure_key_length=secure_len,
            raw_sifted_key_length=total_sifted, simulation_time_seconds=elapsed_time,
            status=status
        )


# --- Command-Line Interface ---

def create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production-Grade Finite-Key QKD Simulation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("param_file", help="Path to the JSON file with simulation parameters.")
    parser.add_argument("-o", "--output", help="Path to save the results JSON file. Prints to stdout if not provided.")
    parser.add_argument("--seed", type=int, default=None, help="Master seed for the simulation for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes. Overrides value in param file.")
    parser.add_argument("--force-sequential", action="store_true", help="Force sequential execution, even if num_workers > 1.")
    parser.add_argument("--save-seed", action="store_true", help="Save the master seed in the output metadata. WARNING: For reproducibility only.")
    parser.add_argument("--lp-solver-method", type=str, default=None, choices=LP_SOLVER_METHODS, help="Primary LP solver. Overrides value in param file.")
    parser.add_argument('--verbosity', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Set the logging verbosity level.')
    parser.add_argument('--dry-run', action='store_true', help='Load and validate params, print derived values, and exit without running simulation.')
    parser.add_argument('--allow-unsafe-mdi-approx', action='store_true', help='Allow using the unsafe/placeholder MDI-QKD proof. FOR EXPERIMENTAL USE ONLY.')
    return parser


def main():
    """Main entry point for the simulation."""
    # For cross-platform reproducibility, set the start method to 'spawn'.
    if sys.platform != "win32":
        multiprocessing.set_start_method('spawn', force=True)

    parser = create_cli_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(args.verbosity)

    try:
        with open(args.param_file, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
        
        # Override params from CLI args
        if args.num_workers is not None: params_dict['num_workers'] = args.num_workers
        if args.force_sequential: params_dict['force_sequential'] = True
        if args.lp_solver_method: params_dict['lp_solver_method'] = args.lp_solver_method
        if args.allow_unsafe_mdi_approx: params_dict['allow_unsafe_mdi_approx'] = True

        params = QKDParams.from_dict(params_dict)

        if args.dry_run:
            print("--- Dry Run: Parameter Validation and Derived Values ---")
            print(f"Parameter file '{args.param_file}' loaded and validated successfully.")
            print("\nDerived Parameters:")
            print(f"  - Channel Transmittance: {params.channel.transmittance:.4e}")
            print(f"  - Overall Detector Efficiency (incl. channel): {params.channel.transmittance * params.detector.det_eff:.4e}")
            print("\nConfiguration appears valid. Exiting dry run.")
            sys.exit(0)

        logger.info(f"Starting QKD simulation with parameters from: {args.param_file}")
        if args.seed: logger.info(f"Using provided master seed: {args.seed}")
        if args.save_seed: logger.warning("Master seed will be saved in the output file for reproducibility.")

        system = QKDSystem(params, seed=args.seed, save_master_seed=args.save_seed)
        results = system.run_simulation()

        logger.info(f"Simulation finished in {results.simulation_time_seconds:.2f} seconds.")
        logger.info(f"Status: {results.status}")
        if results.status == "OK":
            logger.info(f"Raw Sifted Key Length: {results.raw_sifted_key_length}")
            logger.info(f"Final Secure Key Length: {results.secure_key_length}")

        if args.output:
            results.save_json(args.output)
        else:
            print("\n--- Simulation Results ---")
            print(json.dumps(results.to_serializable_dict(), indent=2))
            print("------------------------\n")

    except FileNotFoundError:
        logger.critical(f"Parameter file not found: {args.param_file}")
        sys.exit(1)
    except (json.JSONDecodeError, ParameterValidationError, NotImplementedError) as e:
        logger.critical(f"Error loading or validating parameters from {args.param_file}: {e}", exc_info=False)
        sys.exit(2)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
