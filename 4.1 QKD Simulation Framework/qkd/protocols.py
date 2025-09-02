# -*- coding: utf-8 -*-
"""
Implements Quantum Key Distribution (QKD) protocols with a focus on correctness,
robustness, and clear separation of concerns.

This module provides an abstract base class `Protocol` and concrete implementations
for BB84 with decoy states and Measurement-Device-Independent (MDI) QKD.

Key Design Principles:
- **Causality & Reproducibility:** All stochastic choices are made upfront using
  a provided, seeded `np.random.Generator`. The protocols are stateless. For
  multiprocessing, it is critical to use a robust seeding strategy, such as
  spawning child RNGs from a `np.random.SeedSequence` using the included
  `make_worker_rngs` helper.
- **Separation of Concerns:** Protocol logic (sifting) is strictly separated
  from the physical simulation of detectors. This module expects detection
  results to be pre-computed with all physical effects already modeled.
- **Strict, Immutable Contracts:** The API uses frozen dataclasses with rigorous
  __post_init__ validation. All NumPy arrays within these dataclasses are made
  immutable to prevent accidental mutation and ensure run-to-run consistency.
- **Portability:** Relies on NumPy features compatible with version >= 1.22 and
  SciPy for statistical functions. RNG calls and type hints are structured for
  maximum compatibility.

Version: 5.0.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Union, Any

import numpy as np
import scipy.stats as stats
from numpy.random import Generator, PCG64, SeedSequence
from numpy.typing import NDArray

# Import the source from the corrected qkd.sources module
from qkd.sources import PoissonSource

__all__ = [
    "BB84DecoyProtocol", "MDIQKDProtocol", "Protocol",
    "BB84PreparedStates", "MDIPreparedStates", "DetectionResults", "SiftingResults",
    "DoubleClickPolicy", "ParameterValidationError", "ShapeError", "DomainError",
    "PoissonSource", "PulseConfig", "make_worker_rngs", "sample_for_parameter_estimation"
]

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# --- Type Aliases for Clarity ---
BitArray = NDArray[np.uint8]  # Use unsigned int for {0, 1}
IndexArray = NDArray[np.int64] # Use fixed-size int for portability
MaskArray = NDArray[np.bool_]
RNG = Generator # Alias for type hinting

# --- Constants ---
Z_BASIS: int = 0
X_BASIS: int = 1
DETECTOR_0_KEY: str = 'click0'
DETECTOR_1_KEY: str = 'click1'

# --- Custom Exceptions ---
class ParameterValidationError(ValueError):
    """Base custom exception for invalid simulation parameters."""
    pass

class ShapeError(ParameterValidationError):
    """Exception for array shape or dimension mismatches."""
    pass

class DomainError(ParameterValidationError):
    """Exception for values outside their expected domain."""
    pass

# --- Data Structures for Configuration ---

@dataclass(frozen=True)
class PulseConfig:
    """Configuration for a single type of pulse (e.g., signal or decoy)."""
    probability: float
    mean_photon_number: float
    name: str

    def __post_init__(self):
        if not self.name:
            raise ParameterValidationError("PulseConfig 'name' cannot be empty.")
        if not (0.0 <= self.probability <= 1.0):
            raise DomainError(f"Probability must be in [0,1], got {self.probability}")
        if not np.isfinite(self.mean_photon_number) or self.mean_photon_number < 0:
            raise DomainError(f"Mean photon number must be non-negative, got {self.mean_photon_number}")

# --- Helper Functions ---

def _validate_and_freeze_array(arr: np.ndarray, name: str, expected_dtype: type, expected_ndim: int = 1) -> np.ndarray:
    """Validates array properties, coerces type, and makes it immutable."""
    dtype_obj = np.dtype(expected_dtype)
    arr = np.ascontiguousarray(arr) # Ensure contiguous memory layout

    if not np.issubdtype(arr.dtype, dtype_obj):
        arr = arr.astype(dtype_obj)

    if arr.ndim != expected_ndim:
        raise ShapeError(f"Array '{name}' expected {expected_ndim}-D, but got {arr.ndim}-D.")
    
    arr.flags.writeable = False
    return arr

def make_worker_rngs(master_seed: Union[int, SeedSequence], num_workers: int) -> List[RNG]:
    """
    Creates a list of independent, high-quality RNGs for parallel processing.
    This is the recommended way to ensure reproducible parallel simulations.
    """
    if isinstance(master_seed, Generator):
        raise TypeError("Cannot create worker RNGs from a Generator instance. Pass an int or SeedSequence.")
    seed_seq = SeedSequence(master_seed) if isinstance(master_seed, int) else master_seed
    child_seeds = seed_seq.spawn(num_workers)
    return [Generator(PCG64(s)) for s in child_seeds]

def sample_for_parameter_estimation(sifting_results: "SiftingResults", rng: RNG, sample_fraction: float) -> MaskArray:
    """
    Randomly samples a fraction of sifted bits for parameter estimation.
    """
    if not (0.0 <= sample_fraction <= 1.0):
        raise DomainError(f"sample_fraction must be in [0,1], got {sample_fraction}")
    
    sifted_indices = np.flatnonzero(sifting_results.sifted_mask)
    num_sifted = len(sifted_indices)
    num_samples = int(round(num_sifted * sample_fraction))
    
    sampled_indices = rng.choice(sifted_indices, size=num_samples, replace=False)
    
    param_estimation_mask = np.zeros(sifting_results.num_pulses, dtype=bool)
    param_estimation_mask[sampled_indices] = True
    return param_estimation_mask

# --- Data Structures for API Contracts ---

@dataclass(frozen=True)
class BB84PreparedStates:
    """Immutable dataclass for states prepared in a BB84 protocol run."""
    num_pulses: int
    alice_bits: BitArray
    alice_bases: BitArray
    alice_pulse_type_indices: IndexArray
    bob_bases: BitArray

    def __post_init__(self):
        """Runtime validation and freezing of arrays."""
        arrays = {
            'alice_bits': (self.alice_bits, np.uint8), 'alice_bases': (self.alice_bases, np.uint8),
            'alice_pulse_type_indices': (self.alice_pulse_type_indices, np.int64),
            'bob_bases': (self.bob_bases, np.uint8)
        }
        for name, (arr, dtype) in arrays.items():
            validated_arr = _validate_and_freeze_array(arr, name, dtype)
            object.__setattr__(self, name, validated_arr)
            if validated_arr.shape[0] != self.num_pulses:
                raise ShapeError(f"Array '{name}' length mismatch: expected {self.num_pulses}, got {validated_arr.shape[0]}.")
        
        for arr in [self.alice_bits, self.alice_bases, self.bob_bases]:
            if not np.all((arr == 0) | (arr == 1)):
                raise DomainError("Bit and basis arrays must contain only 0s and 1s.")

@dataclass(frozen=True)
class MDIPreparedStates:
    """Immutable dataclass for states prepared in an MDI-QKD protocol run."""
    num_pulses: int
    alice_bits: BitArray
    alice_bases: BitArray
    alice_pulse_type_indices: IndexArray
    bob_bits: BitArray
    bob_bases: BitArray
    bob_pulse_type_indices: IndexArray
    
    def __post_init__(self):
        """Runtime validation and freezing of arrays."""
        arrays = {
            'alice_bits': (self.alice_bits, np.uint8), 'alice_bases': (self.alice_bases, np.uint8),
            'alice_pulse_type_indices': (self.alice_pulse_type_indices, np.int64),
            'bob_bits': (self.bob_bits, np.uint8), 'bob_bases': (self.bob_bases, np.uint8),
            'bob_pulse_type_indices': (self.bob_pulse_type_indices, np.int64)
        }
        for name, (arr, dtype) in arrays.items():
            validated_arr = _validate_and_freeze_array(arr, name, dtype)
            object.__setattr__(self, name, validated_arr)
            if validated_arr.shape[0] != self.num_pulses:
                raise ShapeError(f"Array '{name}' length mismatch: expected {self.num_pulses}, got {validated_arr.shape[0]}.")
        
        for arr in [self.alice_bits, self.alice_bases, self.bob_bits, self.bob_bases]:
            if not np.all((arr == 0) | (arr == 1)):
                raise DomainError("Bit and basis arrays must contain only 0s and 1s.")

@dataclass(frozen=True)
class DetectionResults:
    """Immutable dataclass for detection outcomes."""
    num_pulses: int
    click0: MaskArray
    click1: MaskArray
    metadata: Optional[Dict] = None

    def __post_init__(self):
        object.__setattr__(self, 'click0', _validate_and_freeze_array(self.click0, 'click0', np.bool_))
        object.__setattr__(self, 'click1', _validate_and_freeze_array(self.click1, 'click1', np.bool_))
        if self.click0.shape[0] != self.num_pulses or self.click1.shape[0] != self.num_pulses:
            raise ShapeError("Detection array lengths must match num_pulses.")

@dataclass(frozen=True)
class SiftingResults:
    """Structured, immutable return type for sifting results."""
    num_pulses: int
    sifted_mask: MaskArray
    error_mask: MaskArray
    sifted_alice_pulse_indices: IndexArray
    sifted_bob_pulse_indices: Optional[IndexArray] = None # For MDI

    def __post_init__(self):
        """Validates consistency of the result masks."""
        for arr_name, arr in [('sifted_mask', self.sifted_mask), ('error_mask', self.error_mask)]:
            if arr.ndim != 1 or arr.shape[0] != self.num_pulses:
                raise ShapeError(f"Mask '{arr_name}' has invalid shape or length.")
        if np.any(self.error_mask & ~self.sifted_mask):
            raise ValueError("Internal logic error: error_mask contains True values outside of sifted_mask.")

    def summary(self, confidence_level: float = 0.95) -> Dict[str, Union[int, float]]:
        """Returns a dictionary with aggregated counts and QBER with confidence intervals."""
        num_sifted = np.count_nonzero(self.sifted_mask)
        num_errors = np.count_nonzero(self.error_mask)
        qber = (num_errors / num_sifted) if num_sifted > 0 else 0.0
        
        alpha = 1.0 - confidence_level
        if num_errors == 0:
            qber_ci_low = 0.0
        else:
            qber_ci_low = stats.beta.ppf(alpha / 2, num_errors, num_sifted - num_errors + 1)
        
        if num_errors == num_sifted:
            qber_ci_high = 1.0
        else:
            qber_ci_high = stats.beta.ppf(1 - alpha / 2, num_errors + 1, num_sifted - num_errors)

        return {
            "num_sifted": int(num_sifted),
            "num_errors": int(num_errors),
            "qber": float(qber),
            "qber_confidence_level": confidence_level,
            "qber_ci_low": float(np.nan_to_num(qber_ci_low)),
            "qber_ci_high": float(np.nan_to_num(qber_ci_high, nan=1.0)),
        }

# --- Abstract Protocol Base Class ---

class Protocol(ABC):
    """Abstract Base Class for a QKD Protocol."""
    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """A unique string identifier for the protocol."""
        pass

    @abstractmethod
    def prepare_states(self, num_pulses: int, rng: RNG) -> Union[BB84PreparedStates, MDIPreparedStates]:
        pass

    @abstractmethod
    def sift_results(self, prepared_states: Union[BB84PreparedStates, MDIPreparedStates],
                       detection_results: DetectionResults, rng: RNG) -> SiftingResults:
        pass

    @abstractmethod
    def to_config_dict(self) -> Dict[str, Any]:
        """Serializes the protocol's configuration to a dictionary."""
        pass

# --- Concrete Protocol Implementations ---

class DoubleClickPolicy(Enum):
    """Defines how to handle double-click events in detectors."""
    DISCARD = 0
    RANDOM = 1

class BB84DecoyProtocol(Protocol):
    """Concrete implementation of the BB84 protocol with decoy states."""
    def __init__(self, alice_z_basis_prob: float, bob_z_basis_prob: float,
                 source: PoissonSource, double_click_policy: Union[DoubleClickPolicy, str] = DoubleClickPolicy.DISCARD):
        if not (0.0 <= alice_z_basis_prob <= 1.0):
            raise DomainError(f"alice_z_basis_prob must be in [0,1], got {alice_z_basis_prob}.")
        if not (0.0 <= bob_z_basis_prob <= 1.0):
            raise DomainError(f"bob_z_basis_prob must be in [0,1], got {bob_z_basis_prob}.")
        if not isinstance(source, PoissonSource):
            raise TypeError("source must be a PoissonSource instance.")

        # Allow callers to pass either the enum or a string (case-insensitive)
        if isinstance(double_click_policy, str):
            try:
                policy_enum = DoubleClickPolicy[double_click_policy.strip().upper()]
            except KeyError:
                raise DomainError(f"Unknown double_click_policy '{double_click_policy}'. Valid values: {[p.name for p in DoubleClickPolicy]}")
            double_click_policy = policy_enum

        if not isinstance(double_click_policy, DoubleClickPolicy):
            raise TypeError("double_click_policy must be a DoubleClickPolicy instance or a valid policy name string.")

        self.alice_z_basis_prob = alice_z_basis_prob
        self.bob_z_basis_prob = bob_z_basis_prob
        self.source = source
        self.double_click_policy = double_click_policy

    @property
    def protocol_name(self) -> str:
        return "bb84-decoy"

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "alice_z_basis_prob": self.alice_z_basis_prob,
            "bob_z_basis_prob": self.bob_z_basis_prob,
            "double_click_policy": self.double_click_policy.name,
        }

    def prepare_states(self, num_pulses: int, rng: RNG) -> BB84PreparedStates:
        if num_pulses == 0:
            return BB84PreparedStates(num_pulses=0,
                alice_bits=np.array([], dtype=np.uint8), alice_bases=np.array([], dtype=np.uint8),
                alice_pulse_type_indices=np.array([], dtype=np.int64), bob_bases=np.array([], dtype=np.uint8))

        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        alice_bases = (rng.random(size=num_pulses) < self.alice_z_basis_prob).astype(np.uint8)
        bob_bases = (rng.random(size=num_pulses) < self.bob_z_basis_prob).astype(np.uint8)

        probs = np.array([pc.probability for pc in self.source.pulse_configs], dtype=float)
        probs = np.clip(probs, 0.0, None)
        prob_sum = probs.sum()
        if prob_sum <= 0: raise DomainError("Sum of probabilities must be positive for sampling.")
        probs /= prob_sum
        alice_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        return BB84PreparedStates(
            num_pulses=num_pulses, alice_bits=alice_bits, alice_bases=alice_bases,
            alice_pulse_type_indices=alice_pulse_indices, bob_bases=bob_bases
        )

    def sift_results(self, prepared_states: Union[BB84PreparedStates, MDIPreparedStates],
                       detection_results: DetectionResults, rng: RNG) -> SiftingResults:
        if not isinstance(prepared_states, BB84PreparedStates):
            raise TypeError(f"Expected BB84PreparedStates, but got {type(prepared_states).__name__}")

        basis_match = (prepared_states.alice_bases == prepared_states.bob_bases)
        click0, click1 = detection_results.click0, detection_results.click1
        
        conclusive0 = click0 & ~click1
        conclusive1 = click1 & ~click0
        double_click_mask = click0 & click1

        bob_bits_valid = conclusive0 | conclusive1
        bob_bits = np.zeros(prepared_states.num_pulses, dtype=np.uint8)
        bob_bits[conclusive1] = 1

        if self.double_click_policy == DoubleClickPolicy.RANDOM:
            num_dc = np.count_nonzero(double_click_mask)
            if num_dc > 0:
                bob_bits[double_click_mask] = rng.integers(0, 2, size=num_dc, dtype=np.uint8)
                bob_bits_valid[double_click_mask] = True
        elif self.double_click_policy == DoubleClickPolicy.DISCARD:
            bob_bits_valid[double_click_mask] = False

        sifted_mask = basis_match & bob_bits_valid
        error_mask = np.zeros(prepared_states.num_pulses, dtype=bool)
        
        sifted_indices = sifted_mask.nonzero()[0]
        if sifted_indices.size > 0:
            errors_in_sifted = (prepared_states.alice_bits[sifted_indices] != bob_bits[sifted_indices])
            error_mask[sifted_indices] = errors_in_sifted

        return SiftingResults(
            num_pulses=prepared_states.num_pulses, sifted_mask=sifted_mask, error_mask=error_mask,
            sifted_alice_pulse_indices=prepared_states.alice_pulse_type_indices[sifted_mask]
        )

class MDIQKDProtocol(Protocol):
    """
    Concrete implementation of MDI-QKD.
    WARNING: This is a TOY MODEL for the BSM. It assumes any coincident click
    is a successful Psi- projection. A realistic model requires a sophisticated
    detector simulation that accounts for interference visibility, accidental
    coincidences, and time windows.
    """
    def __init__(self, z_basis_prob: float, source: PoissonSource):
        if not (0.0 <= z_basis_prob <= 1.0):
            raise DomainError(f"z_basis_prob must be in [0,1], got {z_basis_prob}.")
        self.z_basis_prob = z_basis_prob
        self.source = source

    @property
    def protocol_name(self) -> str:
        return "mdi-qkd"

    def to_config_dict(self) -> Dict[str, Any]:
        return {"z_basis_prob": self.z_basis_prob}

    def prepare_states(self, num_pulses: int, rng: RNG) -> MDIPreparedStates:
        if num_pulses == 0:
            return MDIPreparedStates(num_pulses=0,
                alice_bits=np.array([], dtype=np.uint8), alice_bases=np.array([], dtype=np.uint8),
                alice_pulse_type_indices=np.array([], dtype=np.int64),
                bob_bits=np.array([], dtype=np.uint8), bob_bases=np.array([], dtype=np.uint8),
                bob_pulse_type_indices=np.array([], dtype=np.int64))

        probs = np.array([pc.probability for pc in self.source.pulse_configs], dtype=float)
        probs = np.clip(probs, 0.0, None)
        prob_sum = probs.sum()
        if prob_sum <= 0: raise DomainError("Sum of probabilities must be positive for sampling.")
        probs /= prob_sum

        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        alice_bases = (rng.random(size=num_pulses) < self.z_basis_prob).astype(np.uint8)
        alice_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)

        bob_bits = rng.integers(0, 2, size=num_pulses, dtype=np.uint8)
        bob_bases = (rng.random(size=num_pulses) < self.z_basis_prob).astype(np.uint8)
        bob_pulse_indices = rng.choice(len(probs), size=num_pulses, p=probs).astype(np.int64)
        
        return MDIPreparedStates(
            num_pulses=num_pulses, alice_bits=alice_bits, alice_bases=alice_bases,
            alice_pulse_type_indices=alice_pulse_indices, bob_bits=bob_bits,
            bob_bases=bob_bases, bob_pulse_type_indices=bob_pulse_indices
        )

    def sift_results(self, prepared_states: Union[BB84PreparedStates, MDIPreparedStates],
                       detection_results: DetectionResults, rng: RNG) -> SiftingResults:
        if not isinstance(prepared_states, MDIPreparedStates):
            raise TypeError(f"Expected MDIPreparedStates, but got {type(prepared_states).__name__}")

        successful_bsm = detection_results.click0 & detection_results.click1
        basis_match = (prepared_states.alice_bases == prepared_states.bob_bases)
        sifted_mask = basis_match & successful_bsm

        error_mask = np.zeros(prepared_states.num_pulses, dtype=bool)
        sifted_indices = sifted_mask.nonzero()[0]

        if sifted_indices.size > 0:
            sifted_alice_bits = prepared_states.alice_bits[sifted_indices]
            sifted_bob_bits = prepared_states.bob_bits[sifted_indices]
            sifted_bases = prepared_states.alice_bases[sifted_indices]

            z_basis_errors = (sifted_bases == Z_BASIS) & (sifted_alice_bits == sifted_bob_bits)
            x_basis_errors = (sifted_bases == X_BASIS) & (sifted_alice_bits != sifted_bob_bits)
            errors_in_sifted = z_basis_errors | x_basis_errors
            error_mask[sifted_indices] = errors_in_sifted

        return SiftingResults(
            num_pulses=prepared_states.num_pulses, sifted_mask=sifted_mask, error_mask=error_mask,
            sifted_alice_pulse_indices=prepared_states.alice_pulse_type_indices[sifted_mask],
            sifted_bob_pulse_indices=prepared_states.bob_pulse_type_indices[sifted_mask]
        )

# --- Example Usage ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # --- Setup a simulation ---
    master_seed = 42
    num_pulses_to_sim = 1_000_000
    num_workers = 4 # Example for parallel simulation
    
    # Recommended RNG setup for parallel runs
    worker_rngs = make_worker_rngs(master_seed, num_workers)
    pulses_per_worker = num_pulses_to_sim // num_workers
    
    print(f"Master seed: {master_seed}. Spawning {num_workers} worker RNGs for simulation.")

    # Use the imported PoissonSource
    from qkd.datatypes import PulseTypeConfig as QKDPulseTypeConfig
    source_config = PoissonSource(pulse_configs=[
        QKDPulseTypeConfig(probability=0.8, mean_photon_number=0.5, name='signal'),
        QKDPulseTypeConfig(probability=0.1, mean_photon_number=0.1, name='decoy'),
        QKDPulseTypeConfig(probability=0.1, mean_photon_number=0.0, name='vacuum'),
    ])

    # --- BB84 Example ---
    print("\n--- Running BB84 Simulation ---")
    bb84_protocol = BB84DecoyProtocol(
        alice_z_basis_prob=0.5, bob_z_basis_prob=0.5,
        source=source_config, double_click_policy=DoubleClickPolicy.DISCARD
    )
    
    all_sifted_results = []
    # In a real parallel setup, this loop would be a process pool
    for i in range(num_workers):
        worker_rng = worker_rngs[i]
        print(f"Simulating batch {i+1}/{num_workers}...")
        
        # 1. Prepare states for this batch
        bb84_states = bb84_protocol.prepare_states(pulses_per_worker, worker_rng)

        # 2. Simulate a simple channel and detection (toy model)
        transmission_prob = 0.1
        dark_count_prob = 1e-6
        photons_at_bob = worker_rng.random(pulses_per_worker) < transmission_prob
        basis_match = bb84_states.alice_bases == bb84_states.bob_bases
        # This is a highly simplified detection model for demonstration only
        ideal_click0 = photons_at_bob & ( (basis_match & (bb84_states.alice_bits == 0)) | (~basis_match & (worker_rng.random(pulses_per_worker) < 0.5)) )
        ideal_click1 = photons_at_bob & ( (basis_match & (bb84_states.alice_bits == 1)) | (~basis_match & (worker_rng.random(pulses_per_worker) < 0.5)) )
        final_click0 = ideal_click0 | (worker_rng.random(pulses_per_worker) < dark_count_prob)
        final_click1 = ideal_click1 | (worker_rng.random(pulses_per_worker) < dark_count_prob)
        bb84_detections = DetectionResults(num_pulses=pulses_per_worker, click0=final_click0, click1=final_click1)

        # 3. Sift results for this batch
        sifted_batch = bb84_protocol.sift_results(bb84_states, bb84_detections, worker_rng)
        all_sifted_results.append(sifted_batch)

    # 4. Aggregate results from all workers
    total_sifted = sum(res.summary()['num_sifted'] for res in all_sifted_results)
    total_errors = sum(res.summary()['num_errors'] for res in all_sifted_results)
    aggregated_qber = (total_errors / total_sifted) if total_sifted > 0 else 0.0
    
    print("\n--- Aggregated BB84 Sifting Summary ---")
    print(f"  Total pulses simulated: {num_pulses_to_sim}")
    print(f"  Total sifted bits: {total_sifted}")
    print(f"  Total errors: {total_errors}")
    print(f"  Aggregated QBER: {aggregated_qber:.6f}")

