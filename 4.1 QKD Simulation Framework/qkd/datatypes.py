# -*- coding: utf-8 -*-
"""
Data structures and enumerations for the QKD simulation framework.

This module defines robust, validated, and serializable data structures for
configuring, running, and analyzing QKD simulations. It incorporates best
practices such as type hinting, validation, and immutability for security-critical
parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np

from .exceptions import ParameterValidationError
from .constants import NUMERIC_ABS_TOL

if TYPE_CHECKING:
    from .params import QKDParams

# The __all__ list is alphabetically sorted and defines the public API of this module.
__all__ = [
    "ConfidenceBoundMethod",
    "DoubleClickPolicy",
    "EpsilonAllocation",
    "PulseTypeConfig",
    "SecurityCertificate",
    "SecurityProof",
    "SimulationResults",
    "TallyCounts",
]


# --- Enums ---
# Enums provide type-safe and explicit choices for simulation parameters.

class DoubleClickPolicy(Enum):
    """Policy for handling double-click events in detectors."""
    DISCARD = "discard"
    RANDOM = "random"


class SecurityProof(Enum):
    """Supported security proofs for key rate calculation."""
    LIM_2014 = "lim-2014"
    TIGHT_PROOF = "tight-proof"
    MDI_QKD = "mdi-qkd"


class ConfidenceBoundMethod(Enum):
    """Statistical methods for calculating confidence bounds."""
    CLOPPER_PEARSON = "clopper-pearson"
    HOEFFDING = "hoeffding"


# --- Dataclasses ---
# Dataclasses are used for structured data with automatic validation and serialization.

@dataclass(frozen=True, slots=True)
class PulseTypeConfig:
    """
    Configuration for a single pulse type in a decoy-state protocol.

    Attributes:
        name: A unique identifier for the pulse type (e.g., "signal", "decoy").
        mean_photon_number: The unitless mean photon number (μ).
        probability: The probability of sending this pulse type.
    """
    name: str
    mean_photon_number: float
    probability: float

    def __post_init__(self):
        """Validate fields after initialization."""
        if not self.name:
            raise ParameterValidationError("PulseTypeConfig.name cannot be empty.")
        if not (np.isfinite(self.mean_photon_number) and self.mean_photon_number >= 0.0):
            raise ParameterValidationError(f"mean_photon_number for '{self.name}' must be a finite, non-negative float.")
        if not (np.isfinite(self.probability) and 0.0 <= self.probability <= 1.0):
            raise ParameterValidationError(f"probability for '{self.name}' must be a finite float between 0 and 1.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PulseTypeConfig:
        """Deserializes a dictionary into a PulseTypeConfig object."""
        return cls(**data)


@dataclass(slots=True)
class TallyCounts:
    """
    A mutable container for event counts during a QKD simulation.

    This class tracks various counts like sent pulses, sifted bits, and errors,
    both in total and per basis (Z and X). It includes validation to ensure
    counts are consistent and non-negative.

    Invariants (checked in __post_init__):
        - All counts must be non-negative integers.
        - Sifted counts cannot exceed sent counts.
        - Error counts cannot exceed sifted counts.
    """
    sent: int = 0
    sifted: int = 0
    errors_sifted: int = 0
    double_clicks_discarded: int = 0
    sent_z: int = 0
    sent_x: int = 0
    sifted_z: int = 0
    sifted_x: int = 0
    errors_sifted_z: int = 0
    errors_sifted_x: int = 0

    def __post_init__(self):
        """Validate the consistency and integrity of the counts."""
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value < 0:
                raise ParameterValidationError(f"TallyCounts field '{name}' must be a non-negative integer, but got {value}.")

        # Check logical invariants
        if self.sifted > self.sent:
            raise ParameterValidationError(f"Sifted count ({self.sifted}) cannot exceed sent count ({self.sent}).")
        if self.errors_sifted > self.sifted:
            raise ParameterValidationError(f"Error count ({self.errors_sifted}) cannot exceed sifted count ({self.sifted}).")
        if self.sifted_z > self.sent_z or self.sifted_x > self.sent_x:
            raise ParameterValidationError("Per-basis sifted counts cannot exceed per-basis sent counts.")
        if self.errors_sifted_z > self.sifted_z or self.errors_sifted_x > self.sifted_x:
            raise ParameterValidationError("Per-basis error counts cannot exceed per-basis sifted counts.")

    def merged(self, other: TallyCounts) -> TallyCounts:
        """
        Returns a new TallyCounts instance representing the sum of this and another.
        """
        if not isinstance(other, TallyCounts):
            raise TypeError("Can only merge with another TallyCounts instance.")
        
        # Using asdict and a dictionary comprehension is a clean way to sum fields.
        self_dict = asdict(self)
        other_dict = asdict(other)
        merged_data = {key: self_dict[key] + other_dict[key] for key in self_dict}
        return TallyCounts(**merged_data)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TallyCounts:
        """Deserializes a dictionary into a TallyCounts object."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EpsilonAllocation:
    """
    Defines the allocation of the total security parameter (epsilon) among
    different components of the security proof.

    This allocation is critical for the overall security claim. The `validate`
    method enforces the security composition theorem.
    """
    eps_sec: float  # Total security parameter for the final key
    eps_cor: float  # Correctness parameter
    eps_pe: float   # Parameter estimation security
    eps_smooth: float # Smoothing parameter for the smooth min-entropy
    eps_pa: float   # Privacy amplification security
    eps_phase_est: float # Phase error estimation security

    def __post_init__(self):
        """Automatically validate after initialization."""
        self.validate()

    def validate(self):
        """
        Ensures all epsilon values are valid and satisfy the security composition theorem.

        The key composition theorem (e.g., from Lim et al., 2014) requires that
        eps_sec >= eps_pe + 2*eps_smooth + eps_pa.
        """
        all_eps = asdict(self)
        for name, val in all_eps.items():
            if not np.isfinite(val) or val < 0.0:
                raise ParameterValidationError(f"Epsilon '{name}' must be a finite, non-negative float, but got {val}.")

        # Check the security composition theorem
        total_sum = self.eps_pe + 2.0 * self.eps_smooth + self.eps_pa
        if total_sum - self.eps_sec > NUMERIC_ABS_TOL:
            raise ParameterValidationError(
                f"Epsilon allocation insecure: sum of components ({total_sum:.3e}) "
                f"exceeds eps_sec ({self.eps_sec:.3e})."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpsilonAllocation:
        """Deserializes a dictionary into an EpsilonAllocation object."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SecurityCertificate:
    """
    An immutable record of the parameters and assumptions used to generate a secure key length.
    """
    proof_name: SecurityProof
    confidence_bound_method: ConfidenceBoundMethod
    assumed_phase_equals_bit_error: bool
    epsilon_allocation: EpsilonAllocation
    lp_solver_diagnostics: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        """Validate field types after initialization."""
        if not isinstance(self.proof_name, SecurityProof):
            raise ParameterValidationError(f"proof_name must be a SecurityProof enum, but got {type(self.proof_name)}.")
        if not isinstance(self.confidence_bound_method, ConfidenceBoundMethod):
            raise ParameterValidationError(
                "confidence_bound_method must be a ConfidenceBoundMethod enum, "
                f"but got {type(self.confidence_bound_method)}."
            )
        # The EpsilonAllocation's own __post_init__ handles its validation.

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the object to a dictionary, converting enums to their string values.
        """
        return {
            "proof_name": self.proof_name.value,
            "confidence_bound_method": self.confidence_bound_method.value,
            "assumed_phase_equals_bit_error": self.assumed_phase_equals_bit_error,
            "epsilon_allocation": self.epsilon_allocation.to_dict(),
            "lp_solver_diagnostics": self.lp_solver_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SecurityCertificate:
        """
        Deserializes a dictionary into a SecurityCertificate object,
        converting string values back to enums.
        """
        return cls(
            proof_name=SecurityProof(data["proof_name"]),
            confidence_bound_method=ConfidenceBoundMethod(data["confidence_bound_method"]),
            assumed_phase_equals_bit_error=data["assumed_phase_equals_bit_error"],
            epsilon_allocation=EpsilonAllocation.from_dict(data["epsilon_allocation"]),
            lp_solver_diagnostics=data.get("lp_solver_diagnostics"),
        )


@dataclass
class SimulationResults:
    """
    A container for the complete results of a QKD simulation run.

    This class is mutable during the simulation process and holds all final
    data, including parameters, counts, security certificates, and metadata.
    A JSON schema could be defined for this object to ensure robust data exchange.
    """
    params: "QKDParams"
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_certificate: Optional[SecurityCertificate] = None
    decoy_estimates: Optional[Dict[str, Any]] = None
    secure_key_length: Optional[int] = None
    raw_sifted_key_length: int = 0
    simulation_time_seconds: float = 0.0
    status: str = "OK"

    def __post_init__(self):
        """Validate fields after initialization."""
        if not isinstance(self.metadata, dict):
            raise ParameterValidationError(f"metadata must be a dict, but got {type(self.metadata)}.")
        if self.secure_key_length is not None and self.secure_key_length < 0:
            raise ParameterValidationError("secure_key_length cannot be negative.")
        if self.raw_sifted_key_length < 0:
            raise ParameterValidationError("raw_sifted_key_length cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the results to a dictionary.

        Note: The 'params' field is complex and its serialization is assumed
        to be handled by the QKDParams class itself. Here we just call its
        to_dict() method if it exists.
        """
        params_dict = self.params.to_dict() if hasattr(self.params, 'to_dict') else str(self.params)
        
        return {
            "params": params_dict,
            "metadata": self.metadata,
            "security_certificate": self.security_certificate.to_dict() if self.security_certificate else None,
            "decoy_estimates": self.decoy_estimates,
            "secure_key_length": self.secure_key_length,
            "raw_sifted_key_length": self.raw_sifted_key_length,
            "simulation_time_seconds": self.simulation_time_seconds,
            "status": self.status,
        }
