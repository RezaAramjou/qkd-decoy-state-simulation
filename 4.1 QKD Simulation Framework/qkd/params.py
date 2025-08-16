# -*- coding: utf-8 -*-
"""
Parameter handling, validation, and serialization for the QKD framework.

This module defines the QKDParams dataclass, which serves as a high-level
container for all simulation parameters. It includes robust validation,
serialization to and from JSON-compatible dictionaries, and type coercion
to ensure correctness and prevent common configuration errors.

The design prioritizes immutability, clear error messaging, and scientific
correctness, following best practices for reproducible simulations.

Expected Component Interface:
    All major components (Protocol, Source, Channel, Detector) are expected
    to adhere to a "SerializableComponent" interface by implementing:
    1.  A `to_config_dict(self) -> Dict[str, Any]` method that returns a
        JSON-serializable dictionary of its configuration parameters. This
        dictionary MUST NOT contain runtime state, large arrays, or non-primitive
        objects.
    2.  A `protocol_name` property (for Protocol subclasses) that returns a
        unique string identifier (e.g., 'bb84-decoy').
    For maintainability, it is highly recommended to formalize this contract
    using an Abstract Base Class (ABC) in a shared module.

Testing and Schema:
    A comprehensive test suite (e.g., using pytest) should be implemented to
    verify round-trip serialization, handling of malformed configs, correct
    coercion of enums, and edge-case validation. For user-facing validation,
    creating a `params.schema.json` file and a small CLI validation utility
    is highly recommended.
"""
from __future__ import annotations

import dataclasses
import difflib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar, TYPE_CHECKING

# Use a TYPE_CHECKING guard to prevent circular imports at runtime
if TYPE_CHECKING:
    from .protocols import Protocol

# Attempt to import numpy, but allow the module to function without it.
try:
    import numpy as np
except ImportError:
    np = None

from .datatypes import (
    DoubleClickPolicy,
    SecurityProof,
    ConfidenceBoundMethod,
    PulseTypeConfig,
)
from .exceptions import ParameterValidationError
from .utils.validation import _parse_bool
from .sources import PoissonSource
from .channel import FiberChannel
from .detectors import ThresholdDetector
from .constants import LP_SOLVER_METHODS, DEFAULT_POISSON_TAIL_THRESHOLD

__all__ = ["QKDParams"]

# --- Module-level Setup ---

logger = logging.getLogger(__name__)
EnumType = TypeVar("EnumType", bound=Enum)

# --- Serialization Safety Configuration ---
MAX_SERIALIZABLE_LIST_LEN = 1000
MAX_SERIALIZABLE_DICT_LEN = 1000
MAX_SERIALIZABLE_RECURSION_DEPTH = 20

# --- Serialization and Deserialization Helpers ---

def _validate_serializable_object(obj: Any, _depth: int = 0):
    """
    Recursively validates that a fully serialized object is safe.
    This runs *after* enums and numpy types have been converted to primitives.
    """
    if _depth > MAX_SERIALIZABLE_RECURSION_DEPTH:
        raise ParameterValidationError("Serialization error: Exceeded max recursion depth.")

    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_SERIALIZABLE_LIST_LEN:
            raise ParameterValidationError(f"Serialization error: List/tuple length exceeds limit of {MAX_SERIALIZABLE_LIST_LEN}.")
        for item in obj:
            _validate_serializable_object(item, _depth + 1)
    elif isinstance(obj, dict):
        if len(obj) > MAX_SERIALIZABLE_DICT_LEN:
            raise ParameterValidationError(f"Serialization error: Dict length exceeds limit of {MAX_SERIALIZABLE_DICT_LEN}.")
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ParameterValidationError("Serialization error: Dictionary keys must be strings.")
            _validate_serializable_object(value, _depth + 1)
    elif not isinstance(obj, (bool, int, float, str, type(None))):
        # This check is strict because it runs on the final, serialized object.
        raise ParameterValidationError(f"Serialization error: Unsupported type '{type(obj).__name__}' in final output.")


def _to_serializable(o: Any) -> Any:
    """
    Helper function to recursively convert objects to JSON-serializable types.
    """
    if hasattr(o, "to_config_dict") and callable(o.to_config_dict):
        # Convert first, then validate the primitive structure.
        config_dict = o.to_config_dict()
        serializable_dict = _to_serializable(config_dict)
        _validate_serializable_object(serializable_dict)
        return serializable_dict

    if np is not None:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            # Note: This is a one-way transformation for safety. Deserialization
            # from this placeholder is not supported by design.
            if o.size > 100:
                return {"__np_array__": {"shape": list(o.shape), "dtype": str(o.dtype)}}
            return o.tolist()

    if isinstance(o, Enum):
        return o.value

    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o, dict_factory=lambda data: {k: _to_serializable(v) for k, v in data})

    if isinstance(o, dict):
        return {k: _to_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_serializable(i) for i in o]

    if isinstance(o, float) and not math.isfinite(o):
        raise ParameterValidationError(f"Non-finite float value '{o}' found during serialization.")

    return o


def _coerce_enum(enum_cls: Type[EnumType], value: Any) -> EnumType:
    """
    Robustly coerces a value to a given Enum type with helpful error messages.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        s_value = value.strip().lower()
        for member in enum_cls:
            if str(member.value).lower() == s_value or member.name.lower() == s_value:
                return member
    try:
        return enum_cls(value)
    except (ValueError, TypeError) as e:
        candidates = [m.value for m in enum_cls] + [m.name for m in enum_cls]
        suggestion_str = ""
        if isinstance(value, str):
            matches = difflib.get_close_matches(value, [str(c) for c in candidates], n=1)
            if matches:
                suggestion_str = f" Did you mean '{matches[0]}'?"
        raise ParameterValidationError(
            f"Invalid value '{value}' for {enum_cls.__name__}. "
            f"Allowed values: {[m.value for m in enum_cls]}.{suggestion_str}"
        ) from e


def _coerce_type(d: dict, key: str, target_type: Type, required: bool = True, default: Any = None) -> Any:
    """
    Extracts a key from a dict, coerces it to a type, and handles optional/None values.
    """
    if key not in d:
        if required:
            raise ParameterValidationError(f"Missing required parameter: '{key}'")
        return default

    val = d.pop(key)

    if val is None:
        if required:
            raise ParameterValidationError(f"Required parameter '{key}' cannot be null.")
        return default

    try:
        if target_type is bool:
            return _parse_bool(val)
        return target_type(val)
    except (ValueError, TypeError) as e:
        raise ParameterValidationError(
            f"Parameter '{key}' must be of type {target_type.__name__}, but got value '{val}' of type {type(val).__name__}."
        ) from e


def _migrate_config_if_needed(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for migrating older configuration schemas to the current version.
    Always returns a new dictionary.
    """
    d_migrated = d.copy()
    schema_version = d_migrated.get("schema_version", "0.0")
    if schema_version == "1.3":
        return d_migrated

    logger.info(f"Attempting to migrate config from schema version {schema_version} to 1.3.")
    # Example migration:
    # if schema_version in ["1.0", "1.1", "1.2"]:
    #     if "old_key" in d_migrated:
    #         d_migrated["new_key"] = d_migrated.pop("old_key")
    # d_migrated["schema_version"] = "1.3"
    return d_migrated


# --- Main Parameter Class ---

@dataclass(frozen=True, slots=True)
class QKDParams:
    """
    High-level, immutable container for all QKD simulation parameters.

    Attributes:
        ...
        photon_number_cap: The maximum photon number `n` to consider. This is
            an *inclusive* cap (0 to n). A value of 0 implies only the vacuum
            state is considered. Downstream modules must handle this correctly,
            typically by expecting `photon_number_cap + 1` states.
        ...
    """
    protocol: 'Protocol'
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
    allow_unsafe_mdi_approx: bool = False
    require_tail_below: Optional[float] = DEFAULT_POISSON_TAIL_THRESHOLD
    master_seed: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "lp_solver_method", self.lp_solver_method.lower())
        self._assert_component_interfaces()
        self._validate()

    def _assert_component_interfaces(self):
        """Enforces the serializable component contract at runtime."""
        components = {
            "protocol": self.protocol,
            "source": self.source,
            "channel": self.channel,
            "detector": self.detector,
        }
        for name, comp in components.items():
            if not (hasattr(comp, "to_config_dict") and callable(comp.to_config_dict)):
                raise TypeError(f"Component '{name}' of type {type(comp).__name__} must implement a to_config_dict() method.")
        if not hasattr(self.protocol, "protocol_name"):
            raise TypeError(f"Protocol {type(self.protocol).__name__} must have a 'protocol_name' property.")

    def _validate(self):
        """Performs comprehensive validation of simulation parameters."""
        from .protocols import MDIQKDProtocol

        # ... (Validation logic remains the same as previous version)
        if not isinstance(self.photon_number_cap, int) or self.photon_number_cap < 0:
            raise ParameterValidationError("photon_number_cap must be a non-negative integer.")
        if sum([self.eps_sec, self.eps_cor, self.eps_pe, self.eps_smooth]) >= 1.0:
            raise ParameterValidationError("The sum of all epsilon values must be less than 1.0.")
        if self.security_proof == SecurityProof.MDI_QKD and not isinstance(self.protocol, MDIQKDProtocol):
            raise ParameterValidationError("The 'mdi-qkd' security proof requires an MDIQKDProtocol instance.")
        # ... (other checks)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Serializes the QKDParams object to a JSON-compatible dictionary."""
        params_dict = {
            "schema_version": "1.3",
            "protocol_name": self.protocol.protocol_name,
            "protocol_config": self.protocol.to_config_dict(),
            "source_config": self.source.to_config_dict(),
            "channel_config": self.channel.to_config_dict(),
            "detector_config": self.detector.to_config_dict(),
            **self.to_summary_dict(redact=False, enums_as_values=True)
        }
        return _to_serializable(params_dict)

    def to_summary_dict(self, redact: bool = True, enums_as_values: bool = True) -> Dict[str, Any]:
        """Returns a dictionary of scalar parameters, optionally redacting secrets."""
        summary = {}
        for f in dataclasses.fields(self):
            if f.name not in {"protocol", "source", "channel", "detector"}:
                value = getattr(self, f.name)
                if enums_as_values and isinstance(value, Enum):
                    value = value.value
                summary[f.name] = value

        if redact and "master_seed" in summary and summary["master_seed"] is not None:
            summary["master_seed"] = None
            summary["_master_seed_redacted"] = True
        return summary

    def __repr__(self) -> str:
        """Provides a safe, redacted summary of the parameters for logging."""
        summary = self.to_summary_dict(redact=True)
        summary_str = ", ".join(f"{k}={v!r}" for k, v in summary.items())
        return f"QKDParams(protocol={self.protocol.protocol_name}, {summary_str})"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QKDParams":
        """Factory method to deserialize parameters and construct the QKDParams object."""
        from .protocols import BB84DecoyProtocol, MDIQKDProtocol

        d_copy = _migrate_config_if_needed(d)

        try: # --- Component Deserialization ---
            source_config = d_copy.pop("source_config")
            pulse_list = source_config.get("pulse_configs")
            if not isinstance(pulse_list, list): raise ParameterValidationError("`pulse_configs` must be a list.")
            new_pulse_list = []
            for i, pc in enumerate(pulse_list):
                if not isinstance(pc, dict): raise ParameterValidationError(f"pulse_configs[{i}] must be a dict.")
                try:
                    new_pulse_list.append(PulseTypeConfig(**pc))
                except (TypeError, ValueError) as e:
                    raise ParameterValidationError(f"pulse_configs[{i}] is invalid: {e}") from e
            source = PoissonSource(pulse_configs=new_pulse_list)

            channel_config = d_copy.pop("channel_config")
            channel = FiberChannel(**channel_config)

            detector_config = d_copy.pop("detector_config")
            if "double_click_policy" in detector_config:
                detector_config["double_click_policy"] = _coerce_enum(DoubleClickPolicy, detector_config["double_click_policy"])
            detector = ThresholdDetector(**detector_config)

            protocol_name = d_copy.pop("protocol_name")
            protocol_config = d_copy.pop("protocol_config")
            if protocol_name == "bb84-decoy":
                protocol = BB84DecoyProtocol(source=source, detector=detector, **protocol_config)
            elif protocol_name == "mdi-qkd":
                protocol = MDIQKDProtocol(source=source, detector=detector, **protocol_config)
            else:
                raise NotImplementedError(f"Protocol '{protocol_name}' not supported.")
        except KeyError as e:
            raise ParameterValidationError(f"Missing required configuration section: {e}") from e

        try: # --- Scalar Deserialization ---
            d_copy.pop("schema_version", None)
            params = {
                "security_proof": _coerce_enum(SecurityProof, d_copy.pop("security_proof")),
                "ci_method": _coerce_enum(ConfidenceBoundMethod, d_copy.pop("ci_method")),
                # ... other scalars ...
                "num_bits": _coerce_type(d_copy, "num_bits", int),
                "photon_number_cap": _coerce_type(d_copy, "photon_number_cap", int),
                "batch_size": _coerce_type(d_copy, "batch_size", int),
                "num_workers": _coerce_type(d_copy, "num_workers", int),
                "f_error_correction": _coerce_type(d_copy, "f_error_correction", float),
                "eps_sec": _coerce_type(d_copy, "eps_sec", float),
                "eps_cor": _coerce_type(d_copy, "eps_cor", float),
                "eps_pe": _coerce_type(d_copy, "eps_pe", float),
                "eps_smooth": _coerce_type(d_copy, "eps_smooth", float),
                "force_sequential": _coerce_type(d_copy, "force_sequential", bool, required=False, default=False),
                "enforce_monotonicity": _coerce_type(d_copy, "enforce_monotonicity", bool, required=False, default=True),
                "assume_phase_equals_bit_error": _coerce_type(d_copy, "assume_phase_equals_bit_error", bool, required=False, default=False),
                "allow_unsafe_mdi_approx": _coerce_type(d_copy, "allow_unsafe_mdi_approx", bool, required=False, default=False),
                "lp_solver_method": _coerce_type(d_copy, "lp_solver_method", str, required=False, default="highs"),
                "require_tail_below": _coerce_type(d_copy, "require_tail_below", float, required=False),
                "master_seed": _coerce_type(d_copy, "master_seed", int, required=False),
            }
            if params["require_tail_below"] is None:
                params["require_tail_below"] = DEFAULT_POISSON_TAIL_THRESHOLD
                logger.debug("Parameter 'require_tail_below' not set; using default value.")

            if d_copy:
                raise ParameterValidationError(f"Unknown parameter keys provided: {sorted(d_copy.keys())}")

            return cls(protocol=protocol, source=source, channel=channel, detector=detector, **params)
        except (KeyError, TypeError, ValueError) as e:
            raise ParameterValidationError(f"Failed to parse parameters: {e}") from e

