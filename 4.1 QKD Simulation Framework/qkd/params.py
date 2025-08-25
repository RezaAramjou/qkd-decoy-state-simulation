# -*- coding: utf-8 -*-
"""
Parameter handling, validation, and serialization for the QKD framework.

This module defines the QKDParams dataclass, which serves as a high-level
container for all simulation parameters. It includes robust validation,
serialization to and from JSON-compatible dictionaries, and type coercion
to ensure correctness and prevent common configuration errors.
"""
from __future__ import annotations

import dataclasses
import difflib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    TYPE_CHECKING,
    Protocol as TypingProtocol,
)

# Use a TYPE_CHECKING guard to prevent circular imports at runtime
if TYPE_CHECKING:
    from .protocols import Protocol as ProtocolFromModule

# Correctly handle optional numpy import to avoid mypy errors.
# Define dummy classes that will be used for isinstance checks if numpy is not installed.
class _NumpyGenericPlaceholder: pass
class _NumpyArrayPlaceholder: pass

# Assign the placeholder types initially.
NumpyGeneric: Type[Any] = _NumpyGenericPlaceholder
NumpyArray: Type[Any] = _NumpyArrayPlaceholder

try:
    import numpy as np
    # If numpy is found, overwrite the placeholders with the actual numpy types.
    NumpyGeneric = np.generic
    NumpyArray = np.ndarray
except ImportError:
    # If numpy is not installed, 'np' will be None, and the placeholder types will be used.
    np = None  # type: ignore[assignment]


from .datatypes import (
    DoubleClickPolicy,
    SecurityProof,
    ConfidenceBoundMethod,
    PulseTypeConfig,
)
from .exceptions import ParameterValidationError
from .utils.validation import parse_bool
from .sources import PoissonSource
from .channel import FiberChannel
from .detectors import ThresholdDetector
from .constants import LP_SOLVER_METHODS, DEFAULT_POISSON_TAIL_THRESHOLD

__all__ = ["QKDParams"]

# --- Module-level Setup ---

logger = logging.getLogger(__name__)
EnumType = TypeVar("EnumType", bound=Enum)

# --- Protocol Interfaces for Static Typing ---

class SerializableComponent(TypingProtocol):
    """Defines the contract for components that can be serialized to a dict."""
    def to_config_dict(self) -> Dict[str, Any]:
        ...

class Protocol(SerializableComponent, TypingProtocol):
    """Defines the contract for a QKD protocol component."""
    protocol_name: str

# --- Serialization Safety Configuration ---
MAX_SERIALIZABLE_LIST_LEN = 1000
MAX_SERIALIZABLE_DICT_LEN = 1000
MAX_SERIALIZABLE_RECURSION_DEPTH = 20

# --- Serialization and Deserialization Helpers ---

def _validate_serializable_object(obj: Any, _depth: int = 0) -> None:
    """
    Recursively validates that a fully serialized object is safe and well-formed.
    Checks for excessive depth, collection sizes, and unsupported types.
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
        raise ParameterValidationError(f"Serialization error: Unsupported type '{type(obj).__name__}' in final output.")


def _to_serializable(o: Any) -> Any:
    """
    Helper function to recursively convert objects to JSON-serializable types.
    Handles dataclasses, enums, numpy types, and custom component objects.
    """
    if hasattr(o, "to_config_dict") and callable(o.to_config_dict):
        config_dict = o.to_config_dict()
        serializable_dict = _to_serializable(config_dict)
        # Final validation after converting the component
        _validate_serializable_object(serializable_dict)
        return serializable_dict

    if np is not None:
        if isinstance(o, NumpyGeneric):
            return o.item()
        if isinstance(o, NumpyArray):
            # For large arrays, serialize as a summary to avoid huge JSON files.
            if o.size > 100:
                return {"__np_array__": {"shape": list(o.shape), "dtype": str(o.dtype)}}
            return o.tolist()

    if isinstance(o, Enum):
        return o.value

    if dataclasses.is_dataclass(o) and not isinstance(o, type):
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
    Handles direct instances, string names (case-insensitive), and raw values.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        s_value = value.strip().upper()
        for member in enum_cls:
            if member.name == s_value:
                return member
    try:
        return enum_cls(value)
    except (ValueError, TypeError) as e:
        candidates = [m.name for m in enum_cls]
        suggestion_str = ""
        if isinstance(value, str):
            matches = difflib.get_close_matches(value.upper(), candidates, n=1)
            if matches:
                suggestion_str = f" Did you mean '{matches[0]}'?"
        raise ParameterValidationError(
            f"Invalid value '{value}' for {enum_cls.__name__}. "
            f"Allowed names: {candidates}.{suggestion_str}"
        ) from e


def _coerce_type(d: Dict[str, Any], key: str, target_type: Type[Any], required: bool = True, default: Any = None) -> Any:
    """
    Extracts a key from a dict, coerces it to a target type, and handles optional/None values.
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
            return parse_bool(val)
        return target_type(val)
    except (ValueError, TypeError) as e:
        raise ParameterValidationError(
            f"Parameter '{key}' must be of type {target_type.__name__}, but got value '{val}' of type {type(val).__name__}."
        ) from e


def _migrate_config_if_needed(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder for migrating older configuration schemas to the current version.
    This allows for backward compatibility as the config format evolves.
    """
    d_migrated = d.copy()
    # Example migration:
    # if "old_param_name" in d_migrated:
    #     d_migrated["new_param_name"] = d_migrated.pop("old_param_name")
    return d_migrated


# --- Main Parameter Class ---

@dataclass(frozen=True, slots=True)
class QKDParams:
    """
    High-level, immutable container for all QKD simulation parameters.

    This class acts as the single source of truth for a simulation's configuration.
    It is responsible for validating the integrity and consistency of all parameters
    upon instantiation.
    """
    protocol: ProtocolFromModule
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

    def __post_init__(self) -> None:
        """Called after the dataclass is initialized to perform validation."""
        # Normalize the solver method name to lowercase for consistency.
        # object.__setattr__ is used because the dataclass is frozen.
        object.__setattr__(self, "lp_solver_method", self.lp_solver_method.lower())
        self._assert_component_interfaces()
        self._validate()

    def _assert_component_interfaces(self) -> None:
        """Enforces that all major components conform to their required protocols."""
        components: Dict[str, Any] = {
            "protocol": self.protocol, "source": self.source,
            "channel": self.channel, "detector": self.detector,
        }
        for name, comp in components.items():
            if not (hasattr(comp, "to_config_dict") and callable(comp.to_config_dict)):
                raise TypeError(f"Component '{name}' of type {type(comp).__name__} must implement a to_config_dict() method.")
        if not hasattr(self.protocol, "protocol_name"):
            raise TypeError(f"Protocol {type(self.protocol).__name__} must have a 'protocol_name' property.")

    def _validate(self) -> None:
        """Performs comprehensive validation of simulation parameters."""
        from .protocols import MDIQKDProtocol

        if not isinstance(self.photon_number_cap, int) or self.photon_number_cap < 0:
            raise ParameterValidationError("photon_number_cap must be a non-negative integer.")
        
        if not (0 < self.f_error_correction < 2):
             logger.warning(f"f_error_correction is typically between 1.0 and 1.2, but got {self.f_error_correction}")

        total_epsilon = self.eps_sec + self.eps_cor + self.eps_pe + self.eps_smooth
        if total_epsilon >= 1.0:
            raise ParameterValidationError(f"The sum of all epsilon values must be less than 1.0, but got {total_epsilon}.")
        
        if self.security_proof == SecurityProof.MDI_QKD and not isinstance(self.protocol, MDIQKDProtocol):
            raise ParameterValidationError("The 'mdi-qkd' security proof requires an MDIQKDProtocol instance.")

        if self.lp_solver_method not in LP_SOLVER_METHODS:
            raise ParameterValidationError(
                f"Invalid lp_solver_method '{self.lp_solver_method}'. "
                f"Allowed methods: {sorted(LP_SOLVER_METHODS)}"
            )

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
            # Exclude complex object fields, which are handled separately
            if f.name not in {"protocol", "source", "channel", "detector"}:
                value = getattr(self, f.name)
                if enums_as_values and isinstance(value, Enum):
                    value = value.value
                summary[f.name] = value
        
        if redact and "master_seed" in summary and summary["master_seed"] is not None:
            summary["master_seed"] = None # Redact the seed for safe logging
            summary["_master_seed_redacted"] = True
        return summary

    def __repr__(self) -> str:
        """Provides a safe, redacted summary of the parameters for logging."""
        summary = self.to_summary_dict(redact=True, enums_as_values=False)
        summary_str = ", ".join(f"{k}={v!r}" for k, v in summary.items())
        return f"QKDParams(protocol={self.protocol.protocol_name}, {summary_str})"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QKDParams:
        """
        Factory method to deserialize parameters from a dictionary and construct the QKDParams object.
        """
        from .protocols import BB84DecoyProtocol, MDIQKDProtocol

        d_copy = _migrate_config_if_needed(d)

        try:
            source_config = d_copy.pop("source_config")
            pulse_list = source_config.get("pulse_configs", [])
            source = PoissonSource(pulse_configs=[PulseTypeConfig(**pc) for pc in pulse_list])

            channel = FiberChannel(**d_copy.pop("channel_config"))

            detector_config = d_copy.pop("detector_config")
            if "double_click_policy" in detector_config:
                detector_config["double_click_policy"] = _coerce_enum(DoubleClickPolicy, detector_config["double_click_policy"])
            detector = ThresholdDetector(**detector_config)

            protocol_name = d_copy.pop("protocol_name")
            protocol_config = d_copy.pop("protocol_config")
            
            protocol: ProtocolFromModule
            if protocol_name == "bb84-decoy":
                # The `type: ignore` is used here because mypy cannot statically verify
                # that the unpacked `protocol_config` dictionary matches the
                # constructor signature of BB84DecoyProtocol. This is a common
                # pattern when deserializing dynamic configurations.
                protocol = BB84DecoyProtocol(source=source, **protocol_config)  # type: ignore[misc]
            elif protocol_name == "mdi-qkd":
                protocol = MDIQKDProtocol(source=source, **protocol_config)  # type: ignore[misc]
            else:
                raise NotImplementedError(f"Protocol '{protocol_name}' not supported.")
        except KeyError as e:
            raise ParameterValidationError(f"Missing required configuration section: {e}") from e
        except TypeError as e:
             raise ParameterValidationError(f"Mismatched parameters in config section: {e}") from e

        try:
            d_copy.pop("schema_version", None)
            params = {
                "security_proof": _coerce_enum(SecurityProof, d_copy.pop("security_proof")),
                "ci_method": _coerce_enum(ConfidenceBoundMethod, d_copy.pop("ci_method")),
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
                "require_tail_below": _coerce_type(d_copy, "require_tail_below", float, required=False, default=DEFAULT_POISSON_TAIL_THRESHOLD),
                "master_seed": _coerce_type(d_copy, "master_seed", int, required=False, default=None),
            }
            if d_copy:
                raise ParameterValidationError(f"Unknown parameter keys provided: {sorted(d_copy.keys())}")

            return cls(protocol=protocol, source=source, channel=channel, detector=detector, **params)
        except (KeyError, TypeError, ValueError) as e:
            raise ParameterValidationError(f"Failed to parse parameters: {e}") from e

