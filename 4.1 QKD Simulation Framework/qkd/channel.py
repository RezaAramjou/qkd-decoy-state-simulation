# -*- coding: utf-8 -*-
"""
Models a simple quantum channel, focusing on fiber optic attenuation.

This module provides a robust, immutable dataclass `FiberChannel` to represent
an optical fiber link. It calculates the channel's transmittance based on
its length and loss characteristics. This model is scalar-only and does not
support vectorized inputs.

Key Features:
- Immutable and unhashable objects for safe use in logic but not as dict keys.
- Strict validation of physical parameters (non-negative, finite, sensible bounds).
- Caching of calculated properties for performance.
- Serialization/deserialization support via dictionary conversion.
- Comprehensive docstrings and type annotations.

Raises:
    ParameterValidationError: If initialization parameters are invalid (e.g.,
                              negative, non-finite, or outside physical bounds).

Security Note:
    This is a simulation model. The parameters used (e.g., distance, loss) are
    not cryptographic secrets. However, using unrealistic or overly optimistic
    channel parameters in a simulation (e.g., zero loss over a long distance)
    could lead to insecure conclusions if the simulation results are used to
    design or validate a real-world Quantum Key Distribution (QKD) system.
    This model only accounts for fiber attenuation; it does not include other
    physical effects like polarization drift, dispersion, or connector losses.
    Always use realistic, measured, or worst-case values for security-critical
    simulations.
"""
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple, Type, TypeVar, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

__version__ = "5.0.0"

# Export the primary class and its specific exception for a clear, usable API.
__all__ = ["FiberChannel", "ParameterValidationError"]

# Handle conditional import for static analysis vs. runtime.
# This pattern allows mypy to find the "real" exception class while
# providing a runtime fallback if the module is not found, resolving the
# [no-redef] error.
if TYPE_CHECKING:
    from .exceptions import ParameterValidationError
else:
    try:
        from .exceptions import ParameterValidationError
    except ImportError:
        logger.debug("Could not import ParameterValidationError from .exceptions, using fallback.")
        class ParameterValidationError(ValueError):
            """A fallback exception for parameter validation errors."""
            pass

# Type aliases for clarity.
Transmittance = float
Distance = float
Loss = float
Numeric = Union[int, float]

# Generic type variable for classmethods.
T_FiberChannel = TypeVar("T_FiberChannel", bound="FiberChannel")

# --- Module-level constants for configuration and clarity ---

# Define explicit tolerances for floating-point equality checks.
# These are chosen as sensible defaults for typical experimental parameter rounding.
EQ_REL_TOL = 1e-9
EQ_ABS_TOL = 1e-12
DB_ZERO_TOLERANCE = 1e-12 # Absolute tolerance for treating total_loss_db as zero.

# The threshold for logging a debug message about near-zero transmittance.
# sys.float_info.min is the smallest positive normalized float, making it a
# non-arbitrary machine-level floor for non-zero values.
TRANSMITTANCE_UNDERFLOW_THRESHOLD = sys.float_info.min

# Define reasonable physical bounds for parameters, with units in their names.
# Rationale: 1M km is beyond most practical fiber links (e.g., interplanetary).
MAX_DISTANCE_KM = 1_000_000
# Rationale: 10 dB/km is very high loss, covering most pathological media.
MAX_FIBER_LOSS_DB_KM = 10.0

# Explicit mapping of parameter names to units for robust error messages.
UNITS = {"distance_km": "km", "fiber_loss_db_km": "dB/km"}


# The original conditional decorator was confusing mypy, leading to [call-arg]
# errors. This simplified version is compatible across Python versions and
# correctly allows keyword arguments.
@dataclass(frozen=True, slots=True)
class FiberChannel:
    """
    Models a simple optical fiber channel defined by attenuation.

    This class is immutable and unhashable. Its equality is based on approximate
    floating-point comparison (`math.isclose`), and it is made unhashable to
    prevent incorrect usage in sets or as dictionary keys where such equality
    semantics would be violated. This model is scalar-only.

    Args:
        distance_km: The length of the fiber channel in kilometers (km).
                     Accepts int or float. Must be non-negative and finite.
        fiber_loss_db_km: The attenuation of the fiber in decibels per
                          kilometer (dB/km). Accepts int or float. Must be
                          non-negative and finite. Zero loss is permitted.

    Example:
        >>> # A 50km fiber with standard 0.2 dB/km loss
        >>> channel = FiberChannel(distance_km=50, fiber_loss_db_km=0.2)
        >>> print(channel)
        FiberChannel(distance_km=50.000, fiber_loss_db_km=0.200, total_loss_db=10.00, transmittance=1.0000e-01)
    """
    distance_km: Distance
    fiber_loss_db_km: Loss

    _total_loss_db: Loss = field(init=False, repr=False)
    _transmittance: Transmittance = field(init=False, repr=False)

    # This line is removed to fix the [assignment] error.
    # For a frozen dataclass with a custom __eq__ method, __hash__ is
    # automatically and correctly set to None, making this explicit
    # assignment redundant and incorrectly typed.
    # __hash__ = None

    def __post_init__(self):
        """
        Validates input parameters and pre-computes derived properties.
        """
        try:
            coerced_distance = float(self.distance_km)
            coerced_loss = float(self.fiber_loss_db_km)
        except (ValueError, TypeError) as e:
            raise ParameterValidationError(
                f"Input parameters must be convertible to float. Original error: {e}"
            ) from e

        self._validate_parameter('distance_km', coerced_distance, MAX_DISTANCE_KM)
        self._validate_parameter('fiber_loss_db_km', coerced_loss, MAX_FIBER_LOSS_DB_KM)

        # Use object.__setattr__ because the class is frozen.
        object.__setattr__(self, 'distance_km', coerced_distance)
        object.__setattr__(self, 'fiber_loss_db_km', coerced_loss)

        total_db = coerced_distance * coerced_loss
        if not math.isfinite(total_db):
            raise ParameterValidationError(
                f"Calculated total_loss_db is not finite ({total_db!r}). "
                f"Check for excessively large inputs."
            )
        object.__setattr__(self, '_total_loss_db', total_db)

        if abs(total_db) < DB_ZERO_TOLERANCE:
            transmittance = 1.0
        else:
            # Use explicit 10.0 for clarity in the dB to linear conversion.
            transmittance = 10.0 ** (-total_db / 10.0)

        clamped_transmittance = min(1.0, max(0.0, transmittance))
        object.__setattr__(self, '_transmittance', clamped_transmittance)

        if 0 < clamped_transmittance <= TRANSMITTANCE_UNDERFLOW_THRESHOLD:
            logger.debug(
                "Transmittance is near or below underflow threshold for "
                "distance_km=%.3f, fiber_loss_db_km=%.3f",
                coerced_distance, coerced_loss
            )

    def _validate_parameter(self, name: str, value: float, upper_bound: float):
        """A helper to validate a single numeric parameter."""
        units = UNITS.get(name, "")
        if not math.isfinite(value):
            raise ParameterValidationError(
                f"Parameter '{name}' must be a finite number, but got {value!r}."
            )
        if value < 0:
            raise ParameterValidationError(
                f"Parameter '{name}' must be non-negative, but got {value!r}."
            )
        if value > upper_bound:
            raise ParameterValidationError(
                f"Parameter '{name}' ({value!r}) exceeds a reasonable physical "
                f"limit of {upper_bound} {units}."
            )

    @property
    def transmittance(self) -> Transmittance:
        """The total channel transmittance (a probability in [0,1])."""
        return self._transmittance

    @property
    def total_loss_db(self) -> Loss:
        """The total loss of the channel in decibels (dB)."""
        return self._total_loss_db

    @property
    def is_lossless(self) -> bool:
        """Returns True if the channel has effectively zero loss."""
        return abs(self.total_loss_db) < DB_ZERO_TOLERANCE

    def linear_loss_per_km(self) -> float:
        """Returns the linear attenuation per kilometer (unitless)."""
        return 10.0 ** (-self.fiber_loss_db_km / 10.0)

    def as_tuple(self) -> Tuple[Distance, Loss]:
        """Returns the channel's defining parameters as a tuple."""
        return (self.distance_km, self.fiber_loss_db_km)

    def to_config_dict(self) -> Dict[str, Any]:
        """Serializes the channel configuration to a dictionary."""
        return {
            "distance_km": self.distance_km,
            "fiber_loss_db_km": self.fiber_loss_db_km,
        }

    @classmethod
    def from_config_dict(cls: Type[T_FiberChannel], config: Mapping[str, Any]) -> T_FiberChannel:
        """
        Creates a FiberChannel instance from a configuration mapping.

        >>> d = {'distance_km': 25, 'fiber_loss_db_km': 0.16}
        >>> FiberChannel.from_config_dict(d).total_loss_db
        4.0
        """
        required_keys = {"distance_km", "fiber_loss_db_km"}
        try:
            missing_keys = required_keys - set(config.keys())
        except TypeError as e:
            raise ParameterValidationError(
                f"Configuration must be a mapping-like object, but got "
                f"{type(config).__name__}. Original error: {e}"
            ) from e

        if missing_keys:
            raise ParameterValidationError(
                f"Configuration is missing required keys: {sorted(list(missing_keys))}"
            )

        # The simplified decorator ensures this call is correctly typed.
        return cls(
            distance_km=config["distance_km"],
            fiber_loss_db_km=config["fiber_loss_db_km"],
        )


    @classmethod
    def from_total_loss(
        cls: Type[T_FiberChannel], distance_km: Numeric, total_loss_db: Numeric
    ) -> T_FiberChannel:
        """
        Creates a FiberChannel from a total loss value and distance.

        >>> c = FiberChannel.from_total_loss(distance_km=100, total_loss_db=20)
        >>> c.fiber_loss_db_km
        0.2
        """
        try:
            coerced_distance = float(distance_km)
            coerced_total_loss = float(total_loss_db)
        except (ValueError, TypeError) as e:
            raise ParameterValidationError(f"Inputs must be numeric. Error: {e}") from e

        if not math.isfinite(coerced_total_loss) or coerced_total_loss < 0:
                 raise ParameterValidationError(
                     f"total_loss_db must be non-negative and finite, got {total_loss_db!r}."
                )
        if coerced_distance <= 0:
            raise ParameterValidationError(
                f"distance_km must be positive when creating from total_loss_db, got {coerced_distance!r}."
            )

        # The constructor will validate the calculated per-km loss against MAX_FIBER_LOSS_DB_KM.
        fiber_loss_db_km = coerced_total_loss / coerced_distance
        return cls(distance_km=coerced_distance, fiber_loss_db_km=fiber_loss_db_km)

    def __repr__(self) -> str:
        """Provides a detailed, unambiguous representation of the object."""
        trans_fmt = ".4f" if self.transmittance > 1e-4 else ".4e"
        loss_fmt = ".2f" if abs(self.total_loss_db) < 1e6 else ".2e"
        return (
            f"{self.__class__.__name__}("
            f"distance_km={self.distance_km:.3f}, "
            f"fiber_loss_db_km={self.fiber_loss_db_km:.3f}, "
            f"total_loss_db={self.total_loss_db:{loss_fmt}}, "
            f"transmittance={self.transmittance:{trans_fmt}})"
        )

    def __eq__(self, other: Any) -> bool:
        """
        Compares two FiberChannel instances for approximate equality.
        Uses `math.isclose()` with explicit, module-level tolerances.
        """
        if not isinstance(other, FiberChannel):
            return NotImplemented
        return (
            math.isclose(
                self.distance_km, other.distance_km,
                rel_tol=EQ_REL_TOL, abs_tol=EQ_ABS_TOL
            ) and
            math.isclose(
                self.fiber_loss_db_km, other.fiber_loss_db_km,
                rel_tol=EQ_REL_TOL, abs_tol=EQ_ABS_TOL
            )
        )

    def validate(self) -> None:
        """
        Explicitly runs the internal validation checks.
        This is an API hook; since the object is immutable, its state is
        guaranteed to be valid after successful initialization.
        """
        self._validate_parameter('distance_km', self.distance_km, MAX_DISTANCE_KM)
        self._validate_parameter('fiber_loss_db_km', self.fiber_loss_db_km, MAX_FIBER_LOSS_DB_KM)
        logger.debug("Instance %r passed validation.", self)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    exit_code = 0

    def run_test(condition: bool, fail_message: str):
        """Helper to run a test and set exit code on failure."""
        global exit_code
        if not condition:
            logger.error(f"FAIL: {fail_message}")
            exit_code = 1

    logger.info(f"--- Running Smoke Tests for {__file__} (v{__version__}) ---")

    try:
        channel1 = FiberChannel(distance_km=50, fiber_loss_db_km=0.2)
        logger.info(f"Standard Channel: {channel1}")
        run_test(math.isclose(channel1.total_loss_db, 10.0), "Total loss calculation mismatch.")
        run_test(math.isclose(channel1.transmittance, 0.1), "Transmittance calculation mismatch.")
        run_test(not channel1.is_lossless, "is_lossless property failed.")
        run_test(channel1.as_tuple() == (50.0, 0.2), "as_tuple() method failed.")

        config = channel1.to_config_dict()
        recreated = FiberChannel.from_config_dict(config)
        run_test(channel1 == recreated, "Serialization/deserialization failed.")
        logger.info("Serialization check: PASS")

        from_total = FiberChannel.from_total_loss(distance_km=50, total_loss_db=10.0)
        run_test(channel1 == from_total, "Alternative constructor failed.")
        logger.info("Alt. Constructor check: PASS")

        logger.info("--- Testing Invalid Inputs ---")
        # Added explicit type hint to fix [arg-type] error for **inputs.
        invalid_inputs: List[Dict[str, Any]] = [
            {'distance_km': -10, 'fiber_loss_db_km': 0.2},
            {'distance_km': 10, 'fiber_loss_db_km': float('nan')},
            {'distance_km': "ten", 'fiber_loss_db_km': 0.2},
            {'distance_km': 1e9, 'fiber_loss_db_km': 0.2},
            {'distance_km': 10, 'fiber_loss_db_km': 20}, # Exceeds max loss
        ]
        for i, inputs in enumerate(invalid_inputs):
            try:
                FiberChannel(**inputs)
                run_test(False, f"Expected error for invalid input #{i+1}: {inputs}")
            except ParameterValidationError as e:
                logger.info(f"PASS: Correctly caught error for {inputs}: {e}")

        logger.info("--- Testing Unhashable ---")
        try:
            my_set = {channel1}
            run_test(False, "Object should be unhashable but was added to a set.")
        except TypeError as e:
            logger.info(f"PASS: Correctly caught TypeError when hashing: {e}")

    except Exception as e:
        logger.error("A test failed unexpectedly: %s", e, exc_info=True)
        exit_code = 1

    if exit_code == 0:
        logger.info("\n--- All smoke tests passed successfully! ---")
    else:
        logger.error("\n--- Some smoke tests failed. ---")

    sys.exit(exit_code)

