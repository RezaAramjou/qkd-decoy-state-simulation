# -*- coding: utf-8 -*-
"""
Custom, research-grade exceptions for the QKD simulation framework.

This module defines a comprehensive, well-structured exception hierarchy for the
framework. It is designed to be:
  - Hierarchical: A single base exception (`QKDException`) allows for catching all
    framework-specific errors.
  - Context-Rich: Exceptions carry structured, typed context beyond a simple
    message, aiding programmatic handling and debugging.
  - Serializable: A `.to_dict()` method provides a JSON-safe representation for
    structured logging, telemetry, and reproducible error reporting.
  - Stable: A machine-readable `ErrorCode` enum provides stable identifiers for
    automated error handling.
  - User-Friendly: Enhanced string representations, clear docstrings, and
    convenience factories (e.g., for SciPy results) improve developer experience.
  - Robust: Exceptions are picklable for use across multiprocessing boundaries,
    with careful sanitization of attached diagnostic data.
"""
from __future__ import annotations

import json
import logging
import pickle
from enum import Enum
from typing import Any, Dict, Optional, Mapping, Type

# Exported names for the module's public API
__all__ = [
    "QKDException",
    "ParameterValidationError",
    "ConfigurationError",
    "QKDSimulationError",
    "LPFailureError",
    "SimulationInterruptedError",
    "ErrorCode",
]


class ErrorCode(str, Enum):
    """
    Stable, machine-readable error codes for QKD exceptions.
    Using an Enum ensures consistency and prevents typos.
    """
    # General Errors
    UNCATEGORIZED = "ERR_UNCATEGORIZED"
    INTERRUPTED = "ERR_INTERRUPTED"

    # Configuration and Parameter Errors
    CONFIG = "ERR_CONFIG"
    PARAM_VALIDATION = "ERR_PARAM_VALIDATION"

    # Runtime Simulation Errors
    SIMULATION = "ERR_SIMULATION"
    LP_SOLVER_FAILED = "ERR_LP_SOLVER_FAILED"


def _sanitize_for_serialization(value: Any) -> Any:
    """
    Recursively sanitizes a value to ensure it is JSON-serializable and picklable.
    Converts numpy scalars to Python natives, and stringifies unknown objects.
    """
    # Attempt to import numpy, but don't fail if it's not installed.
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False

    if has_numpy and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_for_serialization(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_serialization(item) for item in value]

    # Fallback: try to serialize via JSON roundtrip, otherwise stringify.
    try:
        # This is a simple way to check for basic serializability
        json.dumps(value)
        return value
    except (TypeError, OverflowError):
        return str(value)


class QKDException(Exception):
    """
    Base class for all QKD framework exceptions.

    This class provides core functionality inherited by all other custom exceptions,
    including structured context, error codes, serialization, and logging helpers.

    Attributes:
        message (str): Human-readable error message.
        code (ErrorCode): Machine-readable error code.
        context (Dict[str, Any]): Additional structured context for logging.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.UNCATEGORIZED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = {} if context is None else context

        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Provides a clear, informative string representation."""
        return f"{self.__class__.__name__} ({self.code.value}): {self.message}"

    def __repr__(self) -> str:
        """Provides an unambiguous representation for developers."""
        context_keys = f", context_keys={list(self.context.keys())!r}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}, code={self.code!r}{context_keys})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a JSON-serializable dictionary representing the exception.
        This is ideal for structured logging and telemetry.
        """
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code.value,
            "context": _sanitize_for_serialization(self.context),
            "cause": repr(self.__cause__) if self.__cause__ else None,
        }

    def log(self, logger: logging.Logger, level: int = logging.ERROR) -> None:
        """
        Logs the exception's structured data using the provided logger.

        Args:
            logger: The logger instance to use.
            level: The logging level (e.g., logging.ERROR, logging.WARNING).
        """
        logger.log(level, self.message, extra={"qkd_error": self.to_dict()})


class ParameterValidationError(QKDException, ValueError):
    """
    Raised when a simulation parameter or user input is invalid.
    Inherits from ValueError for semantic compatibility.

    Example:
        if not 0 <= attenuation <= 1:
            raise ParameterValidationError(
                "Attenuation must be between 0 and 1",
                param_name="attenuation",
                param_value=attenuation
            )
    """

    def __init__(
        self,
        message: str,
        *,
        param_name: Optional[str] = None,
        param_value: Any = None,
        code: ErrorCode = ErrorCode.PARAM_VALIDATION,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        ctx = context or {}
        if param_name is not None:
            ctx.setdefault("param_name", param_name)
        if param_value is not None:
            ctx.setdefault("param_value", param_value)
        super().__init__(message, code=code, context=ctx, cause=cause)


class ConfigurationError(QKDException):
    """
    Raised for errors in configuration, such as missing files,
    incompatible components, or structural problems in config files.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.CONFIG,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, code=code, context=context, cause=cause)


class QKDSimulationError(QKDException, RuntimeError):
    """
    Generic runtime error during simulation execution.
    Inherits from RuntimeError for semantic compatibility.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.SIMULATION,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, code=code, context=context, cause=cause)


class LPFailureError(QKDSimulationError):
    """
    Raised when a linear programming (LP) solver fails to find a solution.
    This exception captures structured diagnostics from the solver.

    Attributes:
        status (Optional[Any]): The raw status from the solver (e.g., int or str).
        solver_message (Optional[str]): The human-readable message from the solver.
        diagnostics (Dict[str, Any]): A sanitized dictionary of solver diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Any = None,
        solver_message: Optional[str] = None,
        diagnostics: Optional[Mapping[str, Any]] = None,
        code: ErrorCode = ErrorCode.LP_SOLVER_FAILED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        ctx = context or {}
        ctx.setdefault("lp_status", status)
        ctx.setdefault("lp_solver_message", solver_message)
        ctx.setdefault("lp_diagnostics", diagnostics)

        super().__init__(message, code=code, context=ctx, cause=cause)
        self.status = status
        self.solver_message = solver_message
        self.diagnostics = _sanitize_for_serialization(diagnostics or {})

    @classmethod
    def from_scipy_result(
        cls: Type[LPFailureError],
        result: Any,
        **kwargs: Any,
    ) -> LPFailureError:
        """
        Convenience factory to create an LPFailureError from a SciPy result object.
        `scipy.optimize.linprog` returns an object with `status`, `message`, etc.
        """
        # Extract attributes safely, providing defaults if they don't exist.
        status = getattr(result, "status", "unknown")
        solver_message = getattr(result, "message", "No message provided.")

        # Create a diagnostics dict from the result object's attributes.
        diagnostics = result.__dict__ if hasattr(result, "__dict__") else {}

        message = f"LP solver failed with status '{status}': {solver_message}"

        return cls(
            message,
            status=status,
            solver_message=solver_message,
            diagnostics=diagnostics,
            **kwargs,
        )


class SimulationInterruptedError(QKDSimulationError):
    """
    Raised when a simulation is intentionally interrupted (e.g., by Ctrl+C).
    This allows for graceful shutdown logic to distinguish from unexpected errors.
    """

    def __init__(
        self,
        message: str = "Simulation interrupted by user.",
        *,
        code: ErrorCode = ErrorCode.INTERRUPTED,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message, code=code, context=context, cause=cause)


# ==============================================================================
# Demonstration and Self-Testing Block
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Demonstrations and Tests for QKD Exceptions ---")

    # 1. Basic Exception Raising and Catching
    print("\n1. Testing basic exception raising and catching:")
    try:
        raise ParameterValidationError(
            "Decoy state probability must be non-negative.",
            param_name="p_decoy",
            param_value=-0.1,
        )
    except QKDException as e:
        print(f"   Caught: {e}")
        print(f"   Repr: {e!r}")
        assert e.code == ErrorCode.PARAM_VALIDATION
        assert e.context.get("param_name") == "p_decoy"

    # 2. Serialization with to_dict()
    print("\n2. Testing serialization with .to_dict():")
    try:
        raise ConfigurationError(
            "Protocol file 'bb84.toml' not found.",
            context={"path": "/etc/qkd/protocols"},
            cause=FileNotFoundError("No such file or directory"),
        )
    except QKDException as e:
        error_dict = e.to_dict()
        print(f"   Serialized dict: {json.dumps(error_dict, indent=2)}")
        assert error_dict["type"] == "ConfigurationError"
        assert error_dict["code"] == "ERR_CONFIG"
        assert "FileNotFoundError" in error_dict["cause"]

    # 3. LPFailureError with SciPy Factory
    print("\n3. Testing LPFailureError with a mock SciPy result:")
    # Mock a failed result object from scipy.optimize.linprog
    class MockSciPyResult:
        def __init__(self):
            import numpy as np
            self.status = 2  # Infeasible
            self.message = "The problem is infeasible."
            self.success = False
            self.nit = 0
            self.x = np.array([np.nan, np.nan])

    mock_result = MockSciPyResult()
    try:
        raise LPFailureError.from_scipy_result(
            mock_result, context={"iteration": 125}
        )
    except LPFailureError as e:
        print(f"   Caught LP Error: {e}")
        lp_dict = e.to_dict()
        print(f"   Serialized LP dict: {json.dumps(lp_dict, indent=2)}")
        assert lp_dict["context"]["lp_status"] == 2
        # Note: numpy.nan is sanitized to None by our function
        assert lp_dict["context"]["lp_diagnostics"]["x"] == [None, None]
        assert lp_dict["context"]["iteration"] == 125

    # 4. Logging Helper Method
    print("\n4. Testing the .log() helper method:")
    # Mock logger
    class MockLogger(logging.Logger):
        def __init__(self, name):
            super().__init__(name)
            self.last_log = None
        def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
            self.last_log = {"level": level, "msg": msg, "extra": extra}
            print(f"   Logger received: level={level}, msg='{msg}', extra={extra}")

    mock_logger = MockLogger("test_logger")
    try:
        raise QKDSimulationError("Matrix is not positive semi-definite.")
    except QKDException as e:
        e.log(mock_logger, level=logging.WARNING)
        assert mock_logger.last_log["level"] == logging.WARNING
        assert mock_logger.last_log["extra"]["qkd_error"]["code"] == "ERR_SIMULATION"

    # 5. Picklability for Multiprocessing
    print("\n5. Testing picklability for multiprocessing:")
    exc_to_pickle = ParameterValidationError(
        "Invalid wavelength", param_name="lambda_nm", param_value=1551.5
    )
    
    try:
        pickled_exc = pickle.dumps(exc_to_pickle)
        unpickled_exc = pickle.loads(pickled_exc)
        
        print(f"   Original:    {exc_to_pickle}")
        print(f"   Unpickled:   {unpickled_exc}")
        
        assert str(exc_to_pickle) == str(unpickled_exc)
        assert exc_to_pickle.to_dict() == unpickled_exc.to_dict()
        print("   ✅ Pickling and unpickling successful.")
    except Exception as e:
        print(f"   ❌ Pickling failed: {e}")

    print("\n--- All demonstrations complete. ---")
