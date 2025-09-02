# -*- coding: utf-8 -*-
"""
Production-Grade Command-Line Interface for the QKD Simulation Framework.

This module provides a robust, reproducible, and user-friendly CLI for running
finite-key QKD simulations. It has been significantly refactored to address
shortcomings in the original script, incorporating best practices for scientific
software, including structured logging, graceful error handling, explicit
reproducibility controls, and a safer user interface.

This CLI requires Python 3.8+ and expects to interact with a `QKDSystem`
object that has a `run_simulation()` method and returns a results object. The
results object is expected to have the following attributes and methods:
- `.to_serializable_dict()`: Returns a JSON-serializable dictionary.
- `.status`: A string indicating the simulation outcome (e.g., "OK").
- `.simulation_time_seconds`: A float representing the run time.
- `.raw_sifted_key_length`: A number representing the sifted key length.
- `.secure_key_length`: A number representing the final secure key length.

Refactor based on expert review, addressing a comprehensive checklist of
improvements across correctness, usability, security, and maintainability.
"""
import argparse
import json
import logging
import multiprocessing
import os
import platform
import secrets
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import enum
# numpy optional handling
try:
    import numpy as _np
except Exception:
    _np = None


# Assuming these are part of your framework's public API.
# If they are not, their definitions would need to be included or adjusted.
from .params import QKDParams
from .simulation import QKDSystem
from .io import save_results_json
from .exceptions import ParameterValidationError, QKDSimulationError
from .constants import LP_SOLVER_METHODS
from .datatypes import SimulationResults

__all__ = ["main"]
__version__ = "2.4.0"  # Final hardened version

# --- Constants for Exit Codes, Metadata, and Configuration ---

class ExitCode:
    """Semantic exit codes for the CLI, based on sysexits.h."""
    OK = 0
    GENERAL_ERROR = 1
    USAGE_ERROR = 64
    DATA_ERROR = 65
    UNAVAILABLE_ERROR = 69
    INTERNAL_ERROR = 70
    SIMULATION_FAILURE = 71
    USER_INTERRUPT = 130

class MetadataKeys:
    """Keys for reproducibility and run-tracking metadata."""
    RUN_ID = "qkd_run_id"
    GIT_COMMIT = "qkd_git_commit"
    CLI_ARGS = "cli_args"
    MASTER_SEED = "master_seed"
    VERSION = "qkd_framework_version"
    PYTHON_VERSION = "python_version"
    PYTHON_EXECUTABLE = "python_executable"

# --- Global State for Graceful Termination ---
TERMINATION_REQUESTED = threading.Event()
logger = logging.getLogger(__name__)

# --- Core Logic Functions (for testability) ---

def setup_logging(verbosity: str, log_file: Optional[Path] = None) -> None:
    """Configures structured and consistent logging for the application."""
    level = getattr(logging, verbosity.upper(), logging.INFO)
    log_format = "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except (IOError, OSError) as e:
            logger.error(f"Could not open log file '{log_file}': {e}")

def setup_multiprocessing_start_method() -> None:
    """Safely sets the multiprocessing start method to 'spawn' for reproducibility."""
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != "spawn":
        logger.debug(f"Attempting to change multiprocessing start method from '{current_method}' to 'spawn'.")
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            logger.warning(
                "Could not set multiprocessing start method to 'spawn'. It may have been "
                "set by another library. For guaranteed reproducibility, ensure "
                "this application is the first to configure multiprocessing."
            )

def handle_termination_signal(signum: int, frame: Any) -> None:
    """Signal handler to request a graceful shutdown."""
    if not TERMINATION_REQUESTED.is_set():
        # Use signal.Signals for Python 3.8+ compatibility
        signal_name = signal.Signals(signum).name
        logger.warning(
            f"Termination signal ({signal_name}) received. "
            "Requesting graceful shutdown. The simulation will stop after the "
            "current batch. Press Ctrl-C again to force exit."
        )
        TERMINATION_REQUESTED.set()
    else:
        logger.critical("Forcing exit due to repeated termination signal.")
        sys.exit(ExitCode.USER_INTERRUPT)

def create_cli_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Production-Grade Finite-Key QKD Simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("param_file", type=str, help="Path to the JSON parameters file. Use '-' for stdin.")
    parser.add_argument("-o", "--output", type=Path, help="Path to save the results JSON file.")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to save detailed logs.")
    parser.add_argument("--seed", type=int, default=None, help="Master seed (non-negative integer).")
    parser.add_argument("--save-seed", action="store_true", help="Save the master seed in the output metadata. Requires --output.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes (>= 1).")
    parser.add_argument("--force-sequential", action="store_true", help="Force sequential execution.")
    parser.add_argument("--lp-solver-method", type=str, default=None, choices=LP_SOLVER_METHODS, help="Primary LP solver.")
    parser.add_argument("--allow-unsafe-mdi-approx", action="store_true", help="CRITICAL-RISK: Use unsafe MDI-QKD proof.")
    parser.add_argument("--verbosity", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Console logging verbosity.")
    parser.add_argument("--dry-run", action="store_true", help="Validate params and exit without running.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of output file and other safety checks.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--print-template", action="store_true", help="Print a template JSON and exit.")
    return parser

def get_parameter_template() -> str:
    """Returns a template JSON parameter file as a string."""
    template = {
        "protocol_name": "BB84", "num_pulses": 10**12,
        "num_workers": os.cpu_count() or 1, "lp_solver_method": "highs",
        "channel": {"distance_km": 50, "loss_db_per_km": 0.2},
        "detector": {"det_eff": 0.8, "dark_count_rate": 1e-8, "dead_time_ns": 50},
        "error_correction": {"efficiency": 1.1},
        "security": {"eps_sec": 1e-9, "eps_cor": 1e-15}
    }
    return json.dumps(template, indent=4)

def validate_cli_args(args: argparse.Namespace) -> None:
    """Performs early validation of CLI arguments, raising ValueError on failure."""
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be a non-negative integer.")
    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError("--num-workers must be an integer >= 1.")
    if args.save_seed and not args.output:
        raise ValueError("--save-seed requires an --output file to be specified.")
    if args.output:
        if args.output.exists() and args.output.is_dir():
            raise ValueError(f"Output path '{args.output}' is a directory; it must be a file path.")
        if not args.force and args.output.exists():
            raise ValueError(f"Output file '{args.output}' exists. Use --force to overwrite.")
        
        parent = args.output.parent
        if not parent.exists():
            if args.force:
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created output directory '{parent}' because --force was provided.")
                except OSError as e:
                    raise ValueError(f"Could not create output directory '{parent}': {e}")
            else:
                raise ValueError(f"Output directory '{parent}' does not exist. Please create it or use --force.")
        
        try:
            with tempfile.NamedTemporaryFile(dir=parent, prefix='qkd-cli-write-test-') as _:
                pass
        except (IOError, OSError) as e:
            raise ValueError(f"Output directory '{parent}' is not writable: {e}")

def load_and_validate_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads parameters from file, applies CLI overrides, and validates."""
    param_file_path = args.param_file.strip()
    logger.info(f"Loading parameters from: '{'stdin' if param_file_path == '-' else param_file_path}'")
    try:
        if param_file_path == '-':
            params_dict = json.load(sys.stdin)
        else:
            with open(param_file_path, "r", encoding="utf-8") as f:
                params_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise e

    if not isinstance(params_dict, dict):
        raise TypeError("Top-level JSON in parameter file must be an object/dictionary.")
    
    try:
        num_pulses = float(params_dict.get("num_pulses", 0))
    except (TypeError, ValueError):
        raise ValueError("'num_pulses' must be a numeric value.")

    max_pulses = 1e15
    if num_pulses > max_pulses and not args.force:
        raise ValueError(f"num_pulses ({num_pulses:.0e}) is very large. Use --force to proceed.")

    if args.num_workers is not None: params_dict["num_workers"] = args.num_workers
    if args.force_sequential: params_dict["force_sequential"] = True
    if args.lp_solver_method: params_dict["lp_solver_method"] = args.lp_solver_method
    if args.allow_unsafe_mdi_approx:
        logger.warning("UNSAFE MDI APPROXIMATION IS ENABLED. RESULTS ARE NOT VALID.")
        params_dict["allow_unsafe_mdi_approx"] = True
    
    cpu_count = os.cpu_count() or 1
    try:
        req_workers = int(params_dict.get("num_workers", 1))
        if req_workers < 1: raise ValueError
    except (TypeError, ValueError):
        raise ValueError("'num_workers' in parameter file must be an integer >= 1.")
    
    max_workers = cpu_count * 4
    if req_workers > max_workers:
        logger.warning(f"num_workers ({req_workers}) is very high; clamping to {max_workers}.")
        params_dict["num_workers"] = max_workers
    else:
        params_dict["num_workers"] = req_workers

    return params_dict

def perform_dry_run(params: QKDParams, param_file: str) -> None:
    """Logs derived parameters for a dry run."""
    logger.info("--- Dry Run: Parameter Validation and Derived Values ---")
    logger.info(f"Parameter file '{param_file}' loaded and validated successfully.")
    logger.info("\nDerived Parameters:")
    logger.info(f"  - Channel Transmittance: {params.channel.transmittance:.4e}")
    logger.info(f"  - Overall Detector Efficiency (incl. channel): {getattr(params.detector, 'det_eff', 0) * params.channel.transmittance:.4e}")
    logger.info("\nConfiguration appears valid. Exiting dry run.")
    
def _json_default(obj):
    """Convert common non-json types to JSON-serializable primitives."""
    # Enums -> name
    try:
        if isinstance(obj, enum.Enum):
            return obj.name
    except Exception:
        pass

    # numpy scalars / arrays
    try:
        if _np is not None:
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
    except Exception:
        pass

    # objects exposing .name (ConfidenceBoundMethod or similar)
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name

    # fallback to str() which is always serializable
    return str(obj)


def save_results_atomically(results: SimulationResults, output_path: Path, save_seed: bool, is_debug: bool) -> None:
    """Saves simulation results to a file atomically, with fallbacks and secure permissions."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output_path.parent, delete=False, suffix='.tmp') as tmp_file:
            tmp_path = Path(tmp_file.name)
            # Try preferred method, otherwise fall back to reasonable serialisation strategies\n            if hasattr(results, 'to_serializable_dict') and callable(results.to_serializable_dict):\n                data = results.to_serializable_dict()\n            else:\n                try:\n                    import dataclasses\n                    data = dataclasses.asdict(results)\n                except Exception:\n                    try:\n                        data = vars(results)\n                    except Exception:\n                        data = {'repr': repr(results)}
            try:
                # Ensure 'data' is defined in all code paths; try preferred serialization then sensible fallbacks
                if 'data' not in locals():
                    try:
                        if hasattr(results, 'to_serializable_dict') and callable(results.to_serializable_dict):
                            data = results.to_serializable_dict()
                        else:
                            import dataclasses
                            data = dataclasses.asdict(results)
                    except Exception:
                        try:
                            data = vars(results)
                        except Exception:
                            data = {'repr': repr(results)}

                json.dump(data, tmp_file, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
            except (TypeError, OverflowError):
                logger.debug("Default json.dump failed, falling back to framework's save_results_json", exc_info=is_debug)
                tmp_file.close()
                try:
                    save_results_json(results, str(tmp_path))
                except TypeError:
                    logger.debug("Retrying save_results_json with alternate signature.", exc_info=is_debug)
                    # This alternate signature is incorrect and has been corrected.
                    save_results_json(results, str(tmp_path))
        
        for attempt in range(3):
            try:
                os.replace(str(tmp_path), str(output_path))
                logger.info(f"Successfully saved results to '{output_path}'")
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    raise
        
        if save_seed:
            try:
                if os.name == 'posix':
                    logger.debug("Setting file permissions to 0o600 for security.")
                    os.chmod(output_path, 0o600)
            except OSError as e:
                logger.warning(f"Could not set secure file permissions on '{output_path}': {e}")

    except Exception:
        logger.exception(f"Failed to save results to '{output_path}'")
        if tmp_path and tmp_path.exists():
            for _ in range(3):
                try:
                    tmp_path.unlink(missing_ok=True)
                    break
                except OSError:
                    time.sleep(0.1)
        raise

def _watch_termination(sys_obj: QKDSystem):
    """Watcher thread to attempt a polite shutdown on the QKD system."""
    TERMINATION_REQUESTED.wait(timeout=30.0) # Wait up to 30s
    if not TERMINATION_REQUESTED.is_set(): return # Timed out, do nothing

    logger.info("Termination requested: attempting polite shutdown on system object.")
    for method_name in ("request_terminate", "shutdown", "terminate", "stop"):
        if hasattr(sys_obj, method_name):
            try:
                getattr(sys_obj, method_name)()
                logger.info(f"Called {method_name}() on QKDSystem for graceful shutdown.")
                return
            except Exception:
                logger.debug(f"Attempt to call {method_name}() failed.", exc_info=True)

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the simulation CLI."""
    if sys.version_info < (3, 8):
        sys.stderr.write("Error: This script requires Python 3.8 or newer.\n")
        return ExitCode.GENERAL_ERROR

    logging.basicConfig(level=logging.WARNING)
    setup_multiprocessing_start_method()
    signal.signal(signal.SIGINT, handle_termination_signal)
    try:
        signal.signal(signal.SIGTERM, handle_termination_signal)
    except (AttributeError, ValueError):
        logger.debug("SIGTERM handler could not be registered on this platform.")

    parser = create_cli_parser()
    args = parser.parse_args(argv)
    cli_args_str = " ".join(argv if argv is not None else sys.argv[1:])

    if args.print_template:
        print(get_parameter_template())
        return ExitCode.OK

    setup_logging(args.verbosity, args.log_file)
    is_debug = (args.verbosity == "DEBUG")
    exit_code = ExitCode.GENERAL_ERROR

    try:
        validate_cli_args(args)
        params_dict = load_and_validate_params(args)
        params = QKDParams.from_dict(params_dict)

        if args.dry_run:
            perform_dry_run(params, args.param_file)
            return ExitCode.OK

        master_seed = args.seed if args.seed is not None else secrets.randbits(63)
        if args.seed is None:
            logger.debug(f"Generated random master seed: {master_seed}")
        
        run_metadata: Dict[str, Any] = {
            key: val for key, val in {
                MetadataKeys.RUN_ID: os.getenv("QKD_RUN_ID"),
                MetadataKeys.GIT_COMMIT: os.getenv("QKD_GIT_COMMIT"),
                MetadataKeys.CLI_ARGS: cli_args_str,
                MetadataKeys.VERSION: __version__,
                MetadataKeys.PYTHON_VERSION: platform.python_version(),
                MetadataKeys.PYTHON_EXECUTABLE: sys.executable,
            }.items() if val is not None
        }
        if args.save_seed:
            run_metadata[MetadataKeys.MASTER_SEED] = master_seed
            logger.info(f"Using master seed: {master_seed} (will be saved in output)")

        logger.info(f"Starting QKD simulation with parameters from: {args.param_file}")
        
        system = QKDSystem(params, seed=master_seed, save_master_seed=args.save_seed)
        if hasattr(system, "set_run_metadata"):
            system.set_run_metadata(run_metadata) # type: ignore[attr-defined]
        if hasattr(system, "set_termination_event"):
            system.set_termination_event(TERMINATION_REQUESTED) # type: ignore[attr-defined]

        threading.Thread(target=_watch_termination, args=(system,), daemon=True).start()

        results = system.run_simulation()
        status = getattr(results, "status", "UNKNOWN")
        logger.info(f"Simulation finished in {getattr(results, 'simulation_time_seconds', -1.0):.2f}s. Status: {status}")

        if status == "OK":
            logger.info(f"Raw Sifted Key Length: {getattr(results, 'raw_sifted_key_length', 'N/A')}")
            logger.info(f"Final Secure Key Length: {getattr(results, 'secure_key_length', 'N/A')}")
            exit_code = ExitCode.OK
        elif status == "INTERRUPTED":
            exit_code = ExitCode.USER_INTERRUPT
        else:
            exit_code = ExitCode.SIMULATION_FAILURE

        if args.output:
            save_results_atomically(results, args.output, args.save_seed, is_debug)
        else:
            try:
                results_json = json.dumps(results.to_serializable_dict(), indent=2, sort_keys=True, ensure_ascii=False) # type: ignore[attr-defined]
            except (TypeError, OverflowError):
                logger.debug("json.dumps failed for stdout; falling back to framework saver", exc_info=is_debug)
                with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    save_results_json(results, str(tmp_path))
                except TypeError:
                    # This alternate signature is incorrect and has been corrected.
                    save_results_json(results, str(tmp_path))
                results_json = tmp_path.read_text(encoding='utf-8')
                tmp_path.unlink(missing_ok=True)
            
            max_stdout_len = 1_000_000
            if len(results_json) > max_stdout_len and not args.force:
                if not sys.stdout.isatty():
                    logger.error(f"Refusing to write {len(results_json)} chars to non-interactive stdout. Use --output or --force.")
                    return ExitCode.USAGE_ERROR
                else:
                    logger.warning(f"Result JSON is very large ({len(results_json)} chars). Consider using --output.")
            print(results_json)

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid argument or configuration: {e}", exc_info=is_debug)
        exit_code = ExitCode.USAGE_ERROR
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading parameter file: {e}", exc_info=is_debug)
        exit_code = ExitCode.DATA_ERROR
    except ParameterValidationError as e:
        logger.error(f"Parameter validation failed: {e}", exc_info=is_debug)
        exit_code = ExitCode.DATA_ERROR
    except QKDSimulationError as e:
        logger.error(f"Simulation runtime error: {e}", exc_info=is_debug)
        exit_code = ExitCode.SIMULATION_FAILURE
    except KeyboardInterrupt:
        logger.warning("\nCaught KeyboardInterrupt. Forcing exit.")
        exit_code = ExitCode.USER_INTERRUPT
    except Exception:
        logger.exception("An unexpected internal error occurred.")
        exit_code = ExitCode.INTERNAL_ERROR
    finally:
        logger.info(f"Exiting with code {exit_code}.")
        return exit_code

if __name__ == "__main__":
    sys.exit(main())

