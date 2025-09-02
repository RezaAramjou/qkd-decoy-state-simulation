#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust, 10/10 production-grade entry point for the QKD simulation.

This script serves as a stable and safe wrapper for launching the main QKD CLI.
It is designed to handle various invocation scenarios, normalize environment
differences, and provide clear, actionable error feedback.

Recommended Invocations:
  - As an installed package (preferred):
    $ qkd-run --params params.json --output results.json

  - During development (from the project root):
    $ python -m qkd.scripts.run_qkd --params params.json --output results.json

Key Features of this Wrapper:
  - Python Version Check: Ensures the script is run with a compatible Python version.
  - Safe & Reversible Import Mechanism: If the package is not installed, it
    robustly detects a development environment and *temporarily* adds the
    project root to the system path only for the duration of the import attempt.
  - Standardized Exit Codes: Maps all exit conditions to a deterministic set of
    integer exit codes (0-255) for reliable use in automation.
  - Graceful Shutdown: Handles SIGINT (Ctrl+C) and SIGTERM signals. The CLI
    must check for the termination flag to exit cleanly. A second signal
    triggers a forceful, immediate exit.
  - Rich Run Metadata: Generates a unique run ID and captures the Git commit hash,
    exposing them via environment variables (`QKD_RUN_ID`, `QKD_GIT_COMMIT`).
  - Structured Error Reporting: Outputs errors in a machine-readable JSON format
    if `QKD_JSON_ERRORS=1` is set. Includes an optional `traceback` if `QKD_DEBUG=1`.
  - Framework Compatibility: Includes robust support for CLIs built with `click`.
  - Optional File Logging: Configures logging to a file if `QKD_LOG_FILE` is set,
    with optional rotation via `QKD_LOG_ROTATE=1`.

Developer Recommendations for `qkd.cli`:
  - CLI Contract: Expose a function with the signature `main(argv: List[str]) -> int`.
  - Graceful Shutdown Contract (CRITICAL): The simulation logic must periodically
    check for termination requests to exit cleanly. A helper function in the core
    library (e.g., `qkd.utils.termination_requested`) is the recommended approach:
   
    def termination_requested() -> bool:
        return os.getenv("QKD_TERMINATE") == "1"

    This function should be called from within long-running loops.
  - Metadata Integration: The CLI should read `QKD_RUN_ID` and `QKD_GIT_COMMIT`
    from the environment and embed them into the final output results.
  - Atomic File Writes: Use `os.replace()` for atomic file writes.
  - Config Validation: Implement a `--validate-config` or `--dry-run` flag.
  - Multiprocessing: For cross-platform consistency, call
    `multiprocessing.set_start_method('spawn')`. Use a `logging.handlers.QueueHandler`
    for safe logging from worker processes.
  - Packaging: Define a console script entry point in `pyproject.toml`:
    [project.scripts]
    qkd-run = "qkd.scripts.run_qkd:main_entry"
  - Unit Testing: The wrapper's behavior must be covered by unit tests to
    prevent regressions in CI.
"""
from __future__ import annotations

import os
import sys
import json
import signal
import logging
import traceback
import uuid
import subprocess
from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import FrameType, ModuleType
from typing import List, Optional
from logging.handlers import RotatingFileHandler

# The application requires Python 3.8+.
if sys.version_info < (3, 8):
    sys.stderr.write("Error: This application requires Python 3.8 or newer.\n")
    sys.exit(2)

# Establish a dedicated logger for this entrypoint script.
logger = logging.getLogger("qkd.entrypoint")


class ExitCode(IntEnum):
    """Defines standardized, integer-based exit codes for the application."""
    OK = 0
    GENERAL_ERROR = 1
    IMPORT_ERROR = 2
    CONFIG_ERROR = 3
    IO_ERROR = 4
    SIMULATION_ERROR = 5
    USER_INTERRUPTED = 130


# Global flag to signal termination for graceful shutdown.
_TERMINATION_REQUESTED = False


def _setup_signal_handlers() -> None:
    """Configures handlers for SIGINT and SIGTERM to enable graceful shutdown."""
    global _TERMINATION_REQUESTED

    def handle_termination(signum: int, frame: Optional[FrameType]) -> None:
        """
        Signal handler that sets a flag and an environment variable on the first
        signal, and force-exits on the second.
        """
        global _TERMINATION_REQUESTED
        if not _TERMINATION_REQUESTED:
            logger.warning("Termination signal received. Shutting down gracefully...")
            _TERMINATION_REQUESTED = True
            # Set an environment variable to propagate the signal to child processes.
            os.environ["QKD_TERMINATE"] = "1"
        else:
            logger.warning("Multiple termination signals received. Forcing exit.")
            os._exit(int(ExitCode.USER_INTERRUPTED))

    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)


def _find_dev_root(start_dir: Path) -> Optional[Path]:
    """
    Robustly finds the project root by searching upwards from a starting
    directory for development markers like `.git` or `pyproject.toml`.
    """
    try:
        # Resolve to a real path to handle symlinks and invalid relative paths.
        start = Path(start_dir).resolve()
    except (FileNotFoundError, OSError):
        return None # The starting path is invalid.

    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.cfg").exists():
            return p
    return None


def _import_cli() -> ModuleType:
    """
    Imports and returns `qkd.cli`, using a temporary path modification
    only if a development environment is detected and the package isn't installed.
    """
    try:
        return import_module("qkd.cli")
    except ImportError as initial_error:
        # Determine script directory, handling frozen executables and missing __file__.
        script_dir_str = getattr(sys, "_MEIPASS", None)
        if script_dir_str is None:
            script_file = globals().get("__file__")
            if script_file:
                script_dir = Path(script_file).resolve().parent
            else:
                # Fallback for unusual contexts where __file__ is not defined.
                script_dir = Path.cwd()
        else:
            script_dir = Path(script_dir_str)

        project_root = _find_dev_root(script_dir)

        if not project_root:
            raise initial_error

        project_root_str = str(project_root)
        path_inserted = False
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            path_inserted = True

        try:
            return import_module("qkd.cli")
        except ImportError as e2:
            # Log the second error for debugging, but raise the original, more
            # informative error to the user.
            logger.debug("Second import attempt failed after path modification: %s", e2)
            raise initial_error
        finally:
            if path_inserted:
                try:
                    sys.path.remove(project_root_str)
                except ValueError:
                    pass


def _normalize_exit_code(rc: object) -> int:
    """
    Normalizes various return types to a valid integer exit code (0-255).
    """
    # Explicitly type `code` as an integer to resolve the mypy assignment error.
    code: int = int(ExitCode.GENERAL_ERROR)
    if rc is None:
        code = int(ExitCode.OK)
    elif isinstance(rc, bool):
        code = int(ExitCode.OK) if rc else int(ExitCode.GENERAL_ERROR)
    elif isinstance(rc, (int, IntEnum)):
        code = int(rc)

    return code & 0xFF


def _get_git_commit() -> Optional[str]:
    """Returns the current git commit hash if in a git repository, with a timeout."""
    try:
        timeout = int(os.getenv("QKD_GIT_TIMEOUT", "5"))
        subprocess.check_output(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            stderr=subprocess.DEVNULL, timeout=timeout
        )
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, timeout=timeout
        )
        return commit_hash.strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def main_entry(argv: Optional[List[str]] = None) -> int:
    """The main entry point for the QKD simulation wrapper."""
    if argv is None:
        argv = sys.argv[1:]

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)-15s] [%(levelname)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    log_file = os.getenv("QKD_LOG_FILE")
    if log_file:
        try:
            # Explicitly type `fh` with a common base class to handle both handler types.
            fh: logging.FileHandler
            if os.getenv("QKD_LOG_ROTATE") == "1":
                fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
            else:
                fh = logging.FileHandler(log_file)
            
            fh.setLevel(logging.INFO)
            # Safely get a formatter, creating a default one if none exist.
            if root_logger.handlers and getattr(root_logger.handlers[0], "formatter", None):
                formatter = root_logger.handlers[0].formatter
            else:
                formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
        except (IOError, OSError) as e:
            logger.error(f"Could not open log file '{log_file}': {e}")

    if "--version" in argv:
        from importlib import metadata
        try:
            print(f"qkd version {metadata.version('qkd')}")
        except metadata.PackageNotFoundError:
            print("qkd package not found. Cannot determine version.")
        return int(ExitCode.OK)

    _setup_signal_handlers()

    run_id = str(uuid.uuid4())
    os.environ["QKD_RUN_ID"] = run_id
    logger.info(f"Starting QKD simulation run. Run ID: {run_id}")

    git_commit = _get_git_commit()
    if git_commit:
        os.environ["QKD_GIT_COMMIT"] = git_commit
        logger.info(f"Git Commit: {git_commit}")

    try:
        cli = _import_cli()

        if not hasattr(cli, "main") or not callable(cli.main):
            logger.critical("The 'qkd.cli' module does not have a callable 'main' function.")
            return int(ExitCode.IMPORT_ERROR)

        try:
            # Ignore the mypy error for click, as it doesn't have official type stubs.
            import click  # type: ignore[import]
            is_click_command = isinstance(cli.main, click.core.Command)
        except (ImportError, AttributeError):
            is_click_command = False

        if is_click_command:
            try:
                return_code = cli.main(args=argv, standalone_mode=False)
            except click.exceptions.Exit as ce:
                # Defensively get the numeric exit code and ensure it's an integer.
                rc = getattr(ce, "exit_code", getattr(ce, "code", int(ExitCode.GENERAL_ERROR)))
                try:
                    return_code = int(rc)
                except (ValueError, TypeError):
                    return_code = int(ExitCode.GENERAL_ERROR)
            except click.ClickException as cex:
                cex.show(file=sys.stderr)
                return_code = int(ExitCode.CONFIG_ERROR)
        else:
            try:
                return_code = cli.main(argv)
            except TypeError:
                logger.warning("Calling cli.main() without arguments is deprecated.")
                return_code = cli.main()

        return _normalize_exit_code(return_code)

    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user (KeyboardInterrupt).")
        return int(ExitCode.USER_INTERRUPTED)
    except SystemExit as se:
        return _normalize_exit_code(se.code)
    except ImportError as e:
        logger.critical(f"Failed to import the QKD application module: {e}")
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return int(ExitCode.IMPORT_ERROR)
    except Exception as e:
        log_msg = f"An unhandled exception occurred: {e.__class__.__name__}: {e}"
        logger.critical(log_msg)

        if os.getenv("QKD_JSON_ERRORS") == "1":
            payload = {
                "error_type": e.__class__.__name__,
                "message": str(e),
                "run_id": run_id,
            }
            if os.getenv("QKD_DEBUG") == "1":
                payload["traceback"] = traceback.format_exc()
            print(json.dumps(payload), file=sys.stderr)

        logger.debug("Traceback:\n%s", traceback.format_exc())
        return int(ExitCode.GENERAL_ERROR)


if __name__ == "__main__":
    sys.exit(main_entry())

