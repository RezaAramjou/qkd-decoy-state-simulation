# -*- coding: utf-8 -*-
"""
Input/Output and serialization helper functions.

This module provides a robust, atomic, and secure function for saving
simulation results to a JSON file. It incorporates best practices for
data durability, error handling, and portability.
"""
import os
import json
import tempfile
import logging
import math
import gzip
import io
import hashlib
from typing import Any, Dict, Union, TYPE_CHECKING

# CHANGE: Replaced placeholder with actual import from your datatypes.py file.
# This makes the connection between the modules explicit.
if TYPE_CHECKING:
    from .datatypes import SimulationResults

logger = logging.getLogger(__name__)

__all__ = ["save_results_json"]


def _safe_serialize(obj: Any) -> Any:
    """
    Recursively convert an object into JSON-serializable primitives.

    This helper handles common non-serializable types like numpy scalars and
    arrays, and looks for canonical serialization methods on custom objects.
    It converts NaN and Infinity to `None` (JSON null) for RFC compliance.

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable representation of the object.
    """
    # Use canonical serializer if available
    if hasattr(obj, "to_serializable_dict"):
        return _safe_serialize(obj.to_serializable_dict())
    if hasattr(obj, "to_dict"):
        return _safe_serialize(obj.to_dict())

    # Handle numpy types if numpy is installed
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        # CRITICAL FIX: Recursively sanitize elements after converting array to list.
        if isinstance(obj, np.ndarray):
            return _safe_serialize(obj.tolist())
    except ImportError:
        pass  # numpy is not available

    # Handle basic types and collections
    if isinstance(obj, (str, bool, int, type(None))):
        return obj
    
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # Convert to JSON null
        return obj

    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_safe_serialize(x) for x in obj]

    # Fallback for any other type
    return str(obj)


def save_results_json(
    # CHANGE: Updated type hint from 'Any' to the specific 'SimulationResults' class.
    # This improves code clarity and enables static analysis tools to catch errors.
    results: "SimulationResults",
    path: str,
    *,
    overwrite: bool = True,
    compress: bool = False,
    sort_keys: bool = True,
    return_metadata: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Atomically and securely writes a results object to a JSON file.

    This function implements a robust save operation by:
    1.  Using `tempfile.mkstemp` for atomic temporary file creation with secure permissions.
    2.  Writing to a temporary file in the same directory as the target path.
    3.  Correctly handling gzip compression with binary file objects and proper closing order.
    4.  Ensuring both file and directory metadata are synced to disk for durability on POSIX.
    5.  Cleaning up the temporary file on any error and re-raising the exception.
    6.  Recursively serializing non-JSON-native types (e.g., numpy, NaN) safely.
    7.  Offering optional gzip compression, a no-overwrite policy, and metadata return.

    Args:
        results: The SimulationResults object to serialize and save.
        path: The final destination file path.
        overwrite: If False, raises FileExistsError if the destination path already exists.
        compress: If True, saves the file with gzip compression (and a .gz extension).
        sort_keys: If True, sorts dictionary keys for deterministic output.
        return_metadata: If True, returns a dictionary with path, size, and SHA256 hash.

    Returns:
        The absolute path to the saved file, or a dictionary with file metadata.

    Raises:
        ValueError: If the target path points to an existing directory.
        FileExistsError: If `overwrite` is False and the path exists.
        Exception: Propagates underlying exceptions after performing cleanup.

    Note on Concurrency (TOCTOU):
        When `overwrite=False`, a race condition (Time-of-check to time-of-use) exists.
        Another process could create the file between the existence check and the final
        `os.replace`. For strict non-overwrite guarantees, application-level locking
        is recommended.
    """
    full_path = os.path.abspath(path)
    
    if compress and not full_path.endswith(".gz"):
        full_path += ".gz"

    if not overwrite and (os.path.exists(full_path) or os.path.islink(full_path)):
        raise FileExistsError(f"Destination path exists and overwrite is False: {full_path}")

    if os.path.isdir(full_path):
        raise ValueError(f"Target path is a directory, not a file: {full_path}")

    dir_path = os.path.dirname(full_path) or os.getcwd()
    os.makedirs(dir_path, exist_ok=True)

    # IMPORTANT FIX: Choose suffix based on compression for clarity.
    suffix = ".json.gz.tmp" if compress else ".json.tmp"
    
    tmp_fd = -1
    tmp_path = None
    try:
        payload = _safe_serialize(results)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".", suffix=suffix, dir=dir_path)

        if os.name == "posix":
            try:
                os.fchmod(tmp_fd, 0o600)
            except OSError:
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    logger.debug("Failed to set secure permissions on temp file", exc_info=True)

        # CRITICAL FIX: Correctly handle gzip writing in binary mode.
        if compress:
            # Open fd as binary and wrap gzip & text layers.
            with os.fdopen(tmp_fd, "wb") as binary_f:
                tmp_fd = -1  # Ownership transferred to context manager.
                with gzip.GzipFile(fileobj=binary_f, mode="wb") as gz_f:
                    with io.TextIOWrapper(gz_f, encoding="utf-8") as text_writer:
                        json.dump(payload, text_writer, indent=4, ensure_ascii=False, sort_keys=sort_keys)
                # Context managers ensure gz_f and text_writer are closed and flushed.
                # Sync the underlying binary file descriptor after gzip footer is written.
                os.fsync(binary_f.fileno())
        else:
            # Non-compressed: open as text and write.
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                tmp_fd = -1
                json.dump(payload, f, indent=4, ensure_ascii=False, sort_keys=sort_keys)
                f.flush()
                os.fsync(f.fileno())

        os.replace(tmp_path, full_path)
        tmp_path = None  # Mark as moved successfully.

        if os.name == "posix":
            dir_fd = -1
            try:
                dir_fd = os.open(dir_path, os.O_RDONLY)
                os.fsync(dir_fd)
            except OSError as e:
                logger.warning(f"Could not fsync directory {dir_path}: {e}")
            finally:
                if dir_fd != -1:
                    os.close(dir_fd)

        file_size_bytes = os.path.getsize(full_path)
        logger.info(f"Results saved to file: {full_path} ({file_size_bytes / 1024:.2f} KB)")
        
        if return_metadata:
            sha256 = hashlib.sha256()
            with open(full_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return {
                "path": full_path,
                "size_bytes": file_size_bytes,
                "sha256": sha256.hexdigest(),
            }
        return full_path

    except Exception:
        logger.error(f"Failed to save results to {path}", exc_info=True)
        raise
    finally:
        # Guaranteed cleanup for all failure modes.
        if tmp_fd != -1:
            os.close(tmp_fd)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {tmp_path} during cleanup: {e}")
