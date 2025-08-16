# -*- coding: utf-8 -*-
"""
High-level simulation orchestration for Quantum Key Distribution (QKD) systems.

This script orchestrates a full QKD simulation, including parallel batch
processing using a robust, memory-safe streaming submission pattern. It is
designed for reproducibility and resilience, with graceful shutdown mechanisms
and detailed result validation. It is recommended to run with Python >= 3.9
for the most robust process shutdown capabilities.

Expected Worker Contract:
The worker function (e.g., _top_level_worker_function) must accept a parameter
dictionary, a number of pulses (int), and a seed (int). If the parameter
dictionary contains a 'params_file' key, the worker is responsible for loading
the parameters from that file. The worker should periodically check for a
global termination event set by the initializer and exit cleanly if it is set.
The worker must return a dictionary containing at least 'tallies' (dict),
'sifted_count' (int), and 'num_pulses' (int).
"""

# Standard library imports
import datetime
import gc
import inspect
import logging
import multiprocessing
import os
import pickle
import secrets
import signal
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool as BrokenProcessPoolError
from typing import Dict, Optional, Any, Generator, Tuple, List

# Third-party imports
import numpy as np

# A fallback for the tqdm progress bar if it's not installed.
try:
    from tqdm import tqdm
except ImportError:
    class TqdmFallback:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get("total") # Can be None
            self.current = 0
        def __iter__(self): return iter(self.iterable)
        def __len__(self):
            if self.total is None: raise TypeError("object of type 'TqdmFallback' has no len()")
            return self.total
        def update(self, n=1): self.current += n
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    tqdm = TqdmFallback

# Local application/library specific imports
from .params import QKDParams
from .datatypes import SimulationResults, TallyCounts, SecurityCertificate, SecurityProof
from .exceptions import LPFailureError, ParameterValidationError, QKDSimulationError
from .simulation_batch import _top_level_worker_function, _merge_batch_tallies
from .constants import MAX_SEED_INT
from .proofs import Lim2014Proof, BB84TightProof, MDIQKDProof

__all__ = ["QKDSystem"]

logger = logging.getLogger(__name__)

# --- Module-level constants and setup ---
SIMULATION_VERSION = "v11.0-production-final"
METADATA_SCHEMA_VERSION = "1.8"
DEFAULT_PARAMS_PICKLE_THRESHOLD_BYTES = 1_000_000

def _worker_init(terminate_event):
    """Initializer for worker processes to tune environment and handle signals."""
    global _qkd_terminate_event
    _qkd_terminate_event = terminate_event
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

class QKDSystem:
    """
    Orchestrates a full QKD simulation, including parallel batch processing
    and post-processing analysis.
    """

    def __init__(self, params: QKDParams, seed: Optional[int] = None, save_master_seed: bool = False):
        """Initializes the QKD system with a given set of parameters."""
        self.p = params
        self.save_master_seed = save_master_seed

        if not isinstance(MAX_SEED_INT, int) or MAX_SEED_INT < 2:
            raise ParameterValidationError(f"MAX_SEED_INT must be an integer >= 2, but got {MAX_SEED_INT}")

        if seed is not None:
            try:
                seed_val = int(seed)
                if not (1 <= seed_val <= MAX_SEED_INT):
                    raise ValueError(f"Seed must be an integer between 1 and {MAX_SEED_INT}.")
                self.master_seed_int = seed_val
            except (ValueError, TypeError) as e:
                raise ParameterValidationError(f"Invalid seed provided: {seed}.") from e
        else:
            max_safe_seed = min(MAX_SEED_INT, np.iinfo(np.int64).max)
            self.master_seed_int = secrets.randbelow(max_safe_seed) + 1

        self.rng = np.random.default_rng(self.master_seed_int)

        try:
            proof_map = {
                SecurityProof.LIM_2014: Lim2014Proof,
                SecurityProof.TIGHT_PROOF: BB84TightProof,
                SecurityProof.MDI_QKD: MDIQKDProof,
            }
            proof_class = proof_map.get(self.p.security_proof)
            if proof_class:
                worker_func = getattr(self.p, '_test_worker_func', _top_level_worker_function)
                if not callable(worker_func):
                    raise ParameterValidationError("_test_worker_func must be callable.")
                self._worker_function = worker_func
                self.proof_module = proof_class(self.p)
            else:
                raise NotImplementedError(f"Security proof {self.p.security_proof.value} not implemented.")
        except Exception as e:
            raise ParameterValidationError(
                f"Failed to initialize security proof module '{self.p.security_proof.value}'"
            ) from e

    def __repr__(self) -> str:
        """Provide a safer, more concise representation."""
        params_repr = f"QKDParams(type={type(self.p).__name__})"
        seed_repr = "REDACTED" if not self.save_master_seed else self.master_seed_int
        return f"QKDSystem(params={params_repr}, master_seed={seed_repr})"

    def _prepare_task_params(self) -> dict:
        """Prepares the parameters for worker tasks, using a temp file for large objects."""
        params_dict = self.p.to_serializable_dict()
        pickle_threshold = getattr(self.p, 'params_pickle_threshold_bytes', DEFAULT_PARAMS_PICKLE_THRESHOLD_BYTES)
        
        try:
            pickled_params = pickle.dumps(params_dict, protocol=pickle.HIGHEST_PROTOCOL)
            if len(pickled_params) > pickle_threshold:
                logger.info(f"Serialized params are large ({len(pickled_params)} bytes), using temp file transport.")
                with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=".pkl") as tmp:
                    tmp.write(pickled_params)
                    return {"params_file": tmp.name, "delete_file": True}
        except (pickle.PicklingError, TypeError) as e:
            raise ParameterValidationError("Parameters cannot be pickled for worker processes.") from e

        return params_dict

    def _generate_tasks(self, num_batches: int, child_seeds: List[int], task_params: dict) -> Generator[Tuple[Dict, int, int], None, None]:
        """A generator that yields simulation tasks."""
        total_pulses, batch_size = self.p.num_bits, self.p.batch_size
        for i in range(num_batches):
            pulses_in_batch = min(batch_size, total_pulses - i * batch_size)
            yield (task_params, pulses_in_batch, child_seeds[i])

    def run_simulation(self) -> SimulationResults:
        """Executes the entire simulation."""
        # --- 1. Initialization and Validation ---
        start_time = time.monotonic()
        timing_meta = {"start_utc_iso": datetime.datetime.utcnow().isoformat() + "Z", "start_monotonic": start_time}
        run_id = getattr(self.p, 'run_id', 'N/A')
        
        if getattr(self.p, 'debug_mode', False) and not logger.handlers:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        log_seed = self.master_seed_int if self.save_master_seed else "REDACTED"
        logger.debug(f"Starting simulation run with master_seed={log_seed} (run_id: {run_id})")

        if not self.p.source.pulse_configs: raise ParameterValidationError("source.pulse_configs must not be empty.")
        total_pulses = getattr(self.p, 'num_bits', 0)
        batch_size = getattr(self.p, 'batch_size', 0)
        if total_pulses <= 0 or batch_size <= 0: raise ParameterValidationError("num_bits and batch_size must be positive integers.")

        num_batches = (total_pulses + batch_size - 1) // batch_size
        
        metadata: Dict[str, Any] = {
            "version": SIMULATION_VERSION, "schema_version": METADATA_SCHEMA_VERSION,
            "timing": timing_meta, "failures": {}, "run_id": run_id, 
            "batches_completed": 0, "num_batches": num_batches
        }

        use_deterministic_seeds = getattr(self.p, 'use_deterministic_child_seeds', False)
        if not use_deterministic_seeds and (num_batches > MAX_SEED_INT or MAX_SEED_INT > 10**9):
            if MAX_SEED_INT > 10**9: logger.warning(f"MAX_SEED_INT is very large ({MAX_SEED_INT}). Using deterministic child seeds for performance.")
            use_deterministic_seeds = True

        if use_deterministic_seeds:
            logger.info("Using deterministic arithmetic scheme for child seeds.")
            if num_batches > MAX_SEED_INT: logger.warning(f"num_batches ({num_batches}) > MAX_SEED_INT ({MAX_SEED_INT}), deterministic seeds will cycle.")
            child_seeds = [((self.master_seed_int + i -1) % MAX_SEED_INT) + 1 for i in range(num_batches)]
        else:
            child_seeds = self.rng.choice(MAX_SEED_INT, size=num_batches, replace=False) + 1
            child_seeds = child_seeds.tolist()

        overall_stats: Dict[str, TallyCounts] = {pc.name: TallyCounts() for pc in self.p.source.pulse_configs}
        total_sifted: int = 0
        status_info = {"code": "OK", "message": "Simulation completed successfully."}
        if self.save_master_seed: metadata["master_seed"] = self.master_seed_int

        # --- 2. Simulation Execution (MP or Sequential) ---
        executor = None
        params_file_path = None
        delete_temp_file = False
        terminate_event = None
        original_sigterm_handler = None

        try:
            use_mp = getattr(self.p, 'num_workers', 1) > 1 and num_batches > 1 and not getattr(self.p, 'force_sequential', False)
            if getattr(self.p, 'test_mode', False): use_mp = False
            
            mp_context = None
            if use_mp:
                start_method = multiprocessing.get_start_method(allow_none=True)
                if start_method is None:
                    try: multiprocessing.set_start_method("spawn", force=True)
                    except RuntimeError: logger.warning("Could not set multiprocessing start method to 'spawn'.")
                elif start_method != "spawn" and getattr(self.p, 'require_spawn', False):
                    logger.error("Spawn method required but not available. Falling back to sequential execution.")
                    use_mp = False
                
                if use_mp:
                    mp_context = multiprocessing.get_context("spawn")
                    terminate_event = mp_context.Event()
            
            task_params = self._prepare_task_params()
            delete_temp_file = task_params.get("delete_file", False)
            if "params_file" in task_params: params_file_path = task_params["params_file"]
            
            if hasattr(signal, 'SIGTERM') and not getattr(self.p, 'test_mode', False):
                original_sigterm_handler = signal.getsignal(signal.SIGTERM)
                def _handle_term_signal(signum, frame):
                    logger.warning(f"Received signal {signum}, initiating graceful shutdown.")
                    if terminate_event: terminate_event.set()
                signal.signal(signal.SIGTERM, _handle_term_signal)

            if use_mp:
                num_workers = max(1, min(int(getattr(self.p, 'num_workers', 1)), os.cpu_count() or 1))
                metadata["num_workers"] = num_workers
                
                executor_kwargs = {'max_workers': num_workers, 'initializer': _worker_init, 'initargs': (terminate_event,)}
                if 'mp_context' in inspect.signature(ProcessPoolExecutor).parameters:
                    executor_kwargs['mp_context'] = mp_context
                
                executor = ProcessPoolExecutor(**executor_kwargs)
                tasks = self._generate_tasks(num_batches, child_seeds, task_params)
                
                submitted = {}
                next_batch_idx = 0
                batches_completed = 0
                with tqdm(total=num_batches, desc="Simulating Batches (MP)") as pbar:
                    while batches_completed < num_batches:
                        if terminate_event and terminate_event.is_set():
                            status_info = {"code": "USER_ABORT", "message": "Graceful shutdown initiated."}
                            break
                        
                        window_size = getattr(self.p, 'submission_window_size', num_workers * 2)
                        while len(submitted) < window_size and next_batch_idx < num_batches:
                            try:
                                task_args = next(tasks)
                                fut = executor.submit(self._worker_function, *task_args)
                                submitted[fut] = {"seed": task_args[2], "batch_idx": next_batch_idx, 
                                                  "expected_pulses": task_args[1], "retries": 0, "task_params": task_args[0]}
                                next_batch_idx += 1
                            except StopIteration: break
                        
                        if not submitted: break

                        done, _ = wait(list(submitted.keys()), return_when=FIRST_COMPLETED, timeout=1.0)
                        for fut in done:
                            meta = submitted.pop(fut)
                            try:
                                worker_timeout = float(getattr(self.p, 'worker_timeout_seconds', 300))
                                batch_result = fut.result(timeout=worker_timeout)
                                self._validate_and_merge_batch(batch_result, overall_stats, meta['expected_pulses'])
                                total_sifted += int(batch_result.get("sifted_count", 0))
                                del batch_result
                                batches_completed += 1
                                pbar.update(1)
                            except (BrokenProcessPoolError, FutureTimeoutError) as e:
                                logger.error(f"Critical worker error for batch {meta['batch_idx']}: {type(e).__name__}")
                                status_info = {"code": "WORKER_ERROR", "message": f"Critical failure in batch {meta['batch_idx']}."}
                                if terminate_event: terminate_event.set()
                            except Exception as e:
                                max_retries = getattr(self.p, 'worker_retries', 0)
                                if meta['retries'] < max_retries:
                                    meta['retries'] += 1
                                    logger.warning(f"Worker for batch {meta['batch_idx']} failed, retrying ({meta['retries']}/{max_retries})...")
                                    time.sleep(min(2**meta['retries'], 30)) # Exponential backoff
                                    resubmit_args = (meta['task_params'], meta['expected_pulses'], meta['seed'])
                                    new_fut = executor.submit(self._worker_function, *resubmit_args)
                                    submitted[new_fut] = meta
                                else:
                                    logger.error(f"Worker for batch {meta['batch_idx']} failed after {meta['retries']} retries.", exc_info=True)
                                    status_info = {"code": "WORKER_ERROR", "message": f"Failure in batch {meta['batch_idx']}."}
                                    metadata.setdefault("failures", {}).setdefault("failed_batches", []).append({
                                        "batch_idx": meta['batch_idx'], "seed": meta['seed'], "error": type(e).__name__, 
                                        "traceback": "\n".join(traceback.format_exc().splitlines()[-20:])
                                    })
                                    if getattr(self.p, 'abort_on_worker_failure', True) and terminate_event: terminate_event.set()
                metadata["batches_completed"] = batches_completed
            else: # Sequential Execution
                tasks_generator = self._generate_tasks(num_batches, child_seeds, task_params)
                with tqdm(tasks_generator, total=num_batches, desc="Simulating Batches (Seq)") as seq_pbar:
                    for i, task in enumerate(seq_pbar):
                        try:
                            batch_result = self._worker_function(*task)
                            self._validate_and_merge_batch(batch_result, overall_stats, task[1])
                            total_sifted += int(batch_result.get("sifted_count", 0))
                            del batch_result
                            metadata["batches_completed"] = i + 1
                        except Exception as e:
                            logger.error(f"Sequential batch {i} (seed {task[2]}) failed.", exc_info=True)
                            status_info = {"code": "WORKER_ERROR", "message": f"Failure in sequential batch {i}."}
                            break
        except KeyboardInterrupt:
            status_info = {"code": "USER_ABORT", "message": "User interrupted the simulation."}
            if terminate_event: terminate_event.set()
        except Exception as e:
            status_info = {"code": "ORCHESTRATOR_ERROR", "message": str(e)}
            if terminate_event: terminate_event.set()
        finally:
            if executor:
                logger.debug("Shutting down process pool executor.")
                is_abort = terminate_event and terminate_event.is_set()
                if not is_abort:
                    executor.shutdown(wait=True)
                else:
                    try: executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError: executor.shutdown(wait=False)
            if params_file_path and delete_temp_file and os.path.exists(params_file_path):
                try: os.remove(params_file_path)
                except OSError as e: logger.warning(f"Could not remove temp params file {params_file_path}: {e}")
            if original_sigterm_handler is not None and hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, original_sigterm_handler)

        # --- 3. Post-Processing and Result Finalization ---
        timing_meta["end_monotonic"] = time.monotonic()
        timing_meta["simulation_seconds"] = timing_meta["end_monotonic"] - start_time
        
        if status_info["code"] == "OK" and metadata["batches_completed"] < num_batches:
            status_info = {"code": "INCOMPLETE", "message": "Simulation did not complete all batches."}

        if status_info["code"] != "OK":
            return SimulationResults(params=self.p, metadata=metadata, status=status_info["code"], simulation_time_seconds=timing_meta["simulation_seconds"])
        
        try: self._validate_final_tallies(overall_stats, total_sifted)
        except QKDSimulationError as e:
            status_info = {"code": "INCONSISTENT_RESULTS", "message": str(e)}
            return SimulationResults(params=self.p, metadata=metadata, status=status_info["code"], simulation_time_seconds=timing_meta["simulation_seconds"])

        decoy_est, secure_len, cert = None, None, None
        try:
            decoy_est = self.proof_module.estimate_yields_and_errors(overall_stats)
            if not isinstance(decoy_est, dict) or "ok" not in decoy_est: raise LPFailureError("Decoy estimation returned an invalid result structure.")
            
            decoy_valid = False
            if decoy_est.get("ok"):
                y1 = float(decoy_est.get("Y1_L", -1)); e1 = float(decoy_est.get("e1_U", -1))
                if not (np.isfinite(y1) and np.isfinite(e1) and y1 >= -1e-12 and 0 <= e1 <= 1):
                    logger.error(f"Invalid decoy estimates produced: Y1_L={y1}, e1_U={e1}")
                else: decoy_valid = True
            
            metadata["decoy_estimation_valid"] = decoy_valid
            if decoy_est.get("lp_diagnostics"): metadata["lp_diagnostics"] = decoy_est.get("lp_diagnostics")
            
            if not decoy_est.get("ok") or not decoy_valid:
                status_info = {"code": "DECOY_ESTIMATION_FAILED", "message": decoy_est.get('status', 'Unknown')}
            else:
                secure_len_raw = self.proof_module.calculate_key_length(decoy_est, overall_stats)
                secure_len = 0 if secure_len_raw is None else int(secure_len_raw)
                if secure_len < 0: secure_len = 0
                if secure_len > sys.maxsize: secure_len = sys.maxsize
                
                cert = SecurityCertificate(
                    proof_name=self.p.security_proof.value,
                    confidence_bound_method=getattr(getattr(self.p, 'ci_method', None), 'value', None),
                    assumed_phase_equals_bit_error=getattr(self.p, 'assume_phase_equals_bit_error', False),
                    epsilon_allocation=getattr(self.proof_module, "eps_alloc", None),
                    lp_solver_diagnostics=decoy_est.get("lp_diagnostics"),
                )
        except Exception as e:
            status_info = {"code": "POST_PROCESSING_FAILED", "message": f"{type(e).__name__}: {e}"}

        metadata["status_message"] = status_info["message"]

        return SimulationResults(
            params=self.p, metadata=metadata, security_certificate=cert,
            decoy_estimates=decoy_est, secure_key_length=int(secure_len) if secure_len is not None else None,
            raw_sifted_key_length=int(total_sifted),
            simulation_time_seconds=timing_meta["simulation_seconds"],
            status=status_info["code"],
        )

    def _validate_and_merge_batch(self, batch_result: dict, overall_stats: dict, expected_pulses: int):
        """Helper to validate and merge results from a single batch."""
        if not isinstance(batch_result, dict): raise QKDSimulationError("Worker returned non-dict result.")
        if not set(batch_result.get('tallies', {}).keys()).issubset(overall_stats.keys()): raise QKDSimulationError("Worker returned unexpected tally keys.")
        
        num_pulses = batch_result.get("num_pulses", -1)
        if num_pulses != expected_pulses: raise QKDSimulationError(f"Worker processed {num_pulses} pulses, expected {expected_pulses}.")
        
        sifted_count = batch_result.get("sifted_count", -1)
        if not isinstance(sifted_count, (int, np.integer)) or not (0 <= sifted_count <= num_pulses):
            raise QKDSimulationError(f"Worker returned invalid sifted_count: {sifted_count} for {num_pulses} pulses.")

        for tally in batch_result.get('tallies', {}).values():
            for field in ['sent', 'detected_d0', 'detected_d1', 'error_d0', 'error_d1']:
                if isinstance(tally, dict): val = tally.get(field, -1)
                else: val = getattr(tally, field, -1)
                if not isinstance(val, (int, np.integer, np.ndarray)) or (isinstance(val, (int, np.integer)) and val < 0):
                    raise QKDSimulationError(f"Worker returned invalid tally field '{field}': {val}")

        try: _merge_batch_tallies(overall_stats, batch_result)
        except Exception as e: raise QKDSimulationError("Failed to merge batch tallies.") from e

    def _validate_final_tallies(self, overall_stats: dict, total_sifted: int):
        """Performs final consistency checks on aggregated tallies."""
        try:
            first_tally = next(iter(overall_stats.values()), None)
            if first_tally is None: return
            
            sift_field_candidates = ['sent', 'sifted', 'sifted_count']
            sift_field = None
            for field in sift_field_candidates:
                if hasattr(first_tally, field):
                    sift_field = field
                    break
                elif isinstance(first_tally, dict) and field in first_tally:
                    sift_field = field
                    break
            
            if sift_field:
                calculated_sifted = sum(getattr(t, sift_field) if not isinstance(t, dict) else t[sift_field] for t in overall_stats.values())
                if calculated_sifted != total_sifted:
                    raise QKDSimulationError(f"Tally integrity check failed: sum of tallies ({calculated_sifted}) != total_sifted ({total_sifted}).")
            else:
                logger.warning("Could not perform final tally integrity check: TallyCounts object is missing a recognized sifted-count field.")
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not perform final tally integrity check due to an error: {e}")
