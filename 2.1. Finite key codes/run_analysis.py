# run_analysis.py
#
# Author: Gemini
# Date: August 11, 2025
# Version: 12.4 (Production Grade + Optimizer)
# Description: A production-grade, reproducible, and robust analysis and
# orchestration script for QKD simulations. This version adds an 'optimize'
# command to automatically find the best batch_size and num_workers.

import argparse
import json
import numpy as np
import matplotlib
import os
import sys
import logging
import math
import numbers
import hashlib
import enum
import time
import signal
import platform
import uuid
import traceback
from multiprocessing import get_context
from logging.handlers import QueueHandler, QueueListener
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Sequence, TypedDict, get_type_hints, Union, get_origin, get_args
from inspect import signature

# --- Headless Matplotlib Backend Configuration ---
if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
    matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- Setup Logger ---
logger = logging.getLogger("QKDAnalysis")

# --- Import from Main Simulation Script ---
try:
    from qkd_simulation_main import QKDParams, QKDSystem
    QKD_SIM_VERSION = getattr(sys.modules['qkd_simulation_main'], '__version__', 'unknown')
except ImportError:
    logger.critical("Could not import from 'qkd_simulation_main.py'.")
    sys.exit(1)

# --- Type Definitions for Structured Output ---
class SeedResult(TypedDict):
    seed: int
    status: str
    skl_abs: float
    skl_per_pulse: float
    duration_seconds: float
    error: Optional[str]
    traceback: Optional[str]

class SweepPointResult(TypedDict):
    sweep_value: Any
    mean_skl_per_pulse: float
    ci_raw_lower: float
    ci_lower: float
    ci_upper: float
    ci_clamped_lower: bool
    ci_method: str
    bootstrap_seed: Optional[int]
    num_requested_runs: int
    num_successful_runs: int
    num_failures: int
    total_duration_seconds: float
    seed_results: Optional[List[SeedResult]]

class SweepMetadata(TypedDict):
    script_version: str
    config_hash: str
    base_params: Optional[Dict[str, Any]]
    sweep_param: str
    master_seed: int
    contains_nan: bool
    environment: Dict[str, str]

class SweepResults(TypedDict):
    metadata: SweepMetadata
    results: List[SweepPointResult]
    pending_sweep_values: Optional[List[Any]]

class Scenario(TypedDict):
    name: str
    param_path: str
    value: Any


# --- Core Utilities ---

def make_json_serializable(obj: Any, strict: bool = False) -> Any:
    """Recursively converts an object to be JSON serializable."""
    if obj is None or isinstance(obj, (str, bool)): return obj
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, numbers.Number): return float(obj)
    if hasattr(obj, 'to_dict'): return make_json_serializable(obj.to_dict(), strict)
    if isinstance(obj, dict): return {k: make_json_serializable(v, strict) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, np.ndarray)): return [make_json_serializable(v, strict) for v in obj]
    if isinstance(obj, enum.Enum): return getattr(obj, 'value', obj.name)
    if strict: raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    return {'__type__': type(obj).__name__, '__repr__': repr(obj)[:200]}

def get_type_hint_from_path(cls: type, path: str) -> type:
    """Introspects a dataclass to find the type hint for a nested attribute."""
    try:
        parts = path.replace("]", "").replace("[", ".").split(".")
        current_type = cls
        for part in parts:
            if hasattr(current_type, '__annotations__'):
                hints = get_type_hints(current_type, globalns=sys.modules[cls.__module__].__dict__)
                if part in hints:
                    current_type = hints[part]
                    origin = get_origin(current_type)
                    if origin in (list, List, Union):
                        args = get_args(current_type)
                        current_type = next((t for t in args if t is not type(None)), Any)
                else: raise ValueError(f"Path part '{part}' not found in type hints for {current_type}.")
            else: raise ValueError(f"Type {current_type} has no annotations to resolve path part '{part}'.")
        return current_type
    except Exception as e:
        raise ValueError(f"Could not resolve type hint for path '{path}': {e}") from e

def cast_value(value: Any, target_type: type) -> Any:
    """Robustly casts a value to a target type using type hints."""
    if value is None: return None
    if target_type is Any: return value
    origin = get_origin(target_type)
    args = get_args(target_type)
    if isinstance(target_type, type) and not origin and isinstance(value, target_type): return value
    if origin is Union:
        non_none_types = [t for t in args if t is not type(None)]
        if not non_none_types: return value
        target_type = non_none_types[0]
        origin = get_origin(target_type)
        args = get_args(target_type)
    if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
        if isinstance(value, str):
            try: return target_type[value.upper()]
            except KeyError:
                for member in target_type:
                    if str(member.value) == value or member.name.lower() == value.lower(): return member
                raise ValueError(f"'{value}' is not a valid name or value for enum {target_type.__name__}")
        return target_type(value)
    if origin is list or target_type is list:
        inner_type = args[0] if args else Any
        if isinstance(value, str): value = json.loads(value)
        if not isinstance(value, (list, tuple)): raise ValueError(f"Cannot cast {type(value)} to list")
        return [cast_value(v, inner_type) for v in value]
    if origin is dict or target_type is dict:
        key_type, val_type = (args + (Any, Any))[:2]
        if isinstance(value, str): value = json.loads(value)
        if not isinstance(value, dict): raise ValueError(f"Cannot cast {type(value)} to dict")
        return {cast_value(k, key_type): cast_value(v, val_type) for k, v in value.items()}
    if target_type is bool: return str(value).lower() in ('1', 'true', 't', 'yes')
    if target_type is int: return int(float(value))
    if target_type is float: return float(value)
    try: return target_type(value)
    except (TypeError, ValueError) as e:
        tname = getattr(target_type, '__name__', str(target_type))
        raise ValueError(f"Cannot cast '{value}' to type {tname}") from e

def set_param_by_path(params: QKDParams, path: str, value: Any) -> QKDParams:
    """Efficiently sets a parameter in a nested dataclass structure using a copy-on-write approach."""
    try:
        target_type = get_type_hint_from_path(QKDParams, path)
        casted_value = cast_value(value, target_type)
        params_dict = params.to_dict()
        keys = path.replace("]", "").replace("[", ".").split(".")
        def recurse_set(obj, path_keys, val):
            key = path_keys[0]
            if isinstance(obj, list) and key.isdigit():
                idx = int(key)
                new_obj = obj.copy()
                if len(path_keys) == 1: new_obj[idx] = val
                else: new_obj[idx] = recurse_set(new_obj[idx], path_keys[1:], val)
                return new_obj
            elif isinstance(obj, dict):
                new_obj = obj.copy()
                if len(path_keys) == 1: new_obj[key] = val
                else: new_obj[key] = recurse_set(new_obj[key], path_keys[1:], val)
                return new_obj
            else: raise TypeError(f"Cannot traverse non-container of type {type(obj)}")
        new_params_dict = recurse_set(params_dict, keys, casted_value)
        return QKDParams.from_dict(new_params_dict)
    except Exception as e: raise ValueError(f"Failed to set parameter path '{path}': {e}") from e

def permutation_test(data1: Sequence[float], data2: Sequence[float], num_permutations: int, rng: np.random.Generator) -> float:
    """Performs a two-sample permutation test for difference in means with correction."""
    data1, data2 = np.asarray(data1), np.asarray(data2)
    if len(data1) == 0 or len(data2) == 0: return 1.0
    observed_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    count = 0
    m = combined.size
    max_mem_bytes = 200 * 1024**2
    bytes_per_float = 8
    max_elems = max_mem_bytes // bytes_per_float
    max_k_for_m = max(1, max_elems // m) if m > 0 else num_permutations
    batch_size = min(1000, max_k_for_m)
    for i in range(0, num_permutations, batch_size):
        k = min(batch_size, num_permutations - i)
        keys = rng.random((k, m))
        order = np.argsort(keys, axis=1)
        permuted = combined[order]
        permuted_diffs = np.mean(permuted[:, :len(data1)], axis=1) - np.mean(permuted[:, len(data1):], axis=1)
        count += np.sum(np.abs(permuted_diffs) >= abs(observed_diff))
    return (count + 1) / (num_permutations + 1)

def calculate_bootstrap_ci(data: Sequence[float], num_resamples: int, ci_level: float, rng: np.random.Generator, method: str) -> Tuple[float, float, float, float, bool, Optional[int]]:
    """Calculates the mean and bootstrap CI, returning raw and clamped lower bounds."""
    data_array = np.asarray(data, dtype=np.float64)
    data_array = data_array[np.isfinite(data_array)]
    n = len(data_array)
    if n < 30: logger.warning(f"Sample size ({n}) is small. CI may be unreliable.")
    if n == 0: return np.nan, np.nan, np.nan, np.nan, False, None
    val = data_array[0]
    if n == 1 or np.all(data_array == val): return val, val, val, val, False, None
    original_mean = np.mean(data_array)
    seed_for_scipy = int(rng.integers(2**32 - 1))
    if method == 'bca':
        try:
            from scipy import stats
            res = stats.bootstrap((data_array,), np.mean, confidence_level=ci_level, n_resamples=num_resamples, method='BCa', random_state=seed_for_scipy)
            raw_lower, upper_bound = res.confidence_interval.low, res.confidence_interval.high
            return original_mean, raw_lower, max(0.0, raw_lower), upper_bound, raw_lower < 0.0, seed_for_scipy
        except (ImportError, TypeError):
            logger.warning("scipy not available or version is too old for BCa. Falling back to 'percentile'.")
    MAX_RESAMPLES = 500_000
    if num_resamples > MAX_RESAMPLES:
        logger.warning(f"num_resamples {num_resamples} is large, capping to {MAX_RESAMPLES} to conserve memory.")
        num_resamples = MAX_RESAMPLES
    bootstrap_means = np.empty(num_resamples)
    MAX_ENTRIES_PER_CHUNK = 5_000_000
    chunk_size = max(1, MAX_ENTRIES_PER_CHUNK // n) if n > 0 else num_resamples
    idx = 0
    while idx < num_resamples:
        k = min(chunk_size, num_resamples - idx)
        indices = rng.integers(0, n, size=(k, n))
        bootstrap_means[idx:idx+k] = data_array[indices].mean(axis=1)
        idx += k
    raw_lower, upper_bound = np.percentile(bootstrap_means, [(1.0 - ci_level) / 2.0 * 100, (1.0 + ci_level) / 2.0 * 100])
    clamped_lower = max(0.0, raw_lower)
    return original_mean, raw_lower, clamped_lower, upper_bound, clamped_lower != raw_lower, seed_for_scipy

def atomic_write(data: Any, path: str, raise_on_error: bool = True):
    """Writes data to a file atomically."""
    try:
        dir_path = os.path.dirname(path)
        if dir_path: os.makedirs(dir_path, exist_ok=True)
        base, ext = os.path.splitext(path)
        if isinstance(data, Figure) and not ext: ext = '.png'
        temp_path = f"{base}.tmp.{uuid.uuid4().hex}{ext or '.tmp'}"
        if isinstance(data, Figure):
            data.savefig(temp_path, dpi=300, bbox_inches='tight', format=(ext.lstrip('.') if ext else 'png'))
        elif isinstance(data, bytes):
            with open(temp_path, 'wb') as f: f.write(data)
        else:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(make_json_serializable(data), f, indent=4, ensure_ascii=False)
        os.replace(temp_path, path)
        logger.info(f"Successfully wrote output to {path}")
    except Exception as e:
        logger.error(f"Failed to write to file {path}: {e}")
        if raise_on_error: raise

# --- Worker Function and Logging Setup for Parallelism ---
def worker_init(log_queue, level):
    """Initializer for worker processes to configure logging."""
    matplotlib.use("Agg")
    h = QueueHandler(log_queue)
    root = logging.getLogger()
    root.handlers = [h]
    root.setLevel(level)

def run_single_simulation(params_json: str, seed: int) -> Dict:
    """A top-level function to run one simulation instance."""
    start_time = time.monotonic()
    try:
        params = QKDParams.from_dict(json.loads(params_json))
        qkd_system = QKDSystem(params=params, seed=seed)
        sim_results = qkd_system.run_simulation()
        duration = time.monotonic() - start_time
        status = getattr(sim_results, 'status', 'OK')
        skl = getattr(sim_results, 'secure_key_length', None)
        if status != 'OK': raise RuntimeError(f"Simulation failed with status: {status}")
        if skl is None: raise RuntimeError("Simulation returned no secure_key_length and no error status.")
        skl_abs = float(skl)
        skl_per_pulse = skl_abs / params.num_bits if params.num_bits > 0 else 0.0
        return {'seed': seed, 'status': 'OK', 'skl_abs': skl_abs, 'skl_per_pulse': skl_per_pulse, 'duration_seconds': duration, 'error': None, 'traceback': None}
    except Exception as e:
        tb = traceback.format_exc()
        return {'seed': seed, 'status': 'FAILED', 'skl_abs': 0.0, 'skl_per_pulse': np.nan, 'duration_seconds': time.monotonic() - start_time, 'error': str(e), 'traceback': tb}

# --- Main Analysis Functions ---

def run_sweep(args, base_params: QKDParams, sweep_param: str, sweep_values: list) -> SweepResults:
    """Runs a QKD simulation sweep with robust error handling and reproducible RNG."""
    master_ss = np.random.SeedSequence(args.seed)
    point_sequences = master_ss.spawn(len(sweep_values))
    logger.info(f"--- Starting sweep for parameter '{sweep_param}' with master seed {args.seed} ---")
    results_list: List[SweepPointResult] = []
    if args.resume:
        try:
            with open(args.resume, 'r') as f: partial_results = json.load(f)
            if partial_results['metadata']['sweep_param'] != sweep_param: raise ValueError("Resume file sweep parameter does not match.")
            results_list = partial_results['results']
            sweep_values = partial_results.get('pending_sweep_values', sweep_values)
            logger.info(f"Resumed sweep from '{args.resume}', with {len(sweep_values)} points remaining.")
        except Exception as e:
            logger.error(f"Could not resume from '{args.resume}': {e}. Starting a new sweep.")
            results_list = []
    contains_nan = any(np.isnan(r.get('mean_skl_per_pulse', np.nan)) for r in results_list)
    ctx = get_context('spawn')
    log_queue = ctx.Queue()
    listener = QueueListener(log_queue, *list(logger.handlers))
    listener.start()
    executor_params = {'max_workers': args.max_workers, 'initializer': worker_init, 'initargs': (log_queue, logger.getEffectiveLevel())}
    if 'mp_context' in signature(ProcessPoolExecutor).parameters:
        executor_params['mp_context'] = ctx
    try:
        with ProcessPoolExecutor(**executor_params) as executor:
            for i, value in enumerate(tqdm(sweep_values, desc=f"Sweeping {sweep_param}", file=sys.stdout, disable=args.no_progress)):
                point_start_time = time.monotonic()
                point_key_lengths, point_seed_results, num_failures = [], [], 0
                try: modified_params = set_param_by_path(base_params, sweep_param, value)
                except Exception as e:
                    logger.error(f"Failed to set parameter for sweep value '{value}'. Skipping. Error: {e}")
                    results_list.append({'sweep_value': value, 'mean_skl_per_pulse': np.nan, 'ci_raw_lower': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'ci_clamped_lower': False, 'ci_method': args.ci_method, 'bootstrap_seed': None, 'num_requested_runs': args.seeds, 'num_successful_runs': 0, 'num_failures': args.seeds, 'total_duration_seconds': 0.0, 'seed_results': []})
                    contains_nan = True
                    continue
                point_ss = point_sequences[i]
                seeds_rng = np.random.default_rng(point_ss)
                point_seeds = seeds_rng.integers(low=0, high=2**32 - 1, size=args.seeds, dtype=np.uint32)
                params_json = json.dumps(make_json_serializable(modified_params.to_dict()))
                future_to_seed = {executor.submit(run_single_simulation, params_json, int(s)): s for s in point_seeds}
                for future in as_completed(future_to_seed):
                    seed, result = future_to_seed[future], None
                    try:
                        result = future.result()
                        if result['status'] == 'OK':
                            point_key_lengths.append(result['skl_per_pulse'])
                            if args.include_seed_results: point_seed_results.append(result)
                        else: raise RuntimeError(result.get('error', 'Unknown worker error'))
                    except Exception as e:
                        logger.debug(f"Run failed for seed {seed} at sweep value '{value}'. Error: {e}")
                        if result and result.get('traceback'): logger.debug(f"Worker traceback:\n{result['traceback']}")
                        num_failures += 1
                        if args.include_seed_results and result: point_seed_results.append(result)
                bootstrap_ss = point_sequences[i].spawn(1)[0]
                bootstrap_rng = np.random.default_rng(bootstrap_ss)
                mean_skl, raw_lower, ci_lower, ci_upper, clamped, bs_seed = calculate_bootstrap_ci(point_key_lengths, args.bootstrap_resamples, 0.95, bootstrap_rng, args.ci_method)
                if np.isnan(mean_skl): contains_nan = True
                results_list.append({'sweep_value': value, 'mean_skl_per_pulse': mean_skl, 'ci_raw_lower': raw_lower, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'ci_clamped_lower': clamped, 'ci_method': args.ci_method, 'bootstrap_seed': bs_seed, 'num_requested_runs': args.seeds, 'num_successful_runs': len(point_key_lengths), 'num_failures': num_failures, 'total_duration_seconds': time.monotonic() - point_start_time, 'seed_results': point_seed_results if args.include_seed_results else None})
                if args.checkpoint_interval and args.output_json and ((i + 1) % args.checkpoint_interval == 0):
                    logger.info(f"Checkpointing results at sweep point {i+1}...")
                    checkpoint_meta = {'script_version': '12.3', 'config_hash': 'PARTIAL', 'sweep_param': sweep_param, 'master_seed': args.seed, 'checkpoint': True, 'checkpoint_index': i}
                    atomic_write({'metadata': checkpoint_meta, 'results': results_list, 'pending_sweep_values': sweep_values[i+1:]}, args.output_json + ".part")
    finally:
        listener.stop()
    base_params_dict = base_params.to_dict()
    config_hash = hashlib.sha256(json.dumps(make_json_serializable(base_params_dict), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    env_meta = {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'os': platform.platform(), 'qkd_simulation_version': QKD_SIM_VERSION}
    return {'metadata': {'script_version': '12.3', 'config_hash': config_hash, 'base_params': base_params_dict if args.include_base_params else None, 'sweep_param': sweep_param, 'master_seed': args.seed, 'contains_nan': contains_nan, 'environment': env_meta}, 'results': results_list}

def run_sensitivity_analysis(args, base_params: QKDParams, scenarios: List[Scenario]) -> Dict:
    """Runs a sensitivity analysis with statistical significance testing."""
    master_ss = np.random.SeedSequence(args.seed)
    all_sequences = master_ss.spawn(len(scenarios) + 1)
    baseline_ss = all_sequences[0]
    scenario_sequences = all_sequences[1:]
    logger.info(f"--- Starting sensitivity analysis with master seed {args.seed} ---")
    logger.info(f"Establishing baseline with {args.seeds} seeds...")
    baseline_key_lengths = []
    baseline_rng = np.random.default_rng(baseline_ss)
    baseline_seeds = baseline_rng.integers(low=0, high=2**32 - 1, size=args.seeds, dtype=np.uint32)
    params_json = json.dumps(make_json_serializable(base_params.to_dict()))
    for run_seed in tqdm(baseline_seeds, desc="Baseline Runs", file=sys.stdout, disable=args.no_progress):
        result = run_single_simulation(params_json, int(run_seed))
        if result['status'] == 'OK': baseline_key_lengths.append(result['skl_per_pulse'])
        else: logger.warning(f"Baseline run failed for seed {run_seed}: {result['error']}")
    if not baseline_key_lengths:
        logger.error("All baseline runs failed. Cannot proceed.")
        return {}
    baseline_mean_skl = np.mean(baseline_key_lengths)
    if baseline_mean_skl <= 0: logger.warning("Baseline SKL is zero or negative.")
    scenario_results_list = []
    logger.info("\n" + "="*100)
    logger.info(f"{'Scenario Name':<30} | {'Baseline SKL/pulse':>20} | {'Scenario SKL/pulse':>20} | {'% Change':>10} | {'p-value':>10}")
    logger.info("-" * 100)
    for i, scenario in enumerate(scenarios):
        name = scenario.get('name', 'Unnamed Scenario')
        param_path = scenario.get('param_path') or scenario.get('param_to_modify')
        value = scenario.get('value') if 'value' in scenario else scenario.get('new_value')
        if 'modifications' in scenario and isinstance(scenario['modifications'], dict):
            if len(scenario['modifications']) == 1:
                param_path, value = list(scenario['modifications'].items())[0]
            else:
                logger.warning(f"Scenario '{name}' has multiple modifications, which is not supported. Skipping.")
                continue
        if not all([name, param_path, value is not None]):
            logger.warning(f"Skipping invalid scenario (missing name, param_path, or value): {scenario}")
            continue
        try: modified_params = set_param_by_path(base_params, param_path, value)
        except Exception as e:
            logger.error(f"Could not apply scenario '{name}'. Skipping. Error: {e}")
            continue
        scenario_key_lengths = []
        scenario_ss = scenario_sequences[i]
        scenario_rng = np.random.default_rng(scenario_ss)
        scenario_seeds = scenario_rng.integers(low=0, high=2**32 - 1, size=args.seeds, dtype=np.uint32)
        modified_params_json = json.dumps(make_json_serializable(modified_params.to_dict()))
        for run_seed in scenario_seeds:
            result = run_single_simulation(modified_params_json, int(run_seed))
            if result['status'] == 'OK': scenario_key_lengths.append(result['skl_per_pulse'])
        scenario_mean_skl = np.mean(scenario_key_lengths) if scenario_key_lengths else 0.0
        test_ss = scenario_sequences[i].spawn(1)[0]
        test_rng = np.random.default_rng(test_ss)
        p_value = permutation_test(np.asarray(baseline_key_lengths), np.asarray(scenario_key_lengths), args.num_permutations, test_rng)
        change_str = f"{((scenario_mean_skl - baseline_mean_skl) / baseline_mean_skl) * 100:+.2f}%" if baseline_mean_skl > 0 else "N/A"
        logger.info(f"{name:<30} | {baseline_mean_skl:20.6e} | {scenario_mean_skl:20.6e} | {change_str:>10} | {p_value:10.4f}")
        scenario_results_list.append({'name': name, 'param_path': param_path, 'value': value, 'mean_skl_per_pulse': scenario_mean_skl, 'percent_change': float(change_str.strip('%')) if change_str != "N/A" else None, 'p_value': p_value})
    logger.info("="*100)
    return {'metadata': {'master_seed': args.seed, 'base_config_hash': hashlib.sha256(json.dumps(make_json_serializable(base_params.to_dict()), sort_keys=True).encode()).hexdigest()}, 'baseline_mean_skl_per_pulse': baseline_mean_skl, 'scenarios': scenario_results_list}

def plot_sweep_results(sweep_results: SweepResults, output_path: str, use_log_scale: bool):
    """Plots sweep results with confidence intervals, handling numeric and categorical data."""
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    param_name = sweep_results['metadata']['sweep_param']
    results = sweep_results['results']
    x_values = [r['sweep_value'] for r in results]
    y_means = np.array([r['mean_skl_per_pulse'] for r in results])
    y_lower = np.array([r['ci_lower'] for r in results])
    y_upper = np.array([r['ci_upper'] for r in results])
    valid_mask = np.isfinite(y_means) & np.isfinite(y_lower) & np.isfinite(y_upper)
    if np.sum(valid_mask) < 1:
        logger.warning("No valid data points to generate a plot.")
        return
    x_valid, y_valid, lower_valid, upper_valid = np.array(x_values)[valid_mask], y_means[valid_mask], y_lower[valid_mask], y_upper[valid_mask]
    fig, ax = plt.subplots(figsize=(12, 8))
    y_plot = y_valid
    try:
        x_numeric = x_valid.astype(float)
        sort_indices = np.argsort(x_numeric)
        x_plot, y_plot, lower_plot, upper_plot = x_numeric[sort_indices], y_valid[sort_indices], lower_valid[sort_indices], upper_valid[sort_indices]
        ax.plot(x_plot, y_plot, 'o-', label='Mean SKL', color='royalblue')
        ax.fill_between(x_plot, lower_plot, upper_plot, color='cornflowerblue', alpha=0.3, label='95% CI')
    except (ValueError, TypeError):
        logger.info("X-axis values are not numeric; creating a categorical plot.")
        indices = np.arange(len(x_valid))
        ax.errorbar(indices, y_valid, yerr=[y_valid - lower_valid, upper_valid - y_valid], fmt='o', capsize=5, label='Mean SKL with 95% CI')
        ax.set_xticks(indices)
        ax.set_xticklabels([str(v) for v in x_valid], rotation=45, ha="right")
    ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel('Secure Key Length (bits/pulse)', fontsize=14)
    ax.set_title(f'QKD Secure Key Length vs. {param_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    if use_log_scale:
        if np.any(y_plot <= 0): logger.warning("Y-log scale requested but data contains non-positive values. Using linear scale instead.")
        else:
            ax.set_yscale('log')
            ax.set_ylim(bottom=max(np.min(y_plot) * 0.1, 1e-12))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    atomic_write(fig, output_path)
    plt.close(fig)

# --- CLI and Main Execution ---
def configure_logging(verbose_level: int, log_file: Optional[str] = None):
    """Configures the root logger based on CLI arguments."""
    level = logging.WARNING
    if verbose_level == 1: level = logging.INFO
    elif verbose_level >= 2: level = logging.DEBUG
    logger.handlers.clear()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"))
    logger.addHandler(handler)
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode='a')
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s"))
            logger.addHandler(fh)
        except IOError as e: logger.error(f"Could not open log file {log_file}: {e}")

def main():
    """Main function to parse command-line arguments and orchestrate the analysis."""
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    parser = argparse.ArgumentParser(description="QKD Simulation Analysis and Orchestration Tool (v12.3).", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--error-json', type=str, default=None)
    subparsers = parser.add_subparsers(dest='command', required=True)
    parser_sweep = subparsers.add_parser('sweep', help="Run a parameter sweep.")
    sweep_group = parser_sweep.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument('--numeric', nargs=4, metavar=('PARAM_PATH', 'START', 'STOP', 'STEPS'))
    sweep_group.add_argument('--categorical', nargs='+', metavar=('PARAM_PATH', 'VAL1'))
    parser_sweep.add_argument('--seeds', type=int, default=10)
    parser_sweep.add_argument('--output-plot', type=str, default=None)
    parser_sweep.add_argument('--output-json', type=str, default=None)
    parser_sweep.add_argument('--ylog', action='store_true')
    parser_sweep.add_argument('--include-seed-results', action='store_true')
    parser_sweep.add_argument('--bootstrap-resamples', type=int, default=2000)
    parser_sweep.add_argument('--checkpoint-interval', type=int, default=0)
    parser_sweep.add_argument('--parallel', choices=['none', 'sweep_point'], default='none', help="Parallelization strategy.")
    parser_sweep.add_argument('--max-workers', type=int, default=default_workers)
    parser_sweep.add_argument('--include-base-params', action='store_true')
    parser_sweep.add_argument('--resume', type=str, default=None)
    parser_sweep.add_argument('--ci-method', choices=['percentile', 'bca'], default='percentile', help="Method for confidence interval calculation.")
    parser_sensitivity = subparsers.add_parser('sensitivity', help="Run a sensitivity analysis.")
    parser_sensitivity.add_argument('--scenarios-file', type=str, required=True)
    parser_sensitivity.add_argument('--seeds', type=int, default=100, help="Seeds per scenario (recommend >=100 for stats).")
    parser_sensitivity.add_argument('--num-permutations', type=int, default=10000)
    parser_sensitivity.add_argument('--output-json', type=str, default=None)
    args = parser.parse_args()
    configure_logging(args.verbose, args.log_file)
    if args.seed is None: args.seed = int.from_bytes(os.urandom(4), 'little')
    else: args.seed = int(args.seed) & 0xFFFFFFFF
    logger.info(f"Using master seed: {args.seed}")
    def graceful_exit(signum, frame):
        logger.warning(f"Received signal {signum}. Shutting down gracefully.")
        sys.exit(128 + signum)
    if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, graceful_exit)
    if hasattr(signal, 'SIGHUP'): signal.signal(signal.SIGHUP, graceful_exit)
    try:
        with open(args.config_file, 'r') as f:
            if os.path.getsize(args.config_file) > 50 * 1024 * 1024: raise ValueError("Config file size exceeds 50MB limit.")
            config_dict = json.load(f)
            if 'num_bits' not in config_dict or 'pulse_configs' not in config_dict: raise KeyError("Config file missing required keys 'num_bits' or 'pulse_configs'.")
            base_params = QKDParams.from_dict(config_dict)
        logger.info(f"Successfully loaded base parameters from '{args.config_file}'.")
    except Exception as e:
        logger.critical(f"Error loading config file '{args.config_file}': {e}", exc_info=True)
        if args.error_json: atomic_write({'error': str(e), 'type': type(e).__name__, 'traceback': traceback.format_exc()}, args.error_json)
        sys.exit(1)
    try:
        if args.command == 'sweep':
            if not args.output_plot and not args.output_json: parser.error("For 'sweep', at least one output is required.")
            if args.checkpoint_interval and not args.output_json: parser.error("--checkpoint-interval requires --output-json.")
            if args.numeric:
                param, start, stop, steps = args.numeric
                if int(steps) < 2: parser.error("Numeric sweep steps must be >= 2.")
                sweep_values = np.linspace(float(start), float(stop), int(steps)).tolist()
            else:
                param, *sweep_values = args.categorical
            if args.include_seed_results and args.seeds * len(sweep_values) > 10000: logger.warning("High number of total runs with --include-seed-results. Output JSON may be very large.")
            results = run_sweep(args, base_params, param, sweep_values)
            if args.output_plot: plot_sweep_results(results, args.output_plot, args.ylog)
            if args.output_json: atomic_write(results, args.output_json)
        elif args.command == 'sensitivity':
            with open(args.scenarios_file, 'r') as f: scenarios: List[Scenario] = json.load(f)
            results = run_sensitivity_analysis(args, base_params, scenarios)
            if args.output_json: atomic_write(results, args.output_json)
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user. Exiting.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        if args.error_json: atomic_write({'error': str(e), 'type': type(e).__name__, 'traceback': traceback.format_exc()}, args.error_json)
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    main()
