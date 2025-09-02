# -*- coding: utf-8 -*-
"""
QKD Simulation with Rigorous Finite-Key Decoy State Analysis (v16.4 - Relaxation & Robust LP)
This version adds a robust LP solving strategy that attempts progressively
weaker (but still conservative) constraint sets if the strict LP is infeasible.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import struct
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import copy

import numpy as np
from numpy.random import Generator

# SciPy dependencies
try:
    from scipy.optimize import OptimizeResult, linprog
    from scipy.sparse import csr_matrix
    from scipy.stats import beta, poisson
except ImportError:
    logging.critical("CRITICAL ERROR: SciPy is required. Run `pip install scipy`.")
    sys.exit(1)

# Optional plotting (lazy import in plot function)
PLOTTING_AVAILABLE = False
plt = None
sns = None

# tqdm fallback for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("QKDSystem")

# --- Constants & Numeric Tolerances ---
MAX_SEED_INT = 2**63 - 1
LP_SOLVER_METHODS = ["highs"]
NUMERIC_ABS_TOL = 1e-12
NUMERIC_REL_TOL = 1e-9
LP_VIOLATION_MULTIPLIER = 10.0
Y1_SAFE_THRESHOLD = 1e-12
ENTROPY_PROB_CLAMP = 1e-15
PROB_SUM_TOL = 1e-8
POISSON_RENORM_TOL = 1e-7  # If Poisson pmf+sf deviates that much, warn/raise.

# --- Custom Exceptions ---
class ParameterValidationError(Exception):
    pass

class QKDSimulationError(Exception):
    pass

class LPFailureError(Exception):
    pass

# --- Enums and Dataclasses ---
class DoubleClickPolicy(Enum):
    DISCARD = "discard"
    RANDOM = "random"

class SecurityProof(Enum):
    LIM_2014 = "lim-2014"

class ConfidenceBoundMethod(Enum):
    CLOPPER_PEARSON = "clopper-pearson"
    HOEFFDING = "hoeffding"

@dataclass(frozen=True)
class PulseTypeConfig:
    name: str
    mean_photon_number: float
    probability: float

@dataclass
class TallyCounts:
    # Aggregates (total)
    sent: int = 0
    sifted: int = 0
    errors_sifted: int = 0
    double_clicks_discarded: int = 0
    # Basis-resolved counts (Z and X)
    sent_z: int = 0
    sent_x: int = 0
    sifted_z: int = 0
    sifted_x: int = 0
    errors_sifted_z: int = 0
    errors_sifted_x: int = 0

@dataclass
class EpsilonAllocation:
    eps_sec: float
    eps_cor: float
    eps_pe: float
    eps_smooth: float
    eps_pa: float
    eps_phase_est: float

    def validate(self):
        if not (self.eps_cor > 0 and self.eps_pa > 0 and self.eps_pe > 0 and self.eps_smooth > 0 and self.eps_phase_est > 0):
            raise ParameterValidationError("All component epsilons must be > 0.")
        total_sum = self.eps_pe + 2 * self.eps_smooth + self.eps_pa
        if total_sum > self.eps_sec + NUMERIC_ABS_TOL:
            raise ParameterValidationError(
                f"Epsilon allocation insecure: sum(ep_pe + 2*eps_smooth + eps_pa = {total_sum:.2e}) > eps_sec ({self.eps_sec:.2e})."
            )

@dataclass
class SecurityCertificate:
    proof_name: str
    confidence_bound_method: str
    assumed_phase_equals_bit_error: bool
    epsilon_allocation: EpsilonAllocation
    lp_solver_diagnostics: Optional[Dict] = None

@dataclass
class QKDParams:
    num_bits: int
    pulse_configs: List[PulseTypeConfig]
    distance_km: float
    fiber_loss_db_km: float
    det_eff: float
    dark_rate: float
    qber_intrinsic: float
    misalignment: float
    double_click_policy: DoubleClickPolicy
    bob_z_basis_prob: float
    alice_z_basis_prob: float
    f_error_correction: float
    eps_sec: float
    eps_cor: float
    eps_pe: float
    eps_smooth: float
    photon_number_cap: int
    batch_size: int
    num_workers: int
    force_sequential: bool
    security_proof: SecurityProof
    ci_method: ConfidenceBoundMethod
    enforce_monotonicity: bool
    assume_phase_equals_bit_error: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not self.num_bits > 0:
            raise ParameterValidationError("num_bits must be positive.")
        p_sum = sum(pc.probability for pc in self.pulse_configs)
        if not math.isclose(p_sum, 1.0, rel_tol=PROB_SUM_TOL, abs_tol=PROB_SUM_TOL):
            raise ParameterValidationError(f"Sum of pulse_configs probabilities must be ~1.0 (got {p_sum}).")
        epsilons = [self.eps_sec, self.eps_cor, self.eps_pe, self.eps_smooth]
        if not all(0 < e < 1 for e in epsilons):
            raise ParameterValidationError("All epsilon security parameters must be in (0, 1).")
        if not (0 <= self.dark_rate < 1):
            raise ParameterValidationError("dark_rate must be in [0, 1).")
        if not (1.0 <= self.f_error_correction <= 5.0):
            raise ParameterValidationError("f_error_correction must be in [1.0, 5.0].")
        if not (0 <= self.misalignment < 1.0):
            raise ParameterValidationError("misalignment must be in [0, 1).")
        if not (self.batch_size > 0 and self.batch_size <= self.num_bits):
            raise ParameterValidationError("batch_size must be positive and not exceed num_bits.")
        if not (0.0 < self.bob_z_basis_prob < 1.0):
            raise ParameterValidationError("bob_z_basis_prob must be in (0,1).")
        # Allow alice_z_basis_prob equal to 0 or 1 for testing/extreme cases
        if not (0.0 <= self.alice_z_basis_prob <= 1.0):
            raise ParameterValidationError("alice_z_basis_prob must be in [0,1].")
        if self.photon_number_cap < 1:
            raise ParameterValidationError("photon_number_cap must be >= 1.")
        if not (0.0 <= self.det_eff <= 1.0):
            raise ParameterValidationError("det_eff must be in [0,1].")

    def get_pulse_config_by_name(self, name: str) -> Optional[PulseTypeConfig]:
        return next((c for c in self.pulse_configs if c.name == name), None)

    @property
    def transmittance(self) -> float:
        if self.distance_km < 0 or self.fiber_loss_db_km < 0:
            return 0.0
        return 10 ** (-(self.distance_km * self.fiber_loss_db_km) / 10.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_bits": self.num_bits,
            "pulse_configs": [dataclasses.asdict(pc) for pc in self.pulse_configs],
            "distance_km": self.distance_km,
            "fiber_loss_db_km": self.fiber_loss_db_km,
            "det_eff": self.det_eff,
            "dark_rate": self.dark_rate,
            "qber_intrinsic": self.qber_intrinsic,
            "misalignment": self.misalignment,
            "double_click_policy": self.double_click_policy.value,
            "bob_z_basis_prob": self.bob_z_basis_prob,
            "alice_z_basis_prob": self.alice_z_basis_prob,
            "f_error_correction": self.f_error_correction,
            "eps_sec": self.eps_sec,
            "eps_cor": self.eps_cor,
            "eps_pe": self.eps_pe,
            "eps_smooth": self.eps_smooth,
            "photon_number_cap": self.photon_number_cap,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "force_sequential": self.force_sequential,
            "security_proof": self.security_proof.value,
            "ci_method": self.ci_method.value,
            "enforce_monotonicity": self.enforce_monotonicity,
            "assume_phase_equals_bit_error": self.assume_phase_equals_bit_error,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QKDParams":
        pulse_configs = [PulseTypeConfig(**pc) for pc in d["pulse_configs"]]
        return QKDParams(
            num_bits=int(d["num_bits"]),
            pulse_configs=pulse_configs,
            distance_km=float(d["distance_km"]),
            fiber_loss_db_km=float(d["fiber_loss_db_km"]),
            det_eff=float(d["det_eff"]),
            dark_rate=float(d["dark_rate"]),
            qber_intrinsic=float(d["qber_intrinsic"]),
            misalignment=float(d["misalignment"]),
            double_click_policy=DoubleClickPolicy(d["double_click_policy"]),
            bob_z_basis_prob=float(d["bob_z_basis_prob"]),
            alice_z_basis_prob=float(d.get("alice_z_basis_prob", 0.5)),
            f_error_correction=float(d["f_error_correction"]),
            eps_sec=float(d["eps_sec"]),
            eps_cor=float(d["eps_cor"]),
            eps_pe=float(d["eps_pe"]),
            eps_smooth=float(d["eps_smooth"]),
            photon_number_cap=int(d["photon_number_cap"]),
            batch_size=int(d["batch_size"]),
            num_workers=int(d["num_workers"]),
            force_sequential=bool(d["force_sequential"]),
            security_proof=SecurityProof(d["security_proof"]),
            ci_method=ConfidenceBoundMethod(d["ci_method"]),
            enforce_monotonicity=bool(d["enforce_monotonicity"]),
            assume_phase_equals_bit_error=bool(d["assume_phase_equals_bit_error"]),
        )

@dataclass
class SimulationResults:
    params: QKDParams
    metadata: Dict[str, Any]
    security_certificate: Optional[SecurityCertificate] = None
    decoy_estimates: Optional[Dict[str, Any]] = None
    secure_key_length: Optional[int] = None
    raw_sifted_key_length: int = 0
    simulation_time_seconds: float = 0.0
    status: str = "OK"

    def to_serializable_dict(self) -> Dict[str, Any]:
        def convert(o: Any) -> Any:
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, Enum):
                return o.value
            if dataclasses.is_dataclass(o):
                return {k: convert(v) for k, v in asdict(o).items()}
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [convert(i) for i in o]
            if isinstance(o, float) and not np.isfinite(o):
                return str(o)
            return o

        base = {
            "params": self.params.to_dict(),
            "metadata": self.metadata,
            "security_certificate": convert(self.security_certificate) if self.security_certificate else None,
            "decoy_estimates": convert(self.decoy_estimates),
            "secure_key_length": self.secure_key_length,
            "raw_sifted_key_length": self.raw_sifted_key_length,
            "simulation_time_seconds": self.simulation_time_seconds,
            "status": self.status,
        }
        return convert(base)

    def save_json(self, path: str):
        try:
            full_path = os.path.abspath(path)
            dir_path = os.path.dirname(full_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            tmp_path = full_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.to_serializable_dict(), f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, full_path)
            logger.info(f"Results saved to JSON: {full_path}")
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save results to {path}: {e}", exc_info=True)

# --- Worker Utilities ---
def _serialize_params_for_worker(params: QKDParams) -> Dict[str, Any]:
    return params.to_dict()

def _deserialize_params_in_worker(serialized_params: Dict[str, Any]) -> QKDParams:
    return QKDParams.from_dict(copy.deepcopy(serialized_params))

def _top_level_worker_function(serialized_params: Dict, num_pulses: int, seed: int) -> Dict:
    try:
        deserialized_params = _deserialize_params_in_worker(serialized_params)
        rng = np.random.default_rng(int(seed) % MAX_SEED_INT)
        qkd_system = QKDSystem(deserialized_params, seed=int(seed))
        return qkd_system._simulate_quantum_part_batch(num_pulses, rng)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in worker process with seed {seed}: {e}\n{tb}")
        raise RuntimeError(f"Worker error: {e}\n{tb}")

# --- Statistical Helpers ---
def p_n_mu_vector(mu: float, n_cap: int) -> np.ndarray:
    if mu < 0:
        raise ValueError("Mean `mu` must be non-negative.")
    if n_cap < 1:
        raise ValueError("n_cap must be >= 1.")
    ns = np.arange(0, n_cap)
    pmf = poisson.pmf(ns, mu)
    tail = poisson.sf(n_cap - 1, mu)
    vec = np.concatenate([pmf, [tail]])
    s = float(np.sum(vec))
    if not math.isclose(s, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        if abs(1.0 - s) > POISSON_RENORM_TOL:
            logger.warning(f"Poisson PMF + SF sum deviates by {abs(1.0-s):.3e} (mu={mu}, n_cap={n_cap}). Renormalizing anyway.")
        vec = vec / s
    return vec

def hoeffding_bounds(k: int, n: int, failure_prob: float) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    if not (0 < failure_prob < 1):
        raise ValueError("failure_prob must be in (0,1).")
    delta = math.sqrt(math.log(2.0 / failure_prob) / (2.0 * n))
    p_hat = k / n
    return max(0.0, p_hat - delta), min(1.0, p_hat + delta)

def clopper_pearson_bounds(k: int, n: int, failure_prob: float) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    if not (0 < failure_prob < 1):
        raise ValueError("failure_prob must be in (0,1).")
    alpha = failure_prob
    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2.0, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    lower = float(np.nan_to_num(lower, nan=0.0))
    upper = float(np.nan_to_num(upper, nan=1.0))
    if not (0.0 <= lower <= upper <= 1.0):
        logger.warning("Clopper-Pearson produced invalid bounds; falling back to Hoeffding as a safe approximation.")
        return hoeffding_bounds(k, n, failure_prob)
    return lower, upper

# --- Finite-Key Proof Abstraction ---
class FiniteKeyProof:
    def __init__(self, params: QKDParams):
        self.p = params
        self.eps_alloc = self.allocate_epsilons()
        self.eps_alloc.validate()

    def allocate_epsilons(self) -> EpsilonAllocation:
        raise NotImplementedError

    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> Dict[str, Any]:
        raise NotImplementedError

    def calculate_key_length(self, decoy_estimates: Dict[str, Any], signal_stats: TallyCounts) -> int:
        raise NotImplementedError

    def get_bounds(self, k: int, n: int, failure_prob: float) -> Tuple[float, float]:
        if self.p.ci_method == ConfidenceBoundMethod.CLOPPER_PEARSON:
            return clopper_pearson_bounds(k, n, failure_prob)
        elif self.p.ci_method == ConfidenceBoundMethod.HOEFFDING:
            return hoeffding_bounds(k, n, failure_prob)
        else:
            raise NotImplementedError(f"CI method {self.p.ci_method} not implemented.")

# --- Lim et al. 2014 Style Proof (Conservative and Robust LP) ---
class Lim2014Proof(FiniteKeyProof):
    def allocate_epsilons(self) -> EpsilonAllocation:
        n_intensities = len(self.p.pulse_configs)
        n_ci_tests = 4 * n_intensities
        n_phase_err_tests = 1
        total_tests = n_ci_tests + n_phase_err_tests
        eps_pe_total = self.p.eps_pe
        eps_per_test = eps_pe_total / max(1, total_tests)
        eps_phase_est = eps_per_test
        eps_pa_unvalidated = self.p.eps_sec - (eps_pe_total + 2 * self.p.eps_smooth)
        if eps_pa_unvalidated <= 0:
            raise ParameterValidationError(
                f"Insecure epsilon allocation: eps_sec ({self.p.eps_sec}) too small for eps_pe ({eps_pe_total}) and eps_smooth ({self.p.eps_smooth})."
            )
        return EpsilonAllocation(
            eps_sec=self.p.eps_sec,
            eps_cor=self.p.eps_cor,
            eps_pe=eps_pe_total,
            eps_smooth=self.p.eps_smooth,
            eps_pa=eps_pa_unvalidated,
            eps_phase_est=eps_phase_est,
        )

    def _idx_y(self, n: int, Nvar: int) -> int:
        return n

    def _idx_e(self, n: int, Nvar: int) -> int:
        return Nvar + n

    def _solve_lp(self, cost_vector: np.ndarray, A_ub: csr_matrix, b_ub: np.ndarray, n_vars: int) -> Tuple[np.ndarray, Dict]:
        bounds = [(0.0, 1.0)] * n_vars
        res: OptimizeResult = linprog(cost_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise LPFailureError(f"LP solver failed: {res.message} (Status: {res.status})")
        sol = res.x.copy()
        if np.any(sol < -1e-8) or np.any(sol > 1.0 + 1e-8):
            raise LPFailureError(f"LP returned variables outside acceptable numeric bounds: min={np.min(sol)}, max={np.max(sol)}")
        residual = A_ub.dot(sol) - b_ub
        max_violation = float(np.max(residual)) if residual.size > 0 else 0.0
        b_ub_mag = np.max(np.abs(b_ub)) if b_ub.size > 0 else 1.0
        tol_violation = max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * max(1.0, b_ub_mag))
        if max_violation > tol_violation * LP_VIOLATION_MULTIPLIER:
            raise LPFailureError(f"LP solution violates constraints significantly. Max violation: {max_violation:.3e}, tol: {tol_violation:.3e}")
        diagnostics = {
            "method": "highs",
            "status": int(res.status),
            "message": str(res.message),
            "fun": float(res.fun),
            "nit": int(getattr(res, "nit", -1)),
            "max_violation": max_violation,
        }
        return sol, diagnostics

    def _build_constraints(self, required: List[str], stats_map: Dict[str, TallyCounts],
                           use_basis_z: bool, enforce_monotonicity: bool, enforce_half_error: bool) -> Tuple[csr_matrix, np.ndarray, int]:
        """
        Build A_ub, b_ub for the LP according to flags:
        - use_basis_z: whether to use Z-basis counts for CI (True) or totals (False)
        - enforce_monotonicity: include monotonicity constraints (might skip tail)
        - enforce_half_error: include E_n <= 0.5 * Y_n constraint
        Returns (A_ub, b_ub, Nvar)
        """
        cap = self.p.photon_number_cap
        Nvar = cap + 1
        rows, cols, data, b_ub = [], [], [], []
        row_idx = 0
        pulse_map = {pc.name: pc for pc in self.p.pulse_configs}

        # For each intensity, compute Q and R bounds using either Z-basis or total counts
        eps_per_ci = self.eps_alloc.eps_pe / (4 * len(required) + 1)
        Q_L, Q_U, R_L, R_U = {}, {}, {}, {}
        for name in required:
            stats = stats_map[name]
            if use_basis_z:
                sent = stats.sent_z
                sifted = stats.sifted_z
                errors = stats.errors_sifted_z
            else:
                sent = stats.sent
                sifted = stats.sifted
                errors = stats.errors_sifted
            if sent > 0:
                q_l, q_u = self.get_bounds(sifted, sent, eps_per_ci)
                r_l, r_u = self.get_bounds(errors, sent, eps_per_ci)
            else:
                q_l, q_u, r_l, r_u = 0.0, 1.0, 0.0, 1.0
            Q_L[name], Q_U[name], R_L[name], R_U[name] = q_l, q_u, r_l, r_u

        for name in required:
            mu = pulse_map[name].mean_photon_number
            p_vec = p_n_mu_vector(mu, cap)
            # sum p_n * Y_n <= Q_U
            rows.extend([row_idx] * Nvar)
            cols.extend([self._idx_y(n, Nvar) for n in range(Nvar)])
            data.extend(p_vec.tolist())
            b_ub.append(Q_U[name]); row_idx += 1
            # -sum p_n * Y_n <= -Q_L
            rows.extend([row_idx] * Nvar)
            cols.extend([self._idx_y(n, Nvar) for n in range(Nvar)])
            data.extend((-p_vec).tolist())
            b_ub.append(-Q_L[name]); row_idx += 1
            # sum p_n * E_n <= R_U
            rows.extend([row_idx] * Nvar)
            cols.extend([self._idx_e(n, Nvar) for n in range(Nvar)])
            data.extend(p_vec.tolist())
            b_ub.append(R_U[name]); row_idx += 1
            # -sum p_n * E_n <= -R_L
            rows.extend([row_idx] * Nvar)
            cols.extend([self._idx_e(n, Nvar) for n in range(Nvar)])
            data.extend((-p_vec).tolist())
            b_ub.append(-R_L[name]); row_idx += 1

        # E_n <= Y_n
        for n in range(Nvar):
            rows.extend([row_idx, row_idx])
            cols.extend([self._idx_e(n, Nvar), self._idx_y(n, Nvar)])
            data.extend([1.0, -1.0])
            b_ub.append(0.0); row_idx += 1

        # optional E_n <= 0.5 Y_n
        if enforce_half_error:
            for n in range(Nvar):
                rows.extend([row_idx, row_idx])
                cols.extend([self._idx_e(n, Nvar), self._idx_y(n, Nvar)])
                data.extend([1.0, -0.5])
                b_ub.append(0.0); row_idx += 1

        # Monotonicity Y_{n+1} - Y_n <= 0 for n=0..Nvar-3 (avoid tail aggregator)
        if enforce_monotonicity and Nvar >= 3:
            for n in range(Nvar - 2):
                rows.extend([row_idx, row_idx])
                cols.extend([self._idx_y(n + 1, Nvar), self._idx_y(n, Nvar)])
                data.extend([1.0, -1.0])
                b_ub.append(0.0); row_idx += 1

        A_ub = csr_matrix((data, (rows, cols)), shape=(row_idx, 2 * Nvar))
        b_ub_np = np.array(b_ub, dtype=float)
        return A_ub, b_ub_np, Nvar

    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> Dict[str, Any]:
        required = [pc.name for pc in self.p.pulse_configs]
        for r in required:
            if r not in stats_map:
                return {"status": f"MISSING_INTENSITY: {r}", "ok": False}

        # Try solving the LP with a sequence of relaxations:
        # 1) strict: use Z-basis, enforce monotonicity, enforce half-error
        # 2) relax monotonicity
        # 3) relax half-error constraint
        # 4) use total counts instead of Z-basis (with original constraints)
        # 5) fallback conservative
        attempts = []
        try_sequence = [
            {"use_basis_z": True, "enforce_monotonicity": self.p.enforce_monotonicity, "enforce_half_error": True, "label": "Z_mon_half"},
            {"use_basis_z": True, "enforce_monotonicity": False, "enforce_half_error": True, "label": "Z_noMon_half"},
            {"use_basis_z": True, "enforce_monotonicity": False, "enforce_half_error": False, "label": "Z_noMon_noHalf"},
            {"use_basis_z": False, "enforce_monotonicity": self.p.enforce_monotonicity, "enforce_half_error": True, "label": "Total_mon_half"},
            {"use_basis_z": False, "enforce_monotonicity": False, "enforce_half_error": True, "label": "Total_noMon_half"},
        ]

        last_exc = None
        final_lp_diag = []
        for attempt in try_sequence:
            label = attempt["label"]
            try:
                A_ub, b_ub_np, Nvar = self._build_constraints(required, stats_map,
                                                              use_basis_z=attempt["use_basis_z"],
                                                              enforce_monotonicity=attempt["enforce_monotonicity"],
                                                              enforce_half_error=attempt["enforce_half_error"])
                # Solve LPs: Y0_L, Y1_L, E1_U
                # 1. Minimize Y_0
                c_y0 = np.zeros(2 * Nvar); c_y0[self._idx_y(0, Nvar)] = 1.0
                sol_y0, d_y0 = self._solve_lp(c_y0, A_ub, b_ub_np, 2 * Nvar)
                Y0_L = float(sol_y0[self._idx_y(0, Nvar)])
                # 2. Minimize Y_1
                if Nvar >= 2:
                    c_y1 = np.zeros(2 * Nvar); c_y1[self._idx_y(1, Nvar)] = 1.0
                    sol_y1, d_y1 = self._solve_lp(c_y1, A_ub, b_ub_np, 2 * Nvar)
                    Y1_L = float(sol_y1[self._idx_y(1, Nvar)])
                else:
                    Y1_L, d_y1 = 0.0, {}
                # 3. Maximize E_1 (minimize -E_1)
                if Nvar >= 2:
                    c_e1 = np.zeros(2 * Nvar); c_e1[self._idx_e(1, Nvar)] = -1.0
                    sol_e1, d_e1 = self._solve_lp(c_e1, A_ub, b_ub_np, 2 * Nvar)
                    E1_U = float(sol_e1[self._idx_e(1, Nvar)])
                else:
                    E1_U, d_e1 = 0.0, {}
                # success for this attempt
                final_lp_diag.append({"attempt": label, "diag_y0": d_y0, "diag_y1": d_y1, "diag_e1": d_e1})
                # postprocess results
                if Y1_L < Y1_SAFE_THRESHOLD:
                    e1_U = 0.5
                    status_e1 = "OK_CONSERVATIVE_E1"
                else:
                    e1_U = min(0.5, E1_U / Y1_L)
                    status_e1 = "OK"
                return {
                    "Y0_L": Y0_L, "Y1_L": Y1_L, "e1_U": e1_U,
                    "status": status_e1, "ok": True,
                    "lp_diagnostics": {"attempts": final_lp_diag}
                }
            except LPFailureError as e:
                last_exc = e
                final_lp_diag.append({"attempt": label, "error": str(e)})
                logger.debug(f"LP attempt '{label}' failed: {e}")
                # continue to next relaxation attempt
                continue

        # If we reach here, all attempts failed -> fallback conservative (safe)
        logger.warning(f"All LP relaxation attempts failed - falling back to conservative decoy estimates. Last error: {last_exc}")
        return {
            "Y0_L": 0.0, "Y1_L": 0.0, "e1_U": 0.5,
            "status": "LP_INFEASIBLE_FALLBACK", "ok": True,
            "lp_diagnostics": {"attempts": final_lp_diag, "last_error": str(last_exc)}
        }

    @staticmethod
    def binary_entropy(p_err: float) -> float:
        if p_err <= 0.0 or p_err >= 1.0:
            return 0.0
        p = min(max(p_err, ENTROPY_PROB_CLAMP), 1.0 - ENTROPY_PROB_CLAMP)
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    def calculate_key_length(self, decoy_estimates: Dict[str, Any], signal_stats: TallyCounts) -> int:
        Y1_L = decoy_estimates["Y1_L"]
        e1_bit_U = decoy_estimates["e1_U"]
        N_sig_sent = signal_stats.sent
        if N_sig_sent == 0:
            return 0
        p_signal_config = self.p.get_pulse_config_by_name("signal")
        if not p_signal_config:
            return 0
        mu_s = p_signal_config.mean_photon_number
        p1_s = mu_s * math.exp(-mu_s)
        s_z_1_L = N_sig_sent * p1_s * Y1_L * (self.p.alice_z_basis_prob * self.p.bob_z_basis_prob)
        n_z = signal_stats.sifted_z
        m_z = signal_stats.errors_sifted_z
        if n_z == 0 or s_z_1_L <= 0:
            return 0
        qber_z = m_z / n_z if n_z > 0 else 0.0
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z
        e1_phase_U = 0.5
        if self.p.assume_phase_equals_bit_error:
            e1_phase_U = e1_bit_U
        else:
            if s_z_1_L > 0:
                delta = math.sqrt(math.log(2.0 / self.eps_alloc.eps_phase_est) / (2.0 * s_z_1_L))
                e1_phase_U = min(0.5, e1_bit_U + delta)
        pa_term_bits = 2 * (-math.log2(self.eps_alloc.eps_smooth)) + (-math.log2(self.eps_alloc.eps_pa))
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)
        key_length_float = (
            s_z_1_L * (1.0 - self.binary_entropy(e1_phase_U))
            - leak_EC
            - pa_term_bits
            - corr_term_bits
        )
        return max(0, math.floor(key_length_float))

# --- QKD System Implementation ---
class QKDSystem:
    def __init__(self, params: QKDParams, seed: Optional[int] = None):
        self.p = params
        if seed is None:
            seed = struct.unpack("Q", os.urandom(8))[0]
        self.master_seed_int = int(seed) % MAX_SEED_INT
        self.master_rng = np.random.default_rng(self.master_seed_int)
        if self.p.security_proof == SecurityProof.LIM_2014:
            self.proof_module = Lim2014Proof(self.p)
        else:
            raise NotImplementedError(f"Security proof {self.p.security_proof.value} not implemented.")

    def _alice_choices(self, num_pulses: int, rng: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alice_bits = rng.integers(0, 2, size=num_pulses, dtype=np.int8)
        alice_bases = rng.choice([0, 1], size=num_pulses, p=[self.p.alice_z_basis_prob, 1.0 - self.p.alice_z_basis_prob]).astype(np.int8)
        pulse_configs = self.p.pulse_configs
        probs = [pc.probability for pc in pulse_configs]
        alice_pulse_indices = rng.choice(len(pulse_configs), size=num_pulses, p=probs)
        mus = np.array([pc.mean_photon_number for pc in pulse_configs])
        pulse_mus = mus[alice_pulse_indices]
        photon_numbers_raw = rng.poisson(pulse_mus)
        return alice_bits, alice_bases, alice_pulse_indices, photon_numbers_raw

    def _channel_and_detection(self, photon_numbers: np.ndarray, rng: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eta = self.p.transmittance * self.p.det_eff
        if not (0.0 <= eta <= 1.0):
            raise ParameterValidationError(f"Overall detection efficiency eta must be in [0,1], but is {eta:.3f}")
        n = photon_numbers.astype(int)
        p_click_signal = 1.0 - np.power(1.0 - eta, n, where=(n >= 0))
        p_click_signal = np.clip(p_click_signal, 0.0, 1.0)
        signal_click = rng.random(len(n)) < p_click_signal
        dark0 = rng.random(len(n)) < self.p.dark_rate
        dark1 = rng.random(len(n)) < self.p.dark_rate
        detected_counts = signal_click.astype(int)
        return detected_counts, dark0, dark1

    def _sifting_and_errors(self, num_pulses: int,
                            alice_bits: np.ndarray, alice_bases: np.ndarray,
                            detected_counts: np.ndarray, dark0: np.ndarray, dark1: np.ndarray,
                            rng: Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bob_bases = rng.choice([0, 1], size=num_pulses, p=[self.p.bob_z_basis_prob, 1.0 - self.p.bob_z_basis_prob]).astype(np.int8)
        basis_match = (alice_bases == bob_bases)
        signal_counts_0 = np.zeros(num_pulses, dtype=bool)
        signal_counts_1 = np.zeros(num_pulses, dtype=bool)
        mis = self.p.misalignment
        idx_signal = np.nonzero(detected_counts > 0)[0]
        if idx_signal.size > 0:
            same_basis_mask = basis_match[idx_signal]
            diff_mask = ~same_basis_mask
            idx_same = idx_signal[same_basis_mask]
            if idx_same.size > 0:
                flips = rng.random(idx_same.size) < (1.0 - mis)
                alice_bits_same = alice_bits[idx_same]
                detector0 = (alice_bits_same == 0) & flips | (alice_bits_same == 1) & (~flips)
                detector1 = ~detector0
                signal_counts_0[idx_same] = detector0
                signal_counts_1[idx_same] = detector1
            idx_diff = idx_signal[diff_mask]
            if idx_diff.size > 0:
                pick0 = rng.random(idx_diff.size) < 0.5
                signal_counts_0[idx_diff] = pick0
                signal_counts_1[idx_diff] = ~pick0
        click0_final = signal_counts_0 | dark0
        click1_final = signal_counts_1 | dark1
        bob_bits = -1 * np.ones(num_pulses, dtype=np.int8)
        conclusive0 = click0_final & ~click1_final
        conclusive1 = click1_final & ~click0_final
        bob_bits[conclusive0] = 0
        bob_bits[conclusive1] = 1
        double_click_mask = click0_final & click1_final
        if self.p.double_click_policy == DoubleClickPolicy.RANDOM:
            num_dc = np.sum(double_click_mask)
            if num_dc > 0:
                bob_bits[double_click_mask] = rng.integers(0, 2, size=num_dc, dtype=np.int8)
        sifted_mask = basis_match & (bob_bits != -1)
        errors_mask = np.zeros(num_pulses, dtype=bool)
        errors_mask[sifted_mask] = (alice_bits[sifted_mask] != bob_bits[sifted_mask])
        if self.p.qber_intrinsic > 0 and np.any(sifted_mask):
            num_sifted = np.sum(sifted_mask)
            intrinsic_flips = rng.random(num_sifted) < self.p.qber_intrinsic
            errors_mask[sifted_mask] = np.logical_xor(errors_mask[sifted_mask], intrinsic_flips)
        discarded_dc_mask = basis_match & double_click_mask & (self.p.double_click_policy == DoubleClickPolicy.DISCARD)
        return sifted_mask, errors_mask, discarded_dc_mask, basis_match

    def _simulate_quantum_part_batch(self, num_pulses: int, rng: Generator) -> Dict:
        alice_bits, alice_bases, alice_pulse_indices, photon_numbers_raw = self._alice_choices(num_pulses, rng)
        detected_counts, dark0, dark1 = self._channel_and_detection(photon_numbers_raw, rng)
        sifted_mask, errors_mask, discarded_dc_mask, basis_match = self._sifting_and_errors(
            num_pulses, alice_bits, alice_bases, detected_counts, dark0, dark1, rng
        )
        batch_tallies = {}
        for i, pc in enumerate(self.p.pulse_configs):
            pulse_mask = (alice_pulse_indices == i)
            num_sent = int(np.sum(pulse_mask))
            sent_z = int(np.sum(pulse_mask & (alice_bases == 0)))
            sent_x = int(np.sum(pulse_mask & (alice_bases == 1)))
            sifted = int(np.sum(sifted_mask & pulse_mask))
            sifted_z = int(np.sum(sifted_mask & pulse_mask & (alice_bases == 0)))
            sifted_x = int(np.sum(sifted_mask & pulse_mask & (alice_bases == 1)))
            errors_sifted = int(np.sum(errors_mask & pulse_mask))
            errors_sifted_z = int(np.sum(errors_mask & pulse_mask & (alice_bases == 0)))
            errors_sifted_x = int(np.sum(errors_mask & pulse_mask & (alice_bases == 1)))
            double_discarded = int(np.sum(discarded_dc_mask & pulse_mask))
            batch_tallies[pc.name] = {
                "sent": num_sent,
                "sifted": sifted,
                "errors_sifted": errors_sifted,
                "double_clicks_discarded": double_discarded,
                "sent_z": sent_z,
                "sent_x": sent_x,
                "sifted_z": sifted_z,
                "sifted_x": sifted_x,
                "errors_sifted_z": errors_sifted_z,
                "errors_sifted_x": errors_sifted_x,
            }
        return {"overall": batch_tallies, "sifted_count": int(np.sum(sifted_mask))}

    def _merge_batch_tallies(self, overall_stats: Dict[str, TallyCounts], batch_result: Dict):
        for name, tally_dict in batch_result.get("overall", {}).items():
            if name not in overall_stats:
                overall_stats[name] = TallyCounts()
            for key in ["sent", "sifted", "errors_sifted", "double_clicks_discarded", "sent_z", "sent_x", "sifted_z", "sifted_x", "errors_sifted_z", "errors_sifted_x"]:
                val = int(tally_dict.get(key, 0))
                setattr(overall_stats[name], key, getattr(overall_stats[name], key) + val)

    def run_simulation(self) -> SimulationResults:
        start_time = time.time()
        if self.p.assume_phase_equals_bit_error:
            logger.warning("UNSAFE OPTION ENABLED: Assuming phase error equals bit error. Not secure for production.")
        total_pulses = self.p.num_bits
        batch_size = self.p.batch_size
        num_batches = (total_pulses + batch_size - 1) // batch_size
        use_mp = self.p.num_workers > 1 and num_batches > 1 and not self.p.force_sequential
        child_seeds = self.master_rng.integers(0, MAX_SEED_INT, size=num_batches)
        overall_stats = {pc.name: TallyCounts() for pc in self.p.pulse_configs}
        total_sifted = 0
        status = "OK"
        params_dict = _serialize_params_for_worker(self.p)
        tasks = [(params_dict, min(batch_size, total_pulses - i * batch_size), int(child_seeds[i])) for i in range(num_batches)]
        try:
            if use_mp:
                with ProcessPoolExecutor(max_workers=self.p.num_workers) as executor:
                    futures_map = {executor.submit(_top_level_worker_function, *task): task for task in tasks}
                    pbar = tqdm(as_completed(futures_map), total=len(tasks), desc="Simulating Batches (MP)")
                    for fut in pbar:
                        try:
                            batch_result = fut.result()
                        except Exception as e:
                            tb = traceback.format_exc()
                            logger.error(f"A worker failed: {e}\n{tb}")
                            status = f"WORKER_ERROR: {type(e).__name__}"
                            raise
                        self._merge_batch_tallies(overall_stats, batch_result)
                        total_sifted += batch_result["sifted_count"]
            else:
                piter = tqdm(tasks, desc="Simulating Batches (Seq)")
                for task in piter:
                    batch_result = _top_level_worker_function(*task)
                    self._merge_batch_tallies(overall_stats, batch_result)
                    total_sifted += batch_result["sifted_count"]
        except Exception as e:
            logger.exception("A worker process failed, aborting simulation.")
            status = f"WORKER_ERROR: {type(e).__name__}"

        elapsed_time = time.time() - start_time
        if status != "OK":
            return SimulationResults(params=self.p, metadata={}, status=status, simulation_time_seconds=elapsed_time)

        logger.info("--- Overall Observed Statistics ---")
        for name, stats in sorted(overall_stats.items()):
            gain = (stats.sifted / stats.sent) if stats.sent > 0 else 0.0
            qber = (stats.errors_sifted / stats.sifted) if stats.sifted > 0 else 0.0
            logger.info(f"Pulse: {name:<8} | Sent: {stats.sent:<10} | Sifted: {stats.sifted:<8} | Errors: {stats.errors_sifted:<6} | Gain: {gain:.3e} | QBER: {qber:.4f}")

        decoy_est, secure_len, cert = None, None, None
        try:
            decoy_est = self.proof_module.estimate_yields_and_errors(overall_stats)
            if not decoy_est.get("ok"):
                status = f"DECOY_ESTIMATION_FAILED: {decoy_est.get('status')}"
            else:
                secure_len = self.proof_module.calculate_key_length(decoy_est, overall_stats.get("signal", TallyCounts()))
                logger.info(f"Decoy Estimates: Y1_L={decoy_est['Y1_L']:.6g}, e1_U={decoy_est['e1_U']:.6g}")
                logger.info(f"Finite Secure Key Length (bits): {secure_len}")
                cert = SecurityCertificate(
                    proof_name=self.p.security_proof.value,
                    confidence_bound_method=self.p.ci_method.value,
                    assumed_phase_equals_bit_error=self.p.assume_phase_equals_bit_error,
                    epsilon_allocation=self.proof_module.eps_alloc,
                    lp_solver_diagnostics=decoy_est.get("lp_diagnostics"),
                )
        except (LPFailureError, ParameterValidationError) as e:
            status = f"DECOY_ESTIMATION_FAILED: {e}"
            logger.error(status, exc_info=True)

        metadata = {
            "version": "v16-corrected",
            "schema_version": "1.6",
            "timestamp_utc": time.time(),
            "master_seed": self.master_seed_int,
            "total_discarded_double_clicks": sum(s.double_clicks_discarded for s in overall_stats.values()),
        }
        return SimulationResults(
            params=self.p, metadata=metadata, security_certificate=cert,
            decoy_estimates=decoy_est, secure_key_length=secure_len,
            raw_sifted_key_length=total_sifted, simulation_time_seconds=elapsed_time,
            status=status,
        )

# --- Plotting Helper (lazy imports) ---
def plot_skl_vs_parameter(param_values: List, skl_values: List, param_name: str, **kwargs):
    global PLOTTING_AVAILABLE, plt, sns
    try:
        if not PLOTTING_AVAILABLE:
            import matplotlib
            if not os.environ.get("DISPLAY") and "pytest" not in sys.modules:
                matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            import seaborn as _sns
            plt, sns = _plt, _sns
            PLOTTING_AVAILABLE = True
    except ImportError:
        logger.warning("Plotting libraries not available, skipping plot generation.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    with sns.axes_style("whitegrid"):
        p_vals = np.array(param_values)
        s_vals = np.array(skl_values)
        valid_mask = np.isfinite(s_vals) & (s_vals > 0)
        if not np.any(valid_mask):
            logger.warning(f"No valid SKL data (>0) to plot for {param_name}.")
            plt.close(fig)
            return
        ax.plot(p_vals[valid_mask], s_vals[valid_mask], marker="o", linestyle="-")
        ax.set_xlabel(f"{param_name} ({kwargs.get('param_unit', '')})")
        ax.set_ylabel("Secure Key Length (bits)")
        ax.set_title(kwargs.get("title", f"Secure Key Length vs. {param_name}"))
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--")
        plt.tight_layout()
        if path := kwargs.get("output_path"):
            try:
                dir_path = os.path.dirname(path)
                if dir_path: os.makedirs(dir_path, exist_ok=True)
                plt.savefig(path, dpi=300)
                logger.info(f"Plot saved to {path}")
            except IOError as e:
                logger.error(f"Failed to save plot to {path}: {e}")
        else:
            plt.show()
    plt.close(fig)

# --- CLI & Main Execution ---
def create_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rigorous Finite-Key QKD Simulation (v16 revised).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument("--num_bits", type=int, default=1_000_000_000)
    sim_group.add_argument("--seed", type=int, default=None)
    channel_group = parser.add_argument_group("Channel and Detector Parameters")
    channel_group.add_argument("--distance_km", type=float, default=50.0)
    channel_group.add_argument("--fiber_loss_db_km", type=float, default=0.2)
    channel_group.add_argument("--det_eff", type=float, default=0.20)
    channel_group.add_argument("--dark_rate", type=float, default=1e-7)
    channel_group.add_argument("--qber_intrinsic", type=float, default=0.01)
    channel_group.add_argument("--misalignment", type=float, default=0.015)
    protocol_group = parser.add_argument_group("Protocol Parameters")
    protocol_group.add_argument("--bob_z_basis_prob", type=float, default=0.5)
    protocol_group.add_argument("--alice_z_basis_prob", type=float, default=0.5)
    protocol_group.add_argument("--double_click_policy", type=str, default=DoubleClickPolicy.DISCARD.value, choices=[p.value for p in DoubleClickPolicy])
    protocol_group.add_argument("--mu_signal", type=float, default=0.5)
    protocol_group.add_argument("--mu_decoy", type=float, default=0.1)
    protocol_group.add_argument("--mu_vacuum", type=float, default=0.0)
    protocol_group.add_argument("--p_signal", type=float, default=0.7)
    protocol_group.add_argument("--p_decoy", type=float, default=0.15)
    protocol_group.add_argument("--p_vacuum", type=float, default=0.15)
    analysis_group = parser.add_argument_group("Finite-Key Analysis Parameters")
    analysis_group.add_argument("--security_proof", type=str, default=SecurityProof.LIM_2014.value, choices=[p.value for p in SecurityProof])
    analysis_group.add_argument("--ci_method", type=str, default=ConfidenceBoundMethod.CLOPPER_PEARSON.value, choices=[m.value for m in ConfidenceBoundMethod])
    analysis_group.add_argument("--f_error_correction", type=float, default=1.1)
    analysis_group.add_argument("--eps_sec", type=float, default=1e-9)
    analysis_group.add_argument("--eps_cor", type=float, default=1e-15)
    analysis_group.add_argument("--eps_pe", type=float, default=1e-10)
    analysis_group.add_argument("--eps_smooth", type=float, default=1e-10)
    analysis_group.add_argument("--photon_number_cap", type=int, default=12)
    analysis_group.add_argument("--no_monotonicity", dest="enforce_monotonicity", action="store_false")
    analysis_group.add_argument("--assume_phase_equals_bit_error", action="store_true")
    exec_group = parser.add_argument_group("Execution and Output")
    exec_group.add_argument("--num_workers", type=int, default=(os.cpu_count() or 1))
    exec_group.add_argument("--batch_size", type=int, default=200_000)
    exec_group.add_argument("--force_sequential", action="store_true")
    exec_group.add_argument("--output_results_json", type=str, default=None)
    exec_group.add_argument("--plot_skl_vs_distance", action="store_true")
    exec_group.add_argument("--plot_output_dir", type=str, default=".")
    exec_group.add_argument("--run_tests", action="store_true")
    parser.set_defaults(enforce_monotonicity=True, assume_phase_equals_bit_error=False)
    return parser

def main():
    parser = create_cli_parser()
    args = parser.parse_args()
    if args.run_tests:
        print("Run tests via pytest or python -m unittest discover. Exiting.")
        return
    pulse_configs = [
        PulseTypeConfig("signal", args.mu_signal, args.p_signal),
        PulseTypeConfig("decoy", args.mu_decoy, args.p_decoy),
        PulseTypeConfig("vacuum", args.mu_vacuum, args.p_vacuum),
    ]
    qkd_param_fields = {f.name for f in dataclasses.fields(QKDParams)}
    params_dict = {k: v for k, v in vars(args).items() if k in qkd_param_fields}
    params_dict.pop('security_proof', None)
    params_dict.pop('ci_method', None)
    params_dict.pop('double_click_policy', None)
    try:
        qkd_params = QKDParams(
            **params_dict,
            pulse_configs=pulse_configs,
            security_proof=SecurityProof(args.security_proof),
            ci_method=ConfidenceBoundMethod(args.ci_method),
            double_click_policy=DoubleClickPolicy(args.double_click_policy)
        )
    except ParameterValidationError as e:
        parser.error(str(e))
    if args.plot_skl_vs_distance:
        distances = np.linspace(10, 150, 15)
        skl_outputs = []
        master_rng = np.random.default_rng(args.seed)
        child_seeds = master_rng.integers(0, MAX_SEED_INT, size=len(distances))
        for i, dist in enumerate(tqdm(distances, desc="Sweeping distance")):
            current_params = dataclasses.replace(qkd_params, distance_km=float(dist))
            qkd_system = QKDSystem(current_params, seed=int(child_seeds[i]))
            sim_results = qkd_system.run_simulation()
            key_len = sim_results.secure_key_length if sim_results.secure_key_length is not None else 0
            skl_outputs.append(key_len)
        plot_path = os.path.join(args.plot_output_dir, "skl_vs_distance.png")
        plot_skl_vs_parameter(distances.tolist(), skl_outputs, "Distance", param_unit="km", title="Secure Key Length vs. Distance", output_path=plot_path)
    else:
        qkd_system = QKDSystem(qkd_params, seed=args.seed)
        results = qkd_system.run_simulation()
        if args.output_results_json:
            results.save_json(args.output_results_json)
        else:
            print(json.dumps(results.to_serializable_dict(), indent=2))

if __name__ == "__main__":
    main()
