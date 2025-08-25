# -*- coding: utf-8 -*-
"""
Implementation of the Lim et al. 2014 finite-key security proof.

This module provides a robust, audited, and numerically stable implementation
of the decoy-state QKD security proof from:
- Lim, C. C. W., et al. "Concise security proof for practical decoy-state
  quantum key distribution." Physical Review A 89.3 (2014): 032332.
  DOI: 10.1103/PhysRevA.89.032332

This implementation incorporates extensive improvements focusing on:
1.  **Correctness & Safety:** Rigorous validation of inputs, parameters, and
    intermediate results.
2.  **Numerical Stability:** Use of safe mathematical operations, log-space
    calculations, LP row-normalization, and consistent floating-point precision.
3.  **Auditability:** Detailed, structured diagnostics for every step,
    including LP solver outputs, epsilon allocation, and statistical bounds.
4.  **API Consistency:** Adherence to a strict data-class-based API for inputs
    and outputs, ensuring interoperability and maintainability.

Key Parameters:
- cap (photon_number_cap): Typically in the range [10, 30]. Larger values
  increase LP complexity. A hard limit can be set via `p.max_photon_vars`.
- mu (mean_photon_number): For signal/decoy states, typically in [0, 1].
- num_pulses: Should be large enough for statistical significance, e.g., > 1e9.

LP Variable Indexing Scheme:
- y_n (n-photon yield): indices [0, ..., cap]
- e_n (n-photon error term): indices [cap+1, ..., 2*cap+1]

Interpreting Outputs:
- Y1_L: The lower bound on the yield of single-photon states, a crucial
  parameter for key rate calculation. This is a conditional probability.
- e1_U: The upper bound on the bit error rate of single-photon states.
"""
import hashlib
import hmac
import json
import math
import time
import traceback
import gc
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix, vstack

# Corrected imports to resolve type conflicts and attribute errors.
from .base import (
    FiniteKeyProof,
    DecoyEstimates,
    KeyCalculationResult,
    SolverDiagnostics,
    ErrorCode,
    _json_safe_dict,
)
from ..datatypes import TallyCounts as DetailedTallyCounts, EpsilonAllocation as DetailedEpsilonAllocation
from .utils_lp import solve_lp
from ..exceptions import (
    LPFailureError,
    ParameterValidationError,
)
from ..sources import p_n_mu_vector
# Corrected and added constant imports based on `constants.py`
from ..constants import (
    MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST,
    Y1_SAFE_THRESHOLD,
    NUMERIC_ABS_TOL,
    DEFAULT_POISSON_TAIL_THRESHOLD,
)

if TYPE_CHECKING:
    from ..params import QKDParams
    from ..sources import PoissonSource

__all__ = ["Lim2014Proof"]

# Module-level constants for clarity and auditability
MAX_ERROR_RATE = 0.5
MIN_NONZERO = 1e-300


@lru_cache(maxsize=32)
def _cached_p_n_mu_vector(mu: float, cap: int, tail: float) -> tuple:
    """Cached wrapper for the photon number distribution vector."""
    return tuple(p_n_mu_vector(mu, cap, tail_threshold=tail))


class Lim2014Proof(FiniteKeyProof):
    """
    Implements the finite-key security proof from Lim et al., PRA 89, 032332 (2014).
    """
    # This attribute intentionally uses a more specific type than the base class.
    p: "QKDParams"  # type: ignore[assignment]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = args[0]
        self.supported_solvers = {'highs', 'interior-point', 'simplex'}
        # The eps_alloc from the base class init needs to be the detailed one.
        self.eps_alloc: DetailedEpsilonAllocation = self.allocate_epsilons() # type: ignore[assignment]

    def _safe_log2(self, x: float) -> float:
        """Calculates log base 2 of x, clamping small values for safety."""
        if x < MIN_NONZERO:
            self.logger.warning("Log2 argument %e is smaller than safe limit, clamping.", x)
            x = MIN_NONZERO
        return math.log2(x)

    def get_epsilon_policy(self) -> Callable[[DetailedEpsilonAllocation], float]: # type: ignore[override]
        return lambda eps: eps.eps_pe

    def allocate_epsilons(self) -> DetailedEpsilonAllocation: # type: ignore[override]
        """
        Allocates security epsilons.
        NOTE: The return type is more specific than the one in the base class.
        """
        pulse_configs = self.p.source.pulse_configs
        if not pulse_configs:
            raise ParameterValidationError(f"[{self.run_id}] Pulse configurations cannot be empty.")

        n_intensities = len(pulse_configs)
        total_tests = 4 * n_intensities + 1
        eps_pe_total = self.p.eps_pe

        eps_per_test = float(self._safe_divide(eps_pe_total, total_tests, name="eps_per_test"))
        
        eps_budget_for_pa = (self.p.eps_sec - self.p.eps_cor - eps_pe_total - (2 * self.p.eps_smooth))

        if eps_budget_for_pa <= 0 and self.is_audit_mode():
            raise ParameterValidationError(f"Insecure epsilon allocation: PA budget is non-positive ({eps_budget_for_pa:.2e}).")

        return DetailedEpsilonAllocation(
            eps_sec=self.p.eps_sec, eps_cor=self.p.eps_cor, eps_pe=eps_pe_total,
            eps_smooth=self.p.eps_smooth, eps_pa=self._clamp_nonneg(eps_budget_for_pa, name='eps_pa'),
            eps_phase_est=eps_per_test,
        )

    def _idx_y(self, n: int) -> int: return n
    def _idx_e(self, n: int, Nvar: int) -> int: return Nvar + n

    def _get_sanitized_p_vec(self, mu: float, cap: int) -> np.ndarray:
        """Computes, validates, sanitizes, and trims the photon number distribution vector."""
        tail_threshold = self.p.require_tail_below or DEFAULT_POISSON_TAIL_THRESHOLD
        p_vec = np.array(_cached_p_n_mu_vector(mu, cap, tail_threshold), dtype=np.float64)
        
        # Normalize in log space to avoid underflow
        with np.errstate(divide='ignore'): log_p = np.log(p_vec)
        if np.any(np.isfinite(log_p)):
            p_vec = np.exp(log_p - np.max(log_p[np.isfinite(log_p)]))
        
        s = p_vec.sum()
        if s <= MIN_NONZERO:
            p_vec = np.zeros(cap + 1, dtype=np.float64); p_vec[0] = 1.0
            return p_vec

        p_vec /= s # Use direct numpy division

        if tail_threshold > 0:
            keep_idx = int(np.searchsorted(np.cumsum(p_vec), 1 - tail_threshold) + 1)
            p_vec = p_vec[:keep_idx]
            p_vec /= p_vec.sum() # Renormalize after trimming
        return p_vec

    def _build_lp_components(self, required_pulses: List[str], stats_map: Dict[str, DetailedTallyCounts]) -> Tuple[csr_matrix, np.ndarray, int, Dict[str, Any]]:
        """Helper to assemble basis-independent LP components."""
        cap = self.p.photon_number_cap
        Nvar = cap + 1
        rows, cols, data, b_ub_list = [], [], [], []
        constraint_diagnostics: Dict[str, Any] = {'row_names': []}
        row_idx = 0
        pulse_map = {pc.name: pc for pc in self.p.source.pulse_configs}

        eps_per_ci = self.get_epsilon_policy()(self.eps_alloc) / max(1, 4 * len(required_pulses))

        for name in required_pulses:
            stats = stats_map[name]
            q_l, q_u = self.get_bounds(int(stats.sifted_z), int(stats.sent_z), eps_per_ci)
            r_l, r_u = self.get_bounds(int(stats.errors_sifted_z), int(stats.sifted_z), eps_per_ci)
            p_vec = self._get_sanitized_p_vec(pulse_map[name].mean_photon_number, cap)
            
            def _add_row(ridx, coeffs, rhs, name):
                for var_idx, coeff in coeffs.items():
                    rows.append(ridx); cols.append(var_idx); data.append(coeff)
                b_ub_list.append(rhs)
                constraint_diagnostics['row_names'].append(name)
            
            _add_row(row_idx, {self._idx_y(i): p for i, p in enumerate(p_vec)}, q_u, f"{name}_yield_U"); row_idx += 1
            _add_row(row_idx, {self._idx_y(i): -p for i, p in enumerate(p_vec)}, -q_l, f"{name}_yield_L"); row_idx += 1
            _add_row(row_idx, {self._idx_e(i, Nvar): p for i, p in enumerate(p_vec)}, r_u, f"{name}_error_U"); row_idx += 1
            _add_row(row_idx, {self._idx_e(i, Nvar): -p for i, p in enumerate(p_vec)}, -r_l, f"{name}_error_L"); row_idx += 1

        for n in range(Nvar):
            _add_row(row_idx, {self._idx_e(n, Nvar): 1.0, self._idx_y(n): -1.0}, 0.0, f"e{n}<=y{n}"); row_idx += 1
        
        A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(row_idx, 2 * Nvar)).tocsr()
        b = np.array(b_ub_list, dtype=np.float64)
        return A, b, Nvar, constraint_diagnostics

    def _solve_single_lp(self, cost_vector: np.ndarray, A_ub: csr_matrix, b_ub: np.ndarray, name: str, method: str) -> Tuple[np.ndarray, SolverDiagnostics]:
        """Wrapper to solve a single LP and package the results."""
        solution, diagnostics = solve_lp(cost_vector, A_ub, b_ub, A_ub.shape[1], method)
        # If solve_lp fails, it raises an exception, so we can assume success here.
        return solution, SolverDiagnostics(
            is_success=True,
            solver_name=diagnostics.get("method", "unknown"),
            status_message=diagnostics.get("message", "N/A"),
            residual_norm=diagnostics.get("max_violation", -1.0)
        )

    def estimate_yields_and_errors(self, stats_map: Dict[str, DetailedTallyCounts]) -> DecoyEstimates: # type: ignore[override]
        """Estimates Y1_L and e1_U using linear programming."""
        self.validate_stats_map(stats_map, [pc.name for pc in self.p.source.pulse_configs]) # type: ignore[arg-type]

        A, b, Nvar, _ = self._build_lp_components([pc.name for pc in self.p.source.pulse_configs], stats_map)
        
        try:
            c_y1 = np.zeros(2 * Nvar); c_y1[self._idx_y(1)] = 1.0
            c_e1 = np.zeros(2 * Nvar); c_e1[self._idx_e(1, Nvar)] = -1.0
            
            sol_y1, diag_y1 = self._solve_single_lp(c_y1, A, b, "Y1_L", self.p.lp_solver_method)
            sol_e1, diag_e1 = self._solve_single_lp(c_e1, A, b, "e1_U", self.p.lp_solver_method)

            Y1_L = sol_y1[self._idx_y(1)]
            E1_U_num = -sol_e1.dot(c_e1)

            is_feasible = Y1_L > Y1_SAFE_THRESHOLD
            e1_U = self._safe_divide(E1_U_num, Y1_L, default=MAX_ERROR_RATE) if is_feasible else MAX_ERROR_RATE

            return DecoyEstimates(
                yield_1_lower_bound=Y1_L, error_rate_1_upper_bound=e1_U,
                is_feasible=is_feasible, failure_prob_used=self.eps_alloc.eps_pe,
                diagnostics=diag_y1,
            )
        except LPFailureError as e:
            self.logger.error(f"LP Failure: {e}")
            # The `diag` attribute is part of the custom exception, so we ignore the mypy error.
            return self._conservative_decoy_estimate(self.eps_alloc.eps_pe, e.diag) # type: ignore[attr-defined]

    def calculate_key_length(self, decoy_estimates: DecoyEstimates, stats_map: Dict[str, DetailedTallyCounts]) -> KeyCalculationResult: # type: ignore[override]
        """Calculates the final secure key length."""
        signal_stats = stats_map["signal"]
        mu_signal = self.p.source.get_pulse_config_by_name("signal").mean_photon_number
        # p1 (probability of single photon) is calculated, not an attribute of the source.
        p1_signal = mu_signal * math.exp(-mu_signal)
        
        s_z_1_L = p1_signal * signal_stats.sent_z * decoy_estimates.yield_1_lower_bound
        n_z = float(signal_stats.sifted_z)
        
        if n_z <= 0 or s_z_1_L < MIN_SUCCESSFUL_Z1_BASIS_EVENTS_FOR_PHASE_EST:
            return KeyCalculationResult(0, 0, 0, 0, error_codes=[ErrorCode.INSUFFICIENT_STATISTICS])

        qber_z = self._safe_divide(signal_stats.errors_sifted_z, n_z)
        leak_EC = float(self.p.f_error_correction * self.binary_entropy(qber_z) * n_z)
        e1_phase_U = self._compute_phase_error(decoy_estimates.error_rate_1_upper_bound, s_z_1_L)
        pa_term, corr_term = self._compute_finite_key_terms()

        key_len = s_z_1_L * (1 - self.binary_entropy(e1_phase_U)) - leak_EC - pa_term - corr_term
        
        return KeyCalculationResult(
            secure_key_length=self._clamp_key_length(float(key_len)),
            privacy_amplification_term=pa_term,
            error_correction_leakage=leak_EC,
            phase_error_rate_upper_bound=e1_phase_U,
        )

    def _compute_phase_error(self, e1_bit_U: float, s_z_1_L: float) -> float:
        """Computes the phase error upper bound."""
        if self.p.assume_phase_equals_bit_error: return e1_bit_U
        
        eps = self.eps_alloc.eps_phase_est
        delta = math.sqrt(self._safe_log(2 / eps) / (2 * s_z_1_L))
        return self._clamp_nonneg(e1_bit_U + delta, name='e1_phase_U')

    def _compute_finite_key_terms(self) -> Tuple[float, float]:
        """Computes privacy amplification and correctness terms."""
        pa_term = 2 * self._safe_log2(1 / self.eps_alloc.eps_smooth) + self._safe_log2(1 / self.eps_alloc.eps_pa)
        corr_term = self._safe_log2(2 / self.eps_alloc.eps_cor)
        return pa_term, corr_term

    def _get_initial_diagnostics(self, stats_map: Dict[str, DetailedTallyCounts]) -> Dict:
        """Prepares the initial diagnostics dictionary for a run."""
        sanitized_pulses = [pc.to_dict() for pc in self.p.source.pulse_configs]
        p_cfg_json = json.dumps(sanitized_pulses, sort_keys=True).encode()
        s_map_json = json.dumps({k: v.to_dict() for k, v in stats_map.items()}, sort_keys=True).encode()
        
        return {
            "repro_run_id": self.run_id,
            "solver_versions": {'scipy': scipy.__version__},
            "pulse_configs_sha256": hashlib.sha256(p_cfg_json).hexdigest(),
            "stats_map_sha256": hashlib.sha256(s_map_json).hexdigest(),
        }

    def notation_map(self) -> Dict[str, str]:
        return {
            "Y_1^L": "yield_1_lower_bound",
            "e_ph": "phase_error_rate_upper_bound",
            "s_z_1^L": "The lower bound on the number of single-photon events in the Z basis."
        }

