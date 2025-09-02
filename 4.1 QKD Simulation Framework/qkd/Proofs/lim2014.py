# -*- coding: utf-8 -*-
"""
Implementation of the Lim et al. 2014 finite-key security proof.

Moved from the monolithic script. No logic changes beyond bugfixes.
"""
import logging
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from .base import FiniteKeyProof
from .utils_lp import solve_lp
from ..datatypes import TallyCounts
from ..exceptions import (
    LPFailureError,
    ParameterValidationError,
    QKDSimulationError,
)
from ..sources import p_n_mu_vector
from ..constants import S_Z_1_L_MIN_FOR_PHASE_EST, Y1_SAFE_THRESHOLD

__all__ = ["Lim2014Proof"]

logger = logging.getLogger(__name__)


class Lim2014Proof(FiniteKeyProof):
    """Implements the finite-key security proof from Lim et al., PRA 89, 032332 (2014)."""

    def allocate_epsilons(self) -> "EpsilonAllocation":
        from ..datatypes import EpsilonAllocation
        n_intensities = len(self.p.source.pulse_configs)
        total_tests = 4 * n_intensities + 1
        eps_pe_total = self.p.eps_pe
        eps_per_test = eps_pe_total / max(1, total_tests)

        # Robust allocation
        eps_budget_for_pa = self.p.eps_sec - self.p.eps_cor - eps_pe_total - (2 * self.p.eps_smooth)
        if eps_budget_for_pa <= 0:
            raise ParameterValidationError(f"Insecure epsilon allocation: eps_sec ({self.p.eps_sec:.2e}) is too small for other epsilons.")

        return EpsilonAllocation(
            eps_sec=self.p.eps_sec,
            eps_cor=self.p.eps_cor,
            eps_pe=eps_pe_total,
            eps_smooth=self.p.eps_smooth,
            eps_pa=eps_budget_for_pa,
            eps_phase_est=eps_per_test,
        )

    def _idx_y(self, n: int, Nvar: int) -> int:
        return n

    def _idx_e(self, n: int, Nvar: int) -> int:
        return Nvar + n

    def _build_constraints(
        self,
        required: List[str],
        stats_map: Dict[str, TallyCounts],
        use_basis_z: bool,
        enforce_monotonicity: bool,
        enforce_half_error: bool,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        cap = self.p.photon_number_cap
        Nvar = cap + 1
        rows, cols, data, b_ub_list = [], [], [], []

        def _add_row(ridx, coeffs: Dict[int, float], rhs: float):
            for var_idx, coeff in coeffs.items():
                rows.append(ridx)
                cols.append(var_idx)
                data.append(coeff)
            b_ub_list.append(rhs)

        row_idx = 0
        pulse_map = {pc.name: pc for pc in self.p.source.pulse_configs}
        eps_per_ci = self.eps_alloc.eps_pe / max(1, 4 * len(required))

        for name in required:
            stats = stats_map[name]
            sent = stats.sent_z if use_basis_z else stats.sent
            sifted = stats.sifted_z if use_basis_z else stats.sifted
            errors = stats.errors_sifted_z if use_basis_z else stats.errors_sifted

            q_l, q_u = self.get_bounds(sifted, sent, eps_per_ci)
            r_l, r_u = self.get_bounds(errors, sent, eps_per_ci)
            p_vec = p_n_mu_vector(pulse_map[name].mean_photon_number, cap, self.p.require_tail_below)

            _add_row(row_idx, {self._idx_y(i, Nvar): p_vec[i] for i in range(Nvar)}, q_u)
            row_idx += 1
            _add_row(row_idx, {self._idx_y(i, Nvar): -p_vec[i] for i in range(Nvar)}, -q_l)
            row_idx += 1
            _add_row(row_idx, {self._idx_e(i, Nvar): p_vec[i] for i in range(Nvar)}, r_u)
            row_idx += 1
            _add_row(row_idx, {self._idx_e(i, Nvar): -p_vec[i] for i in range(Nvar)}, -r_l)
            row_idx += 1

        for n in range(Nvar):
            _add_row(row_idx, {self._idx_e(n, Nvar): 1.0, self._idx_y(n, Nvar): -1.0}, 0.0)
            row_idx += 1
        if enforce_half_error:
            for n in range(Nvar):
                _add_row(row_idx, {self._idx_e(n, Nvar): 1.0, self._idx_y(n, Nvar): -0.5}, 0.0)
                row_idx += 1

        # Corrected monotonicity constraint loop
        if enforce_monotonicity and Nvar >= 2:
            for n in range(Nvar - 1):
                _add_row(row_idx, {self._idx_y(n + 1, Nvar): 1.0, self._idx_y(n, Nvar): -1.0}, 0.0)
                row_idx += 1

        A_coo = coo_matrix((data, (rows, cols)), shape=(row_idx, 2 * Nvar))
        return A_coo.tocsr(), np.array(b_ub_list, dtype=float), Nvar

    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> Dict[str, any]:
        required = [pc.name for pc in self.p.source.pulse_configs]
        try_sequence = [
            {"use_basis_z": True, "enforce_monotonicity": self.p.enforce_monotonicity, "enforce_half_error": True, "label": "Z_mon_half"},
            {"use_basis_z": True, "enforce_monotonicity": False, "enforce_half_error": True, "label": "Z_noMon_half"},
        ]
        last_exc, final_lp_diag = None, []

        for attempt_config in try_sequence:
            try:
                A_ub, b_ub, Nvar = self._build_constraints(required, stats_map, **{k: v for k, v in attempt_config.items() if k != "label"})
                c_y1 = np.zeros(2 * Nvar)
                c_e1 = np.zeros(2 * Nvar)
                if Nvar >= 2:
                    c_y1[self._idx_y(1, Nvar)] = -1.0
                    c_e1[self._idx_e(1, Nvar)] = -1.0

                sol_y1, diag_y1 = solve_lp(c_y1, A_ub, b_ub, 2 * Nvar, self.p.lp_solver_method)
                sol_e1, diag_e1 = solve_lp(c_e1, A_ub, b_ub, 2 * Nvar, self.p.lp_solver_method)

                Y1_L = float(sol_y1[self._idx_y(1, Nvar)]) if Nvar >= 2 else 0.0
                E1_U = float(sol_e1[self._idx_e(1, Nvar)]) if Nvar >= 2 else 0.0
                final_lp_diag.append({"attempt": attempt_config["label"], "diag_y1": diag_y1, "diag_e1": diag_e1})

                ok = True
                if Y1_L <= Y1_SAFE_THRESHOLD:
                    logger.warning(f"Y1_L ({Y1_L:.2e}) is below safe threshold. Using conservative e1_U=0.5 and marking estimate as not 'ok'.")
                    e1_U = 0.5
                    ok = False
                else:
                    e1_U = min(0.5, E1_U / Y1_L)

                return {"Y1_L": Y1_L, "e1_U": e1_U, "ok": ok, "lp_diagnostics": {"attempts": final_lp_diag}}
            except LPFailureError as e:
                last_exc = e
                logger.debug(f"LP attempt '{attempt_config['label']}' failed definitively: {e}")
                if not any(d.get("attempt") == attempt_config["label"] for d in final_lp_diag):
                    final_lp_diag.append({"attempt": attempt_config["label"], "error": str(e)})

        logger.error(f"All LP attempts failed, falling back to conservative estimate. Last error: {last_exc}")
        return {"Y1_L": 0.0, "e1_U": 0.5, "ok": False, "status": "LP_INFEASIBLE_FALLBACK", "lp_diagnostics": {"attempts": final_lp_diag}}

    def calculate_key_length(self, decoy_estimates: Dict[str, any], stats_map: Dict[str, TallyCounts]) -> int:
        Y1_L, e1_bit_U = decoy_estimates["Y1_L"], decoy_estimates["e1_U"]
        signal_stats = stats_map.get("signal")
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not signal_stats or signal_stats.sent == 0:
            return 0

        alice_z_prob = getattr(self.p.protocol, "alice_z_basis_prob", 0.0)
        bob_z_prob = getattr(self.p.protocol, "bob_z_basis_prob", 0.0)

        mu_s = p_sig_cfg.mean_photon_number
        p1_s = mu_s * math.exp(-mu_s)
        s_z_1_L = signal_stats.sent * p1_s * Y1_L * (alice_z_prob * bob_z_prob)
        n_z, m_z = signal_stats.sifted_z, signal_stats.errors_sifted_z

        if n_z <= 0 or s_z_1_L < S_Z_1_L_MIN_FOR_PHASE_EST:
            return 0

        qber_z = m_z / n_z
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z

        if self.p.assume_phase_equals_bit_error:
            e1_phase_U = e1_bit_U
        else:
            try:
                delta = math.sqrt(math.log(2.0 / self.eps_alloc.eps_phase_est) / (2.0 * s_z_1_L))
            except (ValueError, ZeroDivisionError):
                raise QKDSimulationError("Invalid value in phase error delta calculation.")
            e1_phase_U = min(0.5, e1_bit_U + delta)

        pa_term_bits = 2 * (-math.log2(self.eps_alloc.eps_smooth)) + (-math.log2(self.eps_alloc.eps_pa))
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)

        key_length_float = s_z_1_L * (1.0 - self.binary_entropy(e1_phase_U)) - leak_EC - pa_term_bits - corr_term_bits

        return max(0, math.floor(key_length_float))
