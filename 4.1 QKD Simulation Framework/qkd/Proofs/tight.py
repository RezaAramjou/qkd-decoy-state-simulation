# -*- coding: utf-8 -*-
"""
Implementation of a tighter BB84 finite-key security proof.

Moved from the monolithic script. No logic changes.
"""
import math
from typing import Dict

from .lim2014 import Lim2014Proof
from ..datatypes import TallyCounts
from ..exceptions import ParameterValidationError, QKDSimulationError

__all__ = ["BB84TightProof"]


class BB84TightProof(Lim2014Proof):
    """Implements a tighter finite-key security proof for BB84 decoy-state QKD."""

    def allocate_epsilons(self) -> "EpsilonAllocation":
        from ..datatypes import EpsilonAllocation
        """A more balanced allocation of epsilons."""
        eps_sec = self.p.eps_sec
        eps_cor = self.p.eps_cor

        remaining_eps = eps_sec - eps_cor
        if remaining_eps <= 0:
            raise ParameterValidationError(f"eps_sec ({eps_sec:.2e}) must be greater than eps_cor ({eps_cor:.2e}).")

        eps_pe = remaining_eps / 3
        eps_smooth = remaining_eps / 6
        eps_pa = remaining_eps / 3
        eps_phase_est = eps_pe / (4 * len(self.p.source.pulse_configs) + 1)

        return EpsilonAllocation(
            eps_sec=eps_sec,
            eps_cor=eps_cor,
            eps_pe=eps_pe,
            eps_smooth=eps_smooth,
            eps_pa=eps_pa,
            eps_phase_est=eps_phase_est,
        )

    def calculate_key_length(self, decoy_estimates: Dict[str, any], stats_map: Dict[str, TallyCounts]) -> int:
        """Calculates the secure key length using a tighter formula for privacy amplification."""
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

        if n_z <= 0 or s_z_1_L < self.p.S_Z_1_L_MIN_FOR_PHASE_EST:
            return 0

        qber_z = m_z / n_z
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z

        if self.p.assume_phase_equals_bit_error:
            e1_phase_U = e1_bit_U
        else:
            try:
                delta = math.sqrt(math.log(1.0 / self.eps_alloc.eps_phase_est) / (s_z_1_L))
            except (ValueError, ZeroDivisionError):
                raise QKDSimulationError("Invalid value in phase error delta calculation.")
            e1_phase_U = min(0.5, e1_bit_U + delta)

        pa_term_bits = 2 * math.log2(1.0 / (2 * self.eps_alloc.eps_smooth)) + math.log2(1.0 / self.eps_alloc.eps_pa)
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)

        key_length_float = s_z_1_L * (1.0 - self.binary_entropy(e1_phase_U)) - leak_EC - pa_term_bits - corr_term_bits

        return max(0, math.floor(key_length_float))
