# -*- coding: utf-8 -*-
"""
Implementation of the MDI-QKD finite-key security proof (placeholder).

Moved from the monolithic script. This version explicitly disables key calculation
by default due to the lack of a proper 2D decoy-state analysis.
"""
import logging
import math
from typing import Dict

from .lim2014 import Lim2014Proof
from ..datatypes import TallyCounts
from ..exceptions import QKDSimulationError

__all__ = ["MDIQKDProof"]

logger = logging.getLogger(__name__)


class MDIQKDProof(Lim2014Proof):
    """
    Implements a finite-key security proof for MDI-QKD.
    WARNING: This is a placeholder and is NOT SCIENTIFICALLY VALID.
    """

    def calculate_key_length(self, decoy_estimates: Dict[str, any], stats_map: Dict[str, TallyCounts]) -> int:
        """
        Calculates the secure key length for MDI-QKD. This is an unsafe placeholder.
        """
        if not self.p.allow_unsafe_mdi_approx:
            logger.error(
                "MDI-QKD key rate calculation aborted. The current implementation lacks a "
                "valid 2D decoy-state analysis and is not secure. To override, set "
                "the 'allow_unsafe_mdi_approx' parameter to true in your config file."
            )
            raise QKDSimulationError("Unsafe MDI-QKD approximation not allowed.")

        logger.warning(
            "MDIQKDProof is using a simplified placeholder key rate formula adapted from BB84. "
            "The results are NOT secure or scientifically valid."
        )

        Y11_L, e11_x_U = decoy_estimates["Y1_L"], decoy_estimates["e1_U"]
        signal_stats_A = stats_map.get("signal")
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not signal_stats_A or signal_stats_A.sent == 0:
            return 0

        n_z, n_x = signal_stats_A.sifted_z, signal_stats_A.sifted_x
        m_x = signal_stats_A.errors_sifted_x
        if n_z <= 0 or n_x <= 0:
            return 0

        s_z_11_L = n_z * Y11_L
        m_x_U_bound, _ = self.get_bounds(m_x, n_x, self.eps_alloc.eps_pe)
        e_ph_11_U = m_x_U_bound

        qber_z = signal_stats_A.errors_sifted_z / n_z
        leak_EC = self.p.f_error_correction * self.binary_entropy(qber_z) * n_z

        pa_term_bits = 7 * math.log2(21 / self.eps_alloc.eps_smooth) + 2 * math.log2(1 / self.eps_alloc.eps_pa)
        corr_term_bits = math.log2(2.0 / self.eps_alloc.eps_cor)

        key_length_float = s_z_11_L * (1.0 - self.binary_entropy(e_ph_11_U)) - leak_EC - pa_term_bits - corr_term_bits
        return max(0, math.floor(key_length_float))
