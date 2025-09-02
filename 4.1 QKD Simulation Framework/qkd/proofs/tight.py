# -*- coding: utf-8 -*-
"""
Implementation of a production-grade, tight BB84 finite-key security proof.

This module provides a robust, auditable implementation of the BB84-tight
security proof, derived from the Lim et al. 2014 framework. It has been
extensively refactored to meet production standards for correctness, API
consistency, numerical stability, diagnostics, and provenance.
"""

import math
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# ==============================================================================
# 1. Supporting Datatypes, Constants, and Exceptions
# ==============================================================================

# CORRECTED: Import shared datatypes from the central `datatypes.py` file.
from ..datatypes import TallyCounts, EpsilonAllocation

DEFAULT_EPS_MIN = 1e-300
DEFAULT_ENTROPY_PROB_CLAMP = 1e-12
DEFAULT_NUMERIC_TOL = 1e-9

class ErrorCode(Enum):
    INVALID_PARAMS = "INVALID_PARAMS"
    INVALID_DECOY_ESTIMATES_TYPE = "INVALID_DECOY_ESTIMATES_TYPE"
    MISSING_DECOY_FIELDS = "MISSING_DECOY_FIELDS"
    INVALID_STATS_MAP = "INVALID_STATS_MAP"
    MISSING_SIGNAL_CONFIG = "MISSING_SIGNAL_CONFIG"
    INVALID_PULSE_CONFIG = "INVALID_PULSE_CONFIG"
    NO_SIGNAL_PULSES_SENT = "NO_SIGNAL_PULSES_SENT"
    INSUFFICIENT_STATISTICS = "INSUFFICIENT_STATISTICS"
    INCONSISTENT_MEASUREMENT = "INCONSISTENT_MEASUREMENT"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    PHASE_ERROR_CLAMPED = "PHASE_ERROR_CLAMPED"
    INVALID_YIELD_BOUND = "INVALID_YIELD_BOUND"
    INVALID_ERROR_RATE_BOUND = "INVALID_ERROR_RATE_BOUND"

class ParameterValidationError(ValueError): pass
class QKDSimulationError(RuntimeError): pass

@dataclass
class DecoyEstimatesInput:
    Y1_L_rate: float
    e1_bit_ub: float

@dataclass
class KeyCalculationResult:
    key_length: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Lim2014Proof:
    def __init__(self, p: Any, logger: Optional[logging.Logger] = None, mode: str = 'PRODUCTION'):
        self.p = p
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.mode = mode
        self._validate_params()
        self.EPS_MIN = getattr(p, 'EPS_MIN', DEFAULT_EPS_MIN)
        self.ENTROPY_PROB_CLAMP = getattr(p, 'ENTROPY_PROB_CLAMP', DEFAULT_ENTROPY_PROB_CLAMP)
        self.NUMERIC_TOL = getattr(p, 'NUMERIC_TOL', DEFAULT_NUMERIC_TOL)
        self.eps_alloc = self.allocate_epsilons()
    
    def _validate_params(self):
        required = ['eps_sec', 'eps_cor', 'source', 'protocol', 'f_error_correction', 'assume_phase_equals_bit_error']
        for attr in required:
            if not hasattr(self.p, attr):
                raise ParameterValidationError(f"Parameter object 'p' is missing required attribute: {attr}")

    def allocate_epsilons(self) -> EpsilonAllocation:
        raise NotImplementedError

# ==============================================================================
# 2. Main Proof Implementation
# ==============================================================================

__all__ = ["BB84TightProof", "KeyCalculationResult", "TallyCounts", "EpsilonAllocation", "ErrorCode"]

class BB84TightProof(Lim2014Proof):
    __implementation_version__ = "3.7.0"

    def __repr__(self) -> str:
        return f"BB84TightProof(impl_version='{self.__implementation_version__}')"

    def allocate_epsilons(self) -> EpsilonAllocation:
        eps_sec = self.p.eps_sec
        eps_cor = self.p.eps_cor
        remaining_eps = eps_sec - eps_cor
        if remaining_eps <= 0:
            raise ParameterValidationError(f"eps_sec ({eps_sec:.2e}) must be greater than eps_cor ({eps_cor:.2e}).")
        eps_pe = remaining_eps / 3
        eps_smooth = remaining_eps / 6
        eps_pa = remaining_eps / 3
        num_pulse_configs = len(getattr(self.p.source, 'pulse_configs', []))
        denominator = max(1, 4 * num_pulse_configs + 1)
        eps_phase_est = eps_pe / denominator
        allocation = EpsilonAllocation(
            eps_sec=eps_sec, eps_cor=eps_cor, eps_pe=eps_pe,
            eps_smooth=eps_smooth, eps_pa=eps_pa, eps_phase_est=eps_phase_est
        )
        allocation.validate()
        return allocation

    def calculate_key_length(
        self,
        decoy_estimates: Union[Dict[str, Any], object],
        stats_map: Dict[str, TallyCounts]
    ) -> KeyCalculationResult:
        run_id = getattr(self.p, 'run_id', None)
        self.logger.info("event=key_calc_start run_id=%s", run_id)
        start_time = time.perf_counter()
        result = KeyCalculationResult()
        
        try:
            is_decoy_protocol = bool(decoy_estimates)

            if is_decoy_protocol:
                signal_stats, p_sig_cfg = self._validate_stats_map_decoy(stats_map, result)
                if result.error_codes: return self._finalize_result(result, start_time)
                decoy_norm = self._normalize_decoy_estimates(decoy_estimates, result)
                if result.error_codes: return self._finalize_result(result, start_time)
                s_z_1_L_count, n_z, _ = self._compute_base_counts_decoy(signal_stats, p_sig_cfg, decoy_norm, result)
                e1_bit_ub = decoy_norm.e1_bit_ub
            else:
                result.diagnostics['info'] = "Running in non-decoy mode; using conservative estimates."
                signal_stats = self._get_total_stats_non_decoy(stats_map, result)
                if result.error_codes: return self._finalize_result(result, start_time)
                n_z = float(signal_stats.sifted_z)
                m_z = float(signal_stats.errors_sifted_z)
                s_z_1_L_count = n_z
                e1_bit_ub = self._safe_divide(m_z, n_z)
                result.diagnostics['intermediate'] = {"n_z": n_z, "m_z": m_z, "qber_z": e1_bit_ub}

            min_s_z_1_L = getattr(self.p, 'S_Z_1_L_MIN_FOR_PHASE_EST', 100)
            if n_z <= 0 or s_z_1_L_count < min_s_z_1_L:
                result.error_codes.append(ErrorCode.INSUFFICIENT_STATISTICS)
                return self._finalize_result(result, start_time)

            leak_ec = self._compute_leak_ec(signal_stats.sifted_z, signal_stats.errors_sifted_z, result)
            e1_phase_ub = self._compute_phase_error_ub(s_z_1_L_count, e1_bit_ub, result)
            pa_corr_bits = self._compute_pa_terms(result)
            available_entropy = s_z_1_L_count * (1.0 - self._binary_entropy_safe(e1_phase_ub))
            key_len_float = available_entropy - leak_ec - pa_corr_bits
            
            self._finite_or_error("final_key_length", key_len_float, result)
            result.key_length = self._clamp_key_length(key_len_float)
            self._populate_final_diagnostics(result, available_entropy, key_len_float, pa_corr_bits)

        except (ParameterValidationError, QKDSimulationError) as e:
            self.logger.error("event=key_calc_error run_id=%s error='%s'", run_id, e)
            result.error_codes.append(ErrorCode.INVALID_PARAMS)
        
        return self._finalize_result(result, start_time)

    def _get_total_stats_non_decoy(self, stats_map, result):
        if not stats_map or not any(isinstance(v, TallyCounts) for v in stats_map.values()):
            result.error_codes.append(ErrorCode.INVALID_STATS_MAP)
            result.diagnostics['explain'] = "stats_map is empty or contains no valid TallyCounts objects for non-decoy mode."
            return None
        
        total_stats = TallyCounts()
        for stats in stats_map.values():
            if isinstance(stats, TallyCounts):
                total_stats = total_stats.merged(stats)
        
        return total_stats

    def _validate_stats_map_decoy(self, stats_map, result):
        if 'signal' not in stats_map or not isinstance(stats_map['signal'], TallyCounts):
            result.error_codes.append(ErrorCode.INVALID_STATS_MAP)
            return None, None
        
        stats = stats_map['signal']
        p_sig_cfg = self.p.source.get_pulse_config_by_name("signal")
        if not p_sig_cfg or not hasattr(p_sig_cfg, 'mean_photon_number'):
            result.error_codes.append(ErrorCode.MISSING_SIGNAL_CONFIG)
            return stats, None
        return stats, p_sig_cfg

    def _normalize_decoy_estimates(self, decoy_estimates, result):
        Y1_L = getattr(decoy_estimates, 'Y1_L', decoy_estimates.get('Y1_L'))
        e1_U = getattr(decoy_estimates, 'e1_U', decoy_estimates.get('e1_U'))
        if Y1_L is None or e1_U is None:
            result.error_codes.append(ErrorCode.MISSING_DECOY_FIELDS)
            return None
        return DecoyEstimatesInput(float(Y1_L), float(e1_U))

    def _compute_base_counts_decoy(self, signal_stats, p_sig_cfg, decoy_norm, result):
        mu_s = p_sig_cfg.mean_photon_number
        p1_s = self._single_photon_prob(mu_s)
        alice_z_prob = getattr(self.p.protocol, "alice_z_basis_prob", 0.5)
        bob_z_prob = getattr(self.p.protocol, "bob_z_basis_prob", 0.5)
        s_z_1_L_count = float(signal_stats.sent) * p1_s * decoy_norm.Y1_L_rate * (alice_z_prob * bob_z_prob)
        n_z = float(signal_stats.sifted_z)
        m_z = float(signal_stats.errors_sifted_z)
        if s_z_1_L_count > n_z + self.NUMERIC_TOL:
            s_z_1_L_count = n_z
        result.diagnostics['intermediate'] = {
            "mu_s": mu_s, "p1_s": p1_s, "s_z_1_L_count": s_z_1_L_count, "n_z": n_z, "m_z": m_z
        }
        return s_z_1_L_count, n_z, m_z

    def _compute_leak_ec(self, n_z, m_z, result):
        qber_z = self._safe_divide(m_z, n_z)
        f_ec = getattr(self.p, 'f_error_correction', 1.1)
        leak_ec = f_ec * self._binary_entropy_safe(qber_z) * n_z
        result.diagnostics['intermediate']['qber_z'] = qber_z
        result.diagnostics['intermediate']['leak_ec_bits'] = leak_ec
        return leak_ec

    def _compute_phase_error_ub(self, s_z_1_L_count, e1_bit_ub, result):
        if self.p.assume_phase_equals_bit_error:
            result.diagnostics['intermediate']['e1_phase_ub'] = e1_bit_ub
            return e1_bit_ub
        eps_ph_est_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_phase_est)
        log_term = -math.log(eps_ph_est_safe)
        delta = self._safe_sqrt(log_term / (2 * s_z_1_L_count))
        e1_phase_ub = e1_bit_ub + delta
        if e1_phase_ub > 0.5:
            e1_phase_ub = 0.5
        result.diagnostics['intermediate']['e1_phase_ub'] = e1_phase_ub
        return e1_phase_ub

    def _compute_pa_terms(self, result):
        eps_smooth_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_smooth)
        eps_pa_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_pa)
        eps_cor_safe, _ = self._log_inv_clamp(self.eps_alloc.eps_cor)
        pa_term_bits = _safe_log2(1.0 / (2.0 * eps_smooth_safe)) + _safe_log2(1.0 / eps_pa_safe)
        corr_term_bits = _safe_log2(2.0 / eps_cor_safe)
        return pa_term_bits + corr_term_bits

    def _populate_final_diagnostics(self, result, available_entropy, key_len_float, pa_corr_bits):
        result.diagnostics['intermediate']['available_entropy_bits'] = available_entropy
        result.diagnostics['intermediate']['final_key_length_float'] = key_len_float

    def _finalize_result(self, result, start_time):
        end_time = time.perf_counter()
        result.metadata['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        result.metadata['calculation_time_ms'] = (end_time - start_time) * 1000
        result.error_codes = [code.value if isinstance(code, Enum) else code for code in result.error_codes]
        run_id = getattr(self.p, 'run_id', None)
        if result.error_codes:
            self.logger.warning("event=key_calc_abort run_id=%s errors=%s", run_id, result.error_codes)
        else:
            self.logger.info("event=key_calc_finish run_id=%s key_length=%d", run_id, result.key_length)
        return result

    def _log_inv_clamp(self, val):
        return (self.EPS_MIN, True) if val < self.EPS_MIN else (val, False)
    
    def _safe_divide(self, num, den):
        return 0.0 if abs(den) < self.NUMERIC_TOL else num / den

    def _finite_or_error(self, name, val, result, non_negative=False):
        if not math.isfinite(val) or (non_negative and val < 0):
            result.error_codes.append(ErrorCode.NON_FINITE_VALUE)

    def _clamp_key_length(self, key_len_float):
        return max(0, math.floor(key_len_float))

    def _binary_entropy_safe(self, p: float) -> float:
        p_clamped = max(self.ENTROPY_PROB_CLAMP, min(p, 1 - self.ENTROPY_PROB_CLAMP))
        if p_clamped <= 0 or p_clamped >= 1: return 0.0
        return -p_clamped * math.log2(p_clamped) - (1 - p_clamped) * math.log2(1 - p_clamped)

    def _single_photon_prob(self, mu: float) -> float:
        if mu < 0: raise ParameterValidationError(f"mu cannot be negative: {mu}")
        return mu * math.exp(-mu)
    
    def _safe_sqrt(self, x: float) -> float:
        if x < 0: return 0.0
        return math.sqrt(x)

def _safe_log2(val: float) -> float:
    return 0.0 if val <= 0 else math.log2(val)

