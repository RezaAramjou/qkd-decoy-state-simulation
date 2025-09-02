# -*- coding: utf-8 -*-
"""
Unit Test Suite for the corrected QKD Simulation
"""
import unittest
import numpy as np
import dataclasses
import math
import traceback
from qkd_simulation_main import (
    QKDParams,
    QKDSystem,
    Lim2014Proof,
    PulseTypeConfig,
    TallyCounts,
    EpsilonAllocation,
    ParameterValidationError,
    DoubleClickPolicy,
    SecurityProof,
    ConfidenceBoundMethod,
    p_n_mu_vector,
    clopper_pearson_bounds
)

class TestQKDSimulation(unittest.TestCase):
    def setUp(self):
        self.default_params_dict = {
            'num_bits': 10000, 'distance_km': 10.0, 'fiber_loss_db_km': 0.2,
            'det_eff': 0.2, 'dark_rate': 1e-8, 'qber_intrinsic': 0.01,
            'misalignment': 0.015, 'bob_z_basis_prob': 0.5, 'f_error_correction': 1.1,
            'eps_sec': 1e-9, 'eps_cor': 1e-15, 'eps_pe': 1e-10, 'eps_smooth': 1e-10,
            'photon_number_cap': 10, 'batch_size': 5000, 'num_workers': 1,
            'force_sequential': True, 'enforce_monotonicity': True,
            'assume_phase_equals_bit_error': False,
            'pulse_configs': [
                PulseTypeConfig("signal", 0.5, 0.7),
                PulseTypeConfig("decoy", 0.1, 0.15),
                PulseTypeConfig("vacuum", 0.0, 0.15)
            ],
            'double_click_policy': DoubleClickPolicy.DISCARD,
            'security_proof': SecurityProof.LIM_2014,
            'ci_method': ConfidenceBoundMethod.CLOPPER_PEARSON,
            'alice_z_basis_prob': 0.5,
        }

    def test_p_n_mu_vector_normalization(self):
        vec = p_n_mu_vector(mu=0.5, n_cap=10)
        self.assertEqual(len(vec), 11)
        self.assertAlmostEqual(float(np.sum(vec)), 1.0, places=9)
        from scipy.stats import poisson
        self.assertAlmostEqual(vec[-1], float(poisson.sf(9, 0.5)), places=9)

    def test_clopper_pearson_edges(self):
        lower, upper = clopper_pearson_bounds(k=0, n=100, failure_prob=1e-6)
        self.assertEqual(lower, 0.0)
        self.assertTrue(0 < upper < 1.0)
        lower, upper = clopper_pearson_bounds(k=100, n=100, failure_prob=1e-6)
        self.assertEqual(upper, 1.0)
        self.assertTrue(0 < lower < 1.0)

    def test_binary_entropy(self):
        self.assertEqual(Lim2014Proof.binary_entropy(0.0), 0.0)
        self.assertEqual(Lim2014Proof.binary_entropy(1.0), 0.0)
        self.assertAlmostEqual(Lim2014Proof.binary_entropy(0.5), 1.0)
        p = 0.11
        expected_h_011 = -p * math.log2(p) - (1-p) * math.log2(1-p)
        self.assertAlmostEqual(Lim2014Proof.binary_entropy(0.11), expected_h_011, places=6)

    def test_epsilon_allocation_valid(self):
        params = QKDParams(**self.default_params_dict)
        proof = Lim2014Proof(params)
        self.assertIsInstance(proof.eps_alloc, EpsilonAllocation)

    def test_epsilon_allocation_invalid_raises(self):
        invalid_params_dict = self.default_params_dict.copy()
        invalid_params_dict['eps_sec'] = 1e-20
        params = QKDParams(**invalid_params_dict)
        with self.assertRaises(ParameterValidationError):
            Lim2014Proof(params)

    def test_alice_z_basis_factor_in_s_z_1_L(self):
        params_high = {**self.default_params_dict}
        params_high['alice_z_basis_prob'] = 1.0
        params_low = {**self.default_params_dict}
        params_low['alice_z_basis_prob'] = 0.5
        p_high = QKDParams(**params_high)
        p_low = QKDParams(**params_low)
        proof_high = Lim2014Proof(p_high)
        proof_low = Lim2014Proof(p_low)
        decoy_est = {"Y1_L": 0.1, "e1_U": 0.05}
        sig_stats_high = TallyCounts(sent=1000, sifted=200, errors_sifted=10, sifted_z=120, errors_sifted_z=6)
        sig_stats_low = TallyCounts(sent=1000, sifted=200, errors_sifted=10, sifted_z=60, errors_sifted_z=3)
        kl_high = proof_high.calculate_key_length(decoy_est, sig_stats_high)
        kl_low = proof_low.calculate_key_length(decoy_est, sig_stats_low)
        self.assertGreaterEqual(kl_high, kl_low)

    def test_vectorized_sifting_runs(self):
        params = QKDParams(**self.default_params_dict)
        system = QKDSystem(params, seed=123)
        rng = np.random.default_rng(123)
        res = system._simulate_quantum_part_batch(1000, rng)
        self.assertIn("overall", res)
        self.assertIsInstance(res.get("sifted_count"), int)
        self.assertGreaterEqual(res['sifted_count'], 0)

    def test_reproducibility(self):
        params = QKDParams(**self.default_params_dict)
        system1 = QKDSystem(params, seed=42)
        res1 = system1.run_simulation()
        system2 = QKDSystem(params, seed=42)
        res2 = system2.run_simulation()
        self.assertEqual(res1.status, "OK", "First simulation run failed")
        self.assertEqual(res2.status, "OK", "Second simulation run failed")
        self.assertEqual(res1.raw_sifted_key_length, res2.raw_sifted_key_length)
        self.assertEqual(res1.secure_key_length, res2.secure_key_length)
        self.assertIsNotNone(res1.decoy_estimates)
        self.assertIsNotNone(res2.decoy_estimates)
        self.assertTrue(res1.decoy_estimates.get('ok', False))
        self.assertTrue(res2.decoy_estimates.get('ok', False))
        self.assertAlmostEqual(res1.decoy_estimates['Y1_L'], res2.decoy_estimates['Y1_L'], places=8)
        self.assertAlmostEqual(res1.decoy_estimates['e1_U'], res2.decoy_estimates['e1_U'], places=8)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
