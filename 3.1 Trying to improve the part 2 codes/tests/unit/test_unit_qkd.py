# -*- coding: utf-8 -*-
"""
Unit Test Suite for the corrected QKD Simulation (v17.8)
This version adds the remaining test cases from the review for a more
comprehensive and robust suite, and fixes the final mock error.
"""
import pytest
pytest.importorskip("scipy")

import numpy as np
import math
from unittest.mock import patch, MagicMock
import json

# Import the entire module to help with robust patching
import qkd_simulation_main

# Import specific components from the main simulation file
from qkd_simulation_main import (
    QKDParams, QKDSystem, Lim2014Proof, PulseTypeConfig, TallyCounts,
    EpsilonAllocation, ParameterValidationError, LPFailureError, DoubleClickPolicy,
    SecurityProof, ConfidenceBoundMethod, p_n_mu_vector, clopper_pearson_bounds,
    hoeffding_bounds, ENTROPY_PROB_CLAMP, SimulationResults
)

# --- Test Fixtures ---

@pytest.fixture
def default_params_dict():
    """Provides a default, valid set of parameters for tests."""
    return {
        'num_bits': 10000, 'distance_km': 10.0, 'fiber_loss_db_km': 0.2,
        'det_eff': 0.2, 'dark_rate': 1e-8, 'qber_intrinsic': 0.01,
        'misalignment': 0.015, 'bob_z_basis_prob': 0.5, 'f_error_correction': 1.1,
        'eps_sec': 1e-9, 'eps_cor': 1e-15, 'eps_pe': 1e-10, 'eps_smooth': 1e-10,
        'photon_number_cap': 10, 'batch_size': 1000, 'num_workers': 1,
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

@pytest.fixture
def small_params(default_params_dict):
    """A smaller version of parameters for faster unit tests."""
    params = default_params_dict.copy()
    params['num_bits'] = 1000
    params['batch_size'] = 100
    return QKDParams(**params)

# --- A. Test Design & Engineering Tests ---

class TestParameterValidation:
    """Tests for QKDParams validation logic."""
    def test_valid_params_ok(self, default_params_dict):
        try:
            QKDParams(**default_params_dict)
        except ParameterValidationError:
            pytest.fail("Valid default parameters raised ParameterValidationError unexpectedly.")

    @pytest.mark.parametrize("param,invalid_value,expected_msg", [
        ("num_bits", 0, "num_bits must be positive."),
        ("dark_rate", 1.1, "dark_rate must be in \\[0, 1\\)."),
        ("f_error_correction", 0.9, "f_error_correction must be in \\[1.0, 5.0\\]."),
        ("alice_z_basis_prob", 1.1, "alice_z_basis_prob must be in \\[0,1\\]."),
        ("photon_number_cap", 0, "photon_number_cap must be >= 1."),
    ])
    def test_invalid_params_raise_error(self, default_params_dict, param, invalid_value, expected_msg):
        invalid_params = default_params_dict.copy()
        invalid_params[param] = invalid_value
        with pytest.raises(ParameterValidationError, match=expected_msg):
            QKDParams(**invalid_params)

# --- B. Concrete Test Content Gaps ---

class TestStatisticalHelpers:
    """Tests for standalone statistical utility functions."""
    @patch.object(qkd_simulation_main, 'logger')
    def test_p_n_mu_vector_renormalization_warning(self, mock_logger):
        with patch('qkd_simulation_main.poisson.pmf', return_value=np.array([0.9])):
            with patch('qkd_simulation_main.poisson.sf', return_value=0.3):
                 p_n_mu_vector(mu=1, n_cap=1)
                 mock_logger.warning.assert_called_once()
                 assert "deviates by" in mock_logger.warning.call_args[0][0]

    def test_clopper_pearson_properties(self):
        k, n = 10, 100
        lower, upper = clopper_pearson_bounds(k, n, 0.05)
        assert 0 <= lower <= k/n <= upper <= 1
        assert clopper_pearson_bounds(0, n, 0.05)[0] == 0.0
        assert clopper_pearson_bounds(n, n, 0.05)[1] == 1.0

    @pytest.mark.parametrize("p, expected_h", [
        (0.0, 0.0), (1.0, 0.0), (0.5, 1.0),
        (0.11, -0.11 * math.log2(0.11) - 0.89 * math.log2(0.89)),
    ])
    def test_binary_entropy(self, p, expected_h):
        h = Lim2014Proof.binary_entropy(p)
        assert np.isclose(h, expected_h, atol=1e-9)

    def test_binary_entropy_clamp(self):
        p_clamped = ENTROPY_PROB_CLAMP
        h_clamped = Lim2014Proof.binary_entropy(p_clamped)
        h_below_clamp = Lim2014Proof.binary_entropy(p_clamped / 2)
        assert h_clamped > 0
        assert math.isclose(h_clamped, h_below_clamp)

# --- C. Scientific / Modeling Correctness ---

class TestSiftingLogic:
    """Dedicated tests for the corrected sifting and error model."""
    def test_misalignment_is_deterministic_with_mock_rng(self, small_params):
        num_pulses = 20
        params_dict = small_params.to_dict()
        params_dict['misalignment'] = 0.1
        params_dict['qber_intrinsic'] = 0.0
        params = QKDParams.from_dict(params_dict)
        
        real_rng = np.random.default_rng(42)
        alice_bits = real_rng.integers(0, 2, size=num_pulses)
        alice_bases = np.zeros(num_pulses, dtype=np.int8)
        signal_click = np.ones(num_pulses, dtype=bool)
        dark0, dark1 = np.zeros_like(signal_click), np.zeros_like(signal_click)

        mock_rng = MagicMock(spec=np.random.Generator)
        mock_rng.choice.return_value = np.zeros(num_pulses, dtype=np.int8)
        mock_rng.random.return_value = np.linspace(0, 1, num_pulses)

        sifted, errors, _ = qkd_simulation_main._sifting_and_errors(
            params, num_pulses, alice_bits, alice_bases,
            signal_click, dark0, dark1, mock_rng
        )
        
        assert np.sum(errors) == 2

class TestEpsilonAllocation:
    """Tests for the EpsilonAllocation logic."""
    def test_valid_allocation(self, small_params):
        proof = Lim2014Proof(small_params)
        eps = proof.eps_alloc
        assert isinstance(eps, EpsilonAllocation)
        assert eps.eps_sec == small_params.eps_sec
        assert eps.eps_pe + 2 * eps.eps_smooth + eps.eps_pa <= eps.eps_sec + 1e-12

    def test_insecure_allocation_raises_error(self, default_params_dict):
        params_dict = default_params_dict.copy()
        params_dict['eps_sec'] = 1e-20
        params = QKDParams(**params_dict)
        with pytest.raises(ParameterValidationError, match="Insecure epsilon allocation"):
            Lim2014Proof(params)

# --- D. Numerical & Robustness ---

class TestLPWrapper:
    """Tests the robustness of the LP solver wrapper and decoy estimation."""
    @patch.object(qkd_simulation_main, 'linprog')
    def test_lp_failure_raises_exception(self, mock_linprog, small_params):
        mock_linprog.return_value = MagicMock(success=False, message="Test Failure")
        proof = Lim2014Proof(small_params)
        A_ub, b_ub, cost = np.array([[1, 1]]), np.array([1]), np.array([1, 1])
        with pytest.raises(LPFailureError, match="LP solver failed: Test Failure"):
            proof._solve_lp(cost, A_ub, b_ub, 2)

    def test_estimate_yields_with_zero_counts(self, small_params):
        """Tests that the system behaves conservatively with zero sifted bits."""
        proof = Lim2014Proof(small_params)
        stats_map = {pc.name: TallyCounts(sent=1000) for pc in small_params.pulse_configs}
        results = proof.estimate_yields_and_errors(stats_map)
        assert results['ok']
        assert results['e1_U'] == 0.5

    @patch.object(qkd_simulation_main.Lim2014Proof, '_solve_lp')
    def test_lp_relaxation_fallback(self, mock_solve_lp, small_params):
        n_vars = 2 * (small_params.photon_number_cap + 1)
        sol_success = np.zeros(n_vars)

        mock_solve_lp.side_effect = [
            (sol_success.copy(), {}),
            LPFailureError("Fail E1 on attempt 1"),
            LPFailureError("Fail Y1 on attempt 2"),
            (sol_success.copy(), {}),
            (sol_success.copy(), {}),
        ]
        
        proof = Lim2014Proof(small_params)
        stats_map = {pc.name: TallyCounts(sent=1000, sent_z=500, sifted_z=50, errors_sifted_z=5) for pc in small_params.pulse_configs}
        
        results = proof.estimate_yields_and_errors(stats_map)
        
        assert results['ok']
        assert mock_solve_lp.call_count == 5
        diagnostics = results['lp_diagnostics']['attempts']
        assert "error" in diagnostics[0]
        assert "error" in diagnostics[1]
        assert "error" not in diagnostics[2]

# --- E. API, Contract & I/O Tests ---

class TestSerialization:
    """Tests for data serialization and deserialization."""
    def test_qkdparams_roundtrip(self, default_params_dict):
        params1 = QKDParams(**default_params_dict)
        params_dict = params1.to_dict()
        params2 = QKDParams.from_dict(params_dict)
        assert params1 == params2

    def test_simulationresults_to_serializable(self, small_params):
        results = SimulationResults(
            params=small_params,
            metadata={"master_seed": np.int64(42)},
            secure_key_length=np.int64(12345),
            decoy_estimates={"Y1_L": np.float64(0.123), "e1_U": 0.5}
        )
        serializable_dict = results.to_serializable_dict()
        assert isinstance(serializable_dict['metadata']['master_seed'], int)
        assert isinstance(serializable_dict['secure_key_length'], int)
        assert isinstance(serializable_dict['decoy_estimates']['Y1_L'], float)
        try:
            json.dumps(serializable_dict)
        except TypeError:
            pytest.fail("to_serializable_dict() output is not JSON serializable.")

    @patch('builtins.open')
    @patch('os.replace')
    @patch('os.makedirs')
    @patch.object(qkd_simulation_main, 'logger')
    def test_save_json_io_error(self, mock_logger, mock_makedirs, mock_replace, mock_open, small_params):
        """Tests that save_json logs an error if file I/O fails."""
        mock_open.side_effect = IOError("Permission denied")
        results = SimulationResults(params=small_params, metadata={})
        results.save_json("non_existent_dir/results.json")
        mock_logger.error.assert_called_once()
        assert "Failed to save results" in mock_logger.error.call_args[0][0]
