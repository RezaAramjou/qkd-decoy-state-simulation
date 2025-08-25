# -*- coding: utf-8 -*-
"""
Integration Test Suite for the corrected QKD Simulation (v17.0)
Focuses on end-to-end simulation runs, checking for reproducibility and
overall system behavior. These tests are slower and marked accordingly.
"""
import pytest
import numpy as np
from qkd_simulation_main_corrected import (
    QKDParams, QKDSystem, PulseTypeConfig, DoubleClickPolicy,
    SecurityProof, ConfidenceBoundMethod
)

# --- Test Fixtures ---
@pytest.fixture
def integration_params_dict():
    """Provides a small but complete set of parameters for integration tests."""
    return {
        'num_bits': 20000, 'distance_km': 20.0, 'fiber_loss_db_km': 0.2,
        'det_eff': 0.2, 'dark_rate': 1e-8, 'qber_intrinsic': 0.01,
        'misalignment': 0.015, 'bob_z_basis_prob': 0.5, 'f_error_correction': 1.1,
        'eps_sec': 1e-9, 'eps_cor': 1e-15, 'eps_pe': 1e-10, 'eps_smooth': 1e-10,
        'photon_number_cap': 12, 'batch_size': 5000, 'num_workers': 1,
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

# --- Integration Tests ---

@pytest.mark.integration
class TestFullSimulation:
    """
    Contains slow, end-to-end tests that run the full QKD simulation.
    These tests verify reproducibility and overall system integrity.
    """

    def test_reproducibility_with_seed(self, integration_params_dict):
        """
        Tests that two simulation runs with the same seed produce identical results.
        This is a critical test for deterministic behavior.
        """
        params = QKDParams(**integration_params_dict)
        
        # Run simulation the first time
        system1 = QKDSystem(params, seed=42)
        res1 = system1.run_simulation()

        # Run simulation the second time with the same seed
        system2 = QKDSystem(params, seed=42)
        res2 = system2.run_simulation()

        # --- Assertions ---
        # Check status first
        assert res1.status == "OK", f"First simulation run failed with status: {res1.status}"
        assert res2.status == "OK", f"Second simulation run failed with status: {res2.status}"

        # Check integer counts for exact equality
        assert res1.raw_sifted_key_length == res2.raw_sifted_key_length
        assert res1.secure_key_length == res2.secure_key_length

        # Check decoy estimates for near equality (floats)
        assert res1.decoy_estimates is not None
        assert res2.decoy_estimates is not None
        assert res1.decoy_estimates.get('ok', False)
        assert res2.decoy_estimates.get('ok', False)
        
        # Use numpy's testing utilities for robust float comparison
        np.testing.assert_allclose(
            res1.decoy_estimates['Y1_L'],
            res2.decoy_estimates['Y1_L'],
            rtol=1e-8, atol=1e-12,
            err_msg="Y1_L estimates are not close enough for reproducibility."
        )
        np.testing.assert_allclose(
            res1.decoy_estimates['e1_U'],
            res2.decoy_estimates['e1_U'],
            rtol=1e-8, atol=1e-12,
            err_msg="e1_U estimates are not close enough for reproducibility."
        )

    def test_simulation_runs_without_crashing(self, integration_params_dict):
        """
        A basic sanity check to ensure a standard simulation run completes
        with an 'OK' status and produces a non-negative key length.
        """
        params = QKDParams(**integration_params_dict)
        system = QKDSystem(params, seed=123)
        results = system.run_simulation()

        assert results.status == "OK", f"Simulation failed with status: {results.status}"
        assert results.secure_key_length is not None
        assert results.secure_key_length >= 0
        assert results.raw_sifted_key_length > 0

    def test_alice_z_basis_prob_impact(self, integration_params_dict):
        """
        Tests that increasing Alice's Z-basis probability (for the signal state)
        leads to a higher secure key length, all else being equal.
        """
        params_dict_low = integration_params_dict.copy()
        params_dict_low['alice_z_basis_prob'] = 0.5
        params_low = QKDParams(**params_dict_low)

        params_dict_high = integration_params_dict.copy()
        params_dict_high['alice_z_basis_prob'] = 0.9
        params_high = QKDParams(**params_dict_high)

        # Use the same seed to ensure channel conditions are identical
        seed = 99
        
        system_low = QKDSystem(params_low, seed=seed)
        res_low = system_low.run_simulation()

        system_high = QKDSystem(params_high, seed=seed)
        res_high = system_high.run_simulation()

        assert res_low.status == "OK"
        assert res_high.status == "OK"
        
        # A higher allocation to the Z basis for key generation should result in more secure key
        assert res_high.secure_key_length >= res_low.secure_key_length


if __name__ == '__main__':
    # This allows running the integration tests directly, e.g.,
    # python test_integration_qkd.py
    pytest.main(['-m', 'integration', '-v'])
