# -*- coding: utf-8 -*-
"""
Global fixtures for the QKD simulation test suite.
Fixtures defined here are automatically available to all test files.
"""
import pytest
import copy
from qkd_simulation_main import QKDParams, PulseTypeConfig, DoubleClickPolicy, SecurityProof, ConfidenceBoundMethod

@pytest.fixture
def default_params_dict():
    """
    Provides a default, valid set of parameters for tests.
    Returns a deep copy to ensure test isolation, preventing mutations
    from leaking between tests.
    """
    base = {
        'num_bits': 10000, 'distance_km': 10.0, 'fiber_loss_db_km': 0.2,
        'det_eff': 0.2, 'dark_rate': 1e-8, 'qber_intrinsic': 0.01,
        'misalignment': 0.015, 'bob_z_basis_prob': 0.5, 'f_error_correction': 1.1,
        'eps_sec': 1e-9, 'eps_cor': 1e-15, 'eps_pe': 1e-10, 'eps_smooth': 1e-10,
        'photon_number_cap': 10, 'batch_size': 1000, 'num_workers': 1,
        'force_sequential': True, 'enforce_monotonicity': True,
        'assume_phase_equals_bit_error': False,
        'pulse_configs': [
            # This was changed from PulseTypeConfig objects to dictionaries
            # to prevent FrozenInstanceError during deepcopy in tests.
            {'name': "signal", 'mean_photon_number': 0.5, 'probability': 0.7},
            {'name': "decoy", 'mean_photon_number': 0.1, 'probability': 0.15},
            {'name': "vacuum", 'mean_photon_number': 0.0, 'probability': 0.15}
        ],
        'double_click_policy': DoubleClickPolicy.DISCARD,
        'security_proof': SecurityProof.LIM_2014,
        'ci_method': ConfidenceBoundMethod.CLOPPER_PEARSON,
        'alice_z_basis_prob': 0.5,
        'lp_solver_method': 'highs'
    }
    return copy.deepcopy(base)

@pytest.fixture
def small_params(default_params_dict):
    """A smaller version of parameters for faster unit tests."""
    params = default_params_dict.copy()
    params['num_bits'] = 1000
    params['batch_size'] = 100
    # The from_dict method correctly handles the dictionary format for pulse_configs
    return QKDParams.from_dict(params)
