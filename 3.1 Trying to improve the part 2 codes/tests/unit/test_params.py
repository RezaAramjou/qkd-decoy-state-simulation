# -*- coding: utf-8 -*-
"""
Unit tests for QKDParams validation, computed properties, and edge cases.
"""
import pytest
import copy
import numpy as np
import numpy.testing as npt
import math
from qkd_simulation_main import QKDParams, ParameterValidationError, PulseTypeConfig, PROB_SUM_TOL

class TestParameterValidation:
    """Tests the validation logic within the QKDParams dataclass."""

    def test_valid_params_ok(self, default_params_dict):
        """
        Ensures that a valid set of default parameters instantiates
        without raising any validation errors using the from_dict method.
        """
        QKDParams.from_dict(default_params_dict)

    @pytest.mark.parametrize(
        "param, invalid_value, expected_substr",
        [
            ("num_bits", 0, "num_bits must be positive"),
            ("dark_rate", 1.1, "dark_rate must be in [0,1)"),
            ("f_error_correction", 0.9, "f_error_correction must be in [1.0,5.0]"),
            ("photon_number_cap", 0, "photon_number_cap must be >= 1"),
            ("batch_size", -1, "batch_size must be >0"),
            ("bob_z_basis_prob", 0.0, "bob_z_basis_prob must be in (0,1)"),
            ("bob_z_basis_prob", 1.0, "bob_z_basis_prob must be in (0,1)"),
            ("det_eff", -0.1, "det_eff must be in [0,1]"),
            ("det_eff", 1.1, "det_eff must be in [0,1]"),
            ("eps_sec", 0.0, "eps_sec/eps_cor/eps_pe/eps_smooth must be floats in (0,1)"),
            ("eps_cor", 1.0, "eps_sec/eps_cor/eps_pe/eps_smooth must be floats in (0,1)"),
        ],
        ids=[
            "num_bits_zero", "dark_rate_gt1", "f_error_lt1",
            "photon_cap_zero", "batch_size_neg",
            "bob_z_prob_zero", "bob_z_prob_one", "det_eff_neg",
            "det_eff_gt1", "eps_sec_zero", "eps_cor_one"
        ]
    )
    def test_invalid_params_raise_error(self, default_params_dict, param, invalid_value, expected_substr):
        """
        Verifies that specific out-of-bounds parameters raise ParameterValidationError
        with a message containing an expected substring.
        """
        invalid_params = default_params_dict
        invalid_params[param] = invalid_value
        with pytest.raises(ParameterValidationError) as exc:
            QKDParams.from_dict(invalid_params)
        assert expected_substr in str(exc.value)

    def test_batch_size_exceeding_num_bits_raises(self, default_params_dict):
        """
        Ensures that a batch_size greater than num_bits raises a validation error.
        This test is robust to changes in the fixture's num_bits value.
        """
        params = default_params_dict
        params['batch_size'] = params['num_bits'] + 1
        with pytest.raises(ParameterValidationError) as exc:
            QKDParams.from_dict(params)
        assert "batch_size must be >0 and <= num_bits" in str(exc.value)

    def test_prob_sum_tolerance(self, default_params_dict):
        """
        Tests that the probability sum check correctly handles floating point
        tolerances using math.nextafter for deterministic checks.
        """
        # Case 1: Sum is strictly outside the tolerance (should fail)
        outside_tol = default_params_dict
        p1_outside = math.nextafter(1.0 - 0.15 - 0.15 + PROB_SUM_TOL, 2.0)
        outside_tol['pulse_configs'][0]['probability'] = p1_outside
        with pytest.raises(ParameterValidationError) as exc:
            QKDParams.from_dict(outside_tol)
        assert "Sum of pulse_configs probabilities" in str(exc.value)

        # Case 2: Sum is strictly inside the tolerance (should pass)
        inside_tol = default_params_dict
        p1_inside = math.nextafter(1.0 - 0.15 - 0.15 + PROB_SUM_TOL, 0.0)
        inside_tol['pulse_configs'][0]['probability'] = p1_inside
        QKDParams.from_dict(inside_tol)

    def test_wrong_param_types_raise_error(self, default_params_dict):
        """
        Verifies that passing incorrect data types (e.g., string for a number)
        raises a TypeError or ParameterValidationError during instantiation.
        """
        invalid_params = default_params_dict
        invalid_params['num_bits'] = "one million"
        with pytest.raises((TypeError, ParameterValidationError)):
            QKDParams.from_dict(invalid_params)

    def test_alice_z_basis_prob_endpoints_are_valid(self, default_params_dict):
        """Checks that Alice's Z-basis probability is valid at the [0, 1] endpoints."""
        p0 = default_params_dict
        p0['alice_z_basis_prob'] = 0.0
        QKDParams.from_dict(p0)

        p1 = default_params_dict
        p1['alice_z_basis_prob'] = 1.0
        QKDParams.from_dict(p1)

    def test_valid_boundary_conditions(self, default_params_dict):
        """Tests that parameters at their allowed boundaries instantiate correctly."""
        p = default_params_dict
        p['det_eff'] = 0.0; QKDParams.from_dict(p)
        p['det_eff'] = 1.0; QKDParams.from_dict(p)
        p['f_error_correction'] = 1.0; QKDParams.from_dict(p)
        p['f_error_correction'] = 5.0; QKDParams.from_dict(p)
        p['num_bits'] = 1000; p['batch_size'] = 1000; QKDParams.from_dict(p)

    def test_large_photon_number_cap_valid(self, default_params_dict):
        """Ensures a large but valid photon_number_cap is accepted."""
        params = default_params_dict
        params['photon_number_cap'] = 1000
        QKDParams.from_dict(params)

class TestComputedProperties:
    """Tests for computed properties like transmittance."""

    def test_transmittance_property(self, default_params_dict):
        """
        Verifies the transmittance calculation for various distances and
        loss values, including edge cases, with explicit float tolerances.
        """
        params = QKDParams.from_dict(default_params_dict)
        
        params.distance_km = 50.0
        params.fiber_loss_db_km = 0.2
        expected_transmittance = 10**(- (50.0 * 0.2) / 10.0)
        npt.assert_allclose(params.transmittance, expected_transmittance, rtol=1e-12, atol=1e-12)

        params.distance_km = 0.0
        npt.assert_allclose(params.transmittance, 1.0, rtol=1e-12, atol=1e-12)
        
        params.distance_km = 10.0
        params.fiber_loss_db_km = 0.0
        npt.assert_allclose(params.transmittance, 1.0, rtol=1e-12, atol=1e-12)

        params.distance_km = -10.0
        npt.assert_allclose(params.transmittance, 0.0, atol=1e-12)

        params.distance_km = 10.0
        params.fiber_loss_db_km = -0.1
        npt.assert_allclose(params.transmittance, 0.0, atol=1e-12)
