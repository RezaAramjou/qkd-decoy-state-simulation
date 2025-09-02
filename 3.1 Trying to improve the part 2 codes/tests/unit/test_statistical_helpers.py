# -*- coding: utf-8 -*-
"""
Unit tests for standalone statistical helper functions.
"""
import pytest
pytest.importorskip("scipy")

import numpy as np
import math
from unittest.mock import patch

import qkd_simulation_main
from qkd_simulation_main import (
    p_n_mu_vector, clopper_pearson_bounds, hoeffding_bounds, Lim2014Proof, ENTROPY_PROB_CLAMP
)

class TestStatisticalHelpers:
    """Tests for standalone statistical utility functions."""
    @pytest.mark.parametrize("mu, n_cap, expected_len, expected_first", [
        (0.5, 10, 11, 0.6065306597),
        (0.0, 5, 6, 1.0),
        (10.0, 1, 2, 4.539992976e-05),
    ])
    def test_p_n_mu_vector_cases(self, mu, n_cap, expected_len, expected_first):
        """Tests p_n_mu_vector for various edge cases."""
        vec = p_n_mu_vector(mu=mu, n_cap=n_cap)
        assert len(vec) == expected_len
        np.testing.assert_allclose(vec[0], expected_first, rtol=1e-8)
        np.testing.assert_allclose(np.sum(vec), 1.0, rtol=1e-8)

    def test_p_n_mu_vector_renormalization_warning(self):
        """Tests that a warning is logged if Poisson sum deviates significantly."""
        def fake_pmf(ns, mu):
            arr = np.zeros_like(ns, dtype=float)
            arr[0] = 0.9
            return arr
        def fake_sf(k, mu):
            return 0.3

        with patch.object(qkd_simulation_main, 'logger') as mock_logger, \
             patch('qkd_simulation_main.poisson.pmf', side_effect=fake_pmf), \
             patch('qkd_simulation_main.poisson.sf', side_effect=fake_sf):
            p_n_mu_vector(mu=1.0, n_cap=1)
            mock_logger.warning.assert_called_once()
            assert "deviates by" in mock_logger.warning.call_args[0][0]

    def test_clopper_pearson_properties(self):
        """Tests the mathematical properties of Clopper-Pearson bounds."""
        k, n = 10, 100
        lower, upper = clopper_pearson_bounds(k, n, 0.05)
        assert 0 <= lower <= k/n <= upper <= 1
        assert clopper_pearson_bounds(0, n, 0.05)[0] == 0.0
        assert clopper_pearson_bounds(n, n, 0.05)[1] == 1.0

    def test_hoeffding_bounds_invalid_inputs(self):
        """Tests Hoeffding bounds for invalid inputs."""
        assert hoeffding_bounds(10, 0, 0.05) == (0.0, 1.0)
        with pytest.raises(ValueError):
            hoeffding_bounds(1, 10, failure_prob=0.0)

    @pytest.mark.parametrize("p, expected_h", [
        (0.0, 0.0), (1.0, 0.0), (0.5, 1.0),
        (0.11, -0.11 * math.log2(0.11) - 0.89 * math.log2(0.89)),
    ])
    def test_binary_entropy(self, p, expected_h):
        """Tests binary entropy calculation for various probabilities."""
        h = Lim2014Proof.binary_entropy(p)
        np.testing.assert_allclose(h, expected_h, rtol=1e-9)

    def test_binary_entropy_clamp(self):
        """Tests that probabilities are correctly clamped at the lower bound."""
        p_clamped = ENTROPY_PROB_CLAMP
        h_clamped = Lim2014Proof.binary_entropy(p_clamped)
        h_below_clamp = Lim2014Proof.binary_entropy(p_clamped / 2)
        assert h_clamped > 0
        assert math.isclose(h_clamped, h_below_clamp)
