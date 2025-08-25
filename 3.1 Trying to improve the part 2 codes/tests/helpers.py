# -*- coding: utf-8 -*-
"""
Helper utilities and mock factories for the QKD simulation test suite.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

import qkd_simulation_main
from qkd_simulation_main import QKDSystem

@pytest.fixture
def rng42():
    """Provides a deterministic, seeded RNG instance."""
    return np.random.default_rng(42)

@pytest.fixture
def qkd_system_small(small_params, rng42):
    """Provides a QKDSystem instance configured with small parameters and a deterministic RNG."""
    # The rng_factory ensures that even if the system creates its own RNG, it's deterministic.
    return QKDSystem(small_params, seed=42, rng_factory=lambda s: np.random.default_rng(s))

def make_opt_result(success=True, x=None, message="", status=0, fun=0.0, nit=0):
    """Creates a mock scipy.optimize.OptimizeResult object."""
    res = MagicMock()
    res.success = success
    res.x = np.array(x if x is not None else [])
    res.message = message
    res.status = status
    res.fun = fun
    res.nit = nit
    return res

def fake_solve_lp_factory(Nvar, Y1_val=0.2, E1_val=0.05):
    """
    Creates a side_effect function for mocking Lim2014Proof._solve_lp.
    It inspects the cost vector to return appropriate mock solutions.
    """
    def fake(cost_vector, A_ub, b_ub, n_vars):
        sol = np.zeros(n_vars)
        # Detect Y1 objective: cost has 1.0 at index _idx_y(1,Nvar)
        if cost_vector.sum() == 1.0 and cost_vector[1] == 1.0:
            sol[1] = Y1_val
            return sol, {"method": "fake", "status": 0}
        # Detect E1 objective: negative in e1 index
        if cost_vector.sum() == -1.0 and cost_vector[Nvar+1] == -1.0:
            sol[Nvar+1] = E1_val
            return sol, {"method": "fake", "status": 0}
        # Default for other objectives (e.g., Y0)
        return sol, {}
    return fake

def make_random_side_effect(num_values):
    """Creates a side_effect function for mock_rng.random that handles size."""
    def rnd(size=None):
        if size is None: return np.random.random()
        # Deterministic increasing values between 0 and 1
        return np.linspace(0.0, 1.0, num=size)
    return rnd
