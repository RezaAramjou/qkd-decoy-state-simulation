# -*- coding: utf-8 -*-
"""
Unit Tests for the QKD Simulation Script (v7 - Final Patched)

This script contains a suite of unit tests for the core statistical and
numerical components of the `qkd_simulation_revised.py` script.

To run these tests, you need to have `pytest` installed (`pip install pytest`).
Save this file as `test_qkd_simulation.py` in the same directory as the main
script and run `pytest` from your terminal.

These tests verify:
- The numerical stability and correctness of the Poisson PMF calculation.
- The robustness and accuracy of the binomial confidence interval function.
- The correctness of the LP-based decoy estimation by comparing it to a known
  analytical solution in an ideal case.
- Correct handling of parameter validation and edge cases.
"""
import sys
import pytest
import numpy as np
from dataclasses import asdict

# Import the functions and classes to be tested from the main script
try:
    from qkd_simulation_revised import (
        p_n_mu_vector,
        binomial_ci,
        QKDSystem,
        QKDParams,
        PerPulseTypeDetailedStats,
        PulseTypeConfig,
        DoubleClickPolicy,
        ParameterValidationError
    )
except ImportError as e:
    pytest.skip(f"Could not import from 'qkd_simulation_revised': {e}", allow_module_level=True)


# --- Tests for p_n_mu_vector ---

def test_p_n_mu_vector_sum_to_one():
    """Tests if the probabilities in the PMF vector sum to 1."""
    for mu in [0.1, 1.0, 10.0, 50.0]:
        for n_cap in [5, 20, 100]:
            vec = p_n_mu_vector(mu, n_cap)
            assert np.isclose(np.sum(vec), 1.0), f"PMF for mu={mu}, n_cap={n_cap} does not sum to 1."

def test_p_n_mu_vector_zero_mu():
    """Tests the PMF for a mean photon number of zero."""
    vec = p_n_mu_vector(0, 10)
    expected = np.zeros(11)
    expected[0] = 1.0
    assert np.allclose(vec, expected)

def test_p_n_mu_vector_large_mu_tail():
    """Tests if the tail probability is significant for large mu."""
    # For mu=20, n_cap=10, the tail P(n>=10) should be large.
    vec = p_n_mu_vector(20, 10)
    assert vec[-1] > 0.9, "Tail probability for large mu is unexpectedly small."

def test_p_n_mu_vector_negative_mu_raises_error():
    """Tests that a negative mu raises a ValueError."""
    with pytest.raises(ValueError):
        p_n_mu_vector(-1.0, 10)


# --- Tests for binomial_ci ---

def test_binomial_ci_api_and_bounds():
    """Tests the API consistency and that bounds are always within [0, 1]."""
    for k, n in [(0, 10), (5, 10), (10, 10)]:
        # Two-sided
        low, high = binomial_ci(k, n, conf_level=0.95, side='two-sided')
        assert 0.0 <= low <= high <= 1.0
        # One-sided
        low, high = binomial_ci(k, n, conf_level=0.95, side='lower')
        assert 0.0 <= low <= high <= 1.0
        assert high == 1.0
        low, high = binomial_ci(k, n, conf_level=0.95, side='upper')
        assert 0.0 <= low <= high <= 1.0
        assert low == 0.0

def test_binomial_ci_edge_cases():
    """Tests edge cases like n=0, k=0, k=n."""
    assert binomial_ci(0, 0) == (0.0, 1.0)
    # For k=0, lower bound of one-sided interval must be 0
    low, high = binomial_ci(0, 10, side='lower')
    assert low == 0.0 and high == 1.0
    # For k=n, upper bound of one-sided interval must be 1
    low, high = binomial_ci(10, 10, side='upper')
    assert low == 0.0 and high == 1.0
    # For k=0, lower bound of two-sided interval must be 0
    low, _ = binomial_ci(0, 100)
    assert low == 0.0
    # For k=n, upper bound of two-sided interval must be 1
    _, high = binomial_ci(100, 100)
    assert high == 1.0

def test_binomial_ci_invalid_inputs():
    """Tests that invalid inputs raise ValueErrors."""
    with pytest.raises(ValueError):
        binomial_ci(11, 10) # k > n
    with pytest.raises(ValueError):
        binomial_ci(-1, 10) # k < 0
    with pytest.raises(ValueError):
        binomial_ci(5, 10, conf_level=1.5) # conf_level > 1

# --- Integration Test for Decoy LP ---

@pytest.fixture
def base_qkd_params():
    """Provides a default set of QKD parameters for testing."""
    pulse_configs = [
        PulseTypeConfig("signal", 0.5, 0.7),
        PulseTypeConfig("decoy", 0.1, 0.15),
        PulseTypeConfig("vacuum", 0.0, 0.15)
    ]
    return {
        "num_bits": 1000, "pulse_configs": pulse_configs, "distance_km": 10,
        "fiber_loss_db_km": 0.2, "det_eff": 1.0, "dark_rate": 0.0,
        "qber_intrinsic": 0.0, "misalignment": 0.0,
        "double_click_policy": DoubleClickPolicy.DISCARD,
        "basis_match_probability": 0.5, "f_error_correction": 1.1,
        "confidence_level": 0.99999999, "min_detections_for_stat": 1,
        "photon_number_cap": 2, "batch_size": 1000, "num_workers": 1,
        "force_sequential": True, "verbose_stats": False,
        "enforce_monotonicity": True, "assume_phase_equals_bit_error": True
    }

def test_lp_estimation_matches_analytical(base_qkd_params):
    """
    Tests that the LP estimator matches the known analytical formula for a
    simple two-decoy case (Y0=0, Yn>=2 = 1). This is a strong validation of
    the LP formulation's correctness.
    """
    params = QKDParams(**base_qkd_params)
    qsys = QKDSystem(params)
    eta = qsys.p.transmittance * qsys.p.det_eff * qsys.p.basis_match_probability

    final_stats = {}
    q_ideal = {}
    p_vecs = {}

    for pc in qsys.p.pulse_configs:
        mu = pc.mean_photon_number
        n_cap = qsys.p.photon_number_cap
        p_vecs[pc.name] = p_n_mu_vector(mu, n_cap)
        
        # Ideal physical model: Y0=0, Y1=eta, and we assume the worst-case
        # for the tail where Yn>=2=1.
        Y_n_ideal = np.array([0.0, eta, 1.0])

        q_ideal[pc.name] = np.sum(p_vecs[pc.name] * Y_n_ideal)
        
        # Simulate a very large number of trials to make CIs extremely tight
        total_sent = 10**12
        total_sifted = int(total_sent * q_ideal[pc.name])
        
        final_stats[pc.name] = PerPulseTypeDetailedStats(
            pulse_type_name=pc.name,
            total_sent=total_sent, total_detected_any=0,
            total_sifted=total_sifted, total_errors_sifted=0,
            overall_gain_any=0, overall_sifted_gain=q_ideal[pc.name],
            overall_error_gain=0.0, overall_qber_sifted=0.0
        )

    # --- Analytical Calculation ---
    # With Y0=0 and Yn>=2=1, the lower bound on Y1 is given by the tightest
    # constraint from the decoy and signal states.
    p_decoy = p_vecs["decoy"]
    q_decoy = q_ideal["decoy"]
    # Y1 >= (Q_mu - P(n>=2|mu)) / P(1|mu)
    y1_bound_decoy = (q_decoy - p_decoy[2]) / p_decoy[1] if p_decoy[1] > 0 else 0

    p_signal = p_vecs["signal"]
    q_signal = q_ideal["signal"]
    y1_bound_signal = (q_signal - p_signal[2]) / p_signal[1] if p_signal[1] > 0 else 0
    
    analytical_y1_l = max(0, y1_bound_decoy, y1_bound_signal)

    # --- LP Calculation ---
    decoy_est = qsys.estimate_Y1_e1_lp(final_stats)

    assert decoy_est['status'] == 'OK'
    
    lp_y1_l = decoy_est['Y1_sift_L']
    
    # The LP result should be almost identical to the analytical formula.
    # A small tolerance is needed for numerical precision differences.
    assert np.isclose(lp_y1_l, analytical_y1_l, rtol=1e-4, atol=1e-8)
    
    # The error rate should be effectively zero
    assert decoy_est['e1_sift_U'] < 1e-6

def test_lp_infeasible_stats(base_qkd_params):
    """Tests that the LP returns a failure status for inconsistent stats."""
    params = QKDParams(**base_qkd_params)
    qsys = QKDSystem(params)
    
    # Create stats where error gain > sifted gain, which is impossible
    stats = {
        "signal": PerPulseTypeDetailedStats(
            "signal", total_sent=1000, total_sifted=100, total_errors_sifted=150,
            total_detected_any=0, overall_gain_any=0,
            overall_sifted_gain=0.1, overall_error_gain=0.15, overall_qber_sifted=1.5
        ),
        "decoy": PerPulseTypeDetailedStats("decoy", 1000, 10, 5, 0, 0, 0.01, 0.005, 0.5),
        "vacuum": PerPulseTypeDetailedStats("vacuum", 1000, 1, 0, 0, 0, 0.001, 0, float('nan'))
    }
    
    decoy_est = qsys.estimate_Y1_e1_lp(stats)
    assert decoy_est['status'] == 'INCONSISTENT_STATS_SIGNAL'

def test_full_simulation_run(base_qkd_params):
    """A simple end-to-end test to ensure the simulation runs without crashing."""
    params_dict = base_qkd_params
    params_dict["num_bits"] = 1_000_000 # Increased for better stats
    params_dict["distance_km"] = 50
    params_dict["dark_rate"] = 1e-7
    params_dict["qber_intrinsic"] = 0.01
    
    params = QKDParams(**params_dict)
    qsys = QKDSystem(params, seed=42)
    results = qsys.run_simulation()
    
    assert results.status == "OK"
    assert results.secure_key_rate is not None
    assert results.secure_key_rate >= 0
    assert results.raw_sifted_key_length > 0

if __name__ == "__main__":
    # This allows the test script to be run directly from a Python interpreter
    # (e.g., from an IDE's "Run" button) which will then invoke pytest.
    print("--- Running QKD Simulation Unit Tests ---")
    # The '-v' flag gives verbose output. We pass the current file to pytest.
    pytest.main(['-v', __file__])
