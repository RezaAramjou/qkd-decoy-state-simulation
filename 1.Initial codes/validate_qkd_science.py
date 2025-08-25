# validate_qkd_science.py
from __future__ import annotations
import sys
import time
import math
import json
import logging
from dataclasses import asdict
from typing import Dict, Any, Tuple, List

import numpy as np

# Import the user's module
try:
    import qkd_simulation_revised as qkdmod
except Exception as e:
    print("ERROR: failed to import qkd_simulation_revised:", e)
    sys.exit(2)

# ---------- Configuration / Acceptance Criteria ----------
CONFIG = {
    # Numerical tolerances
    "analytical_rel_tol": 1e-4,      # relative tolerance between LP and analytic Y1 lower bound
    "analytical_abs_tol": 1e-8,      # absolute tolerance
    "lp_residual_rel_tol": 1e-8,     # relative tolerance for LP residuals (times norm(b))
    "lp_residual_abs_tol": 1e-10,    # absolute floor for residual tolerance
    "reproducibility_rel_std_tol": 0.05,  # allowed relative std of SKR across seeds (5%)
    "monotonicity_tol": 1e-12,       # tolerance when checking monotonic behavior (non-increasing / non-decreasing)
    # Simulation sizes for checks (kept moderate so script runs fast; increase for stronger validation)
    "quick_run_num_bits": 10_000_000,
    "repeatability_num_bits": 20_000_000,
    "repeatability_seeds": 6,
    "sweep_num_bits": 1_000_000,
}

logger = logging.getLogger("validate_qkd")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Helpers ----------
def close(a, b, rtol=1e-8, atol=0.0):
    return np.isclose(a, b, rtol=rtol, atol=atol)

def report_pass(msg):
    print(f"[PASS] {msg}")

def report_fail(msg):
    print(f"[FAIL] {msg}")

# ---------- Test Cases ----------
class ValidationSummary:
    def __init__(self):
        self.passes: List[str] = []
        self.fails: List[str] = []
        self.details: Dict[str, Any] = {}

    def mark_pass(self, key: str, msg: str, details: Any = None):
        self.passes.append(key)
        self.details[key] = {"status": "PASS", "msg": msg, "details": details}
        report_pass(msg)

    def mark_fail(self, key: str, msg: str, details: Any = None):
        self.fails.append(key)
        self.details[key] = {"status": "FAIL", "msg": msg, "details": details}
        report_fail(msg)

summary = ValidationSummary()

# ---------- 1) Basic unit checks (re-use tested functions) ----------
def unit_checks():
    try:
        # p_n_mu_vector
        v = qkdmod.p_n_mu_vector(0.1, 10)
        if not close(np.sum(v), 1.0, rtol=1e-12, atol=1e-12):
            summary.mark_fail("p_n_mu_sum", "p_n_mu_vector did not sum to 1.0", {"sum": float(np.sum(v))})
        else:
            summary.mark_pass("p_n_mu_sum", "p_n_mu_vector sums to 1 for mu=0.1,n_cap=10")

        v0 = qkdmod.p_n_mu_vector(0.0, 5)
        if not np.allclose(v0, np.array([1.0] + [0.0]*5)):
            summary.mark_fail("p_n_mu_zero", "p_n_mu_vector(0) mismatch", {"vec": v0.tolist()})
        else:
            summary.mark_pass("p_n_mu_zero", "p_n_mu_vector(0) returns delta at zero")

        # binomial_ci basic sanity
        lo, hi = qkdmod.binomial_ci(5, 10, conf_level=0.95)
        if not (0.0 <= lo <= hi <= 1.0):
            summary.mark_fail("binom_bounds", "binomial_ci returned out-of-range values", {"lo": lo, "hi": hi})
        else:
            summary.mark_pass("binom_bounds", "binomial_ci returns bounds within [0,1]")

    except Exception as e:
        summary.mark_fail("unit_checks_error", f"Exception running unit checks: {e}", {"exc": str(e)})

# ---------- 2) LP analytical match and LP feasibility ----------
def construct_final_stats_for_ideal_case(params: qkdmod.QKDParams) -> Tuple[Dict[str, qkdmod.PerPulseTypeDetailedStats], Dict[str, np.ndarray]]:
    """
    Build final_stats according to ideal model used in tests:
      Y0 = 0, Y1 = eta (transmittance*det_eff*basis_match), Yn>=2 = 1.
    Also returns p_vecs for debugging/analytic calculation.
    """
    p_vecs = {}
    final_stats = {}
    eta = params.transmittance * params.det_eff * params.basis_match_probability
    for pc in params.pulse_configs:
        mu = pc.mean_photon_number
        ncap = params.photon_number_cap
        pvec = qkdmod.p_n_mu_vector(mu, ncap)
        p_vecs[pc.name] = pvec
        Y_n_ideal = np.array([0.0] + [eta] + [1.0]*(ncap-1))[: ncap+1]  # ensure length matches
        q_ideal = float(np.dot(pvec, Y_n_ideal))
        # mimic extremely tight statistics by using very large total_sent
        total_sent = 10**12
        total_sifted = int(total_sent * q_ideal)
        final_stats[pc.name] = qkdmod.PerPulseTypeDetailedStats(
            pulse_type_name=pc.name,
            total_sent=total_sent,
            total_detected_any=0,
            total_sifted=total_sifted,
            total_errors_sifted=0,
            overall_gain_any=0.0,
            overall_sifted_gain=q_ideal,
            overall_error_gain=0.0,
            overall_qber_sifted=0.0
        )
    return final_stats, p_vecs

def reconstruct_lp_matrix_and_b(final_stats: Dict[str, qkdmod.PerPulseTypeDetailedStats], params: qkdmod.QKDParams):
    """
    Recreate A_tmp and b_ub exactly as estimate_Y1_e1_lp constructs them,
    so we can compute residuals against returned solution vector.
    Returns: A_tmp (csr_matrix), b_ub (np.ndarray), Nvar (int)
    """
    from scipy.sparse import csr_matrix
    required = ["signal", "decoy", "vacuum"]
    num_constraints = len(required) * 4
    alpha = 1.0 - params.confidence_level
    conf_level_per = 1.0 - (alpha / num_constraints)
    # Build Q & S intervals
    Q_sift_L, Q_sift_U, S_sift_L, S_sift_U = {}, {}, {}, {}
    for name in required:
        stats = final_stats[name]
        Q_sift_L[name], Q_sift_U[name] = qkdmod.binomial_ci(stats.total_sifted, stats.total_sent, conf_level_per, side='two-sided')
        S_sift_L[name], S_sift_U[name] = qkdmod.binomial_ci(stats.total_errors_sifted, stats.total_sent, conf_level_per, side='two-sided')

    cap = params.photon_number_cap
    Nvar = cap + 1
    Y_indices, S_indices = np.arange(Nvar), np.arange(Nvar, 2*Nvar)
    rows, cols, data = [], [], []
    b_ub = []
    row_idx = 0
    pulse_map = {pc.name: pc for pc in params.pulse_configs}
    for name in required:
        mu = pulse_map[name].mean_photon_number
        p_vec = qkdmod.p_n_mu_vector(mu, cap)
        # lower Q
        rows.extend([row_idx]*Nvar); cols.extend(Y_indices.tolist()); data.extend((-p_vec).tolist())
        b_ub.append(-Q_sift_L[name]); row_idx += 1
        # upper Q
        rows.extend([row_idx]*Nvar); cols.extend(Y_indices.tolist()); data.extend(p_vec.tolist())
        b_ub.append(Q_sift_U[name]); row_idx += 1
        # lower S
        rows.extend([row_idx]*Nvar); cols.extend(S_indices.tolist()); data.extend((-p_vec).tolist())
        b_ub.append(-S_sift_L[name]); row_idx += 1
        # upper S
        rows.extend([row_idx]*Nvar); cols.extend(S_indices.tolist()); data.extend(p_vec.tolist())
        b_ub.append(S_sift_U[name]); row_idx += 1

    for n in range(Nvar):
        rows.extend([row_idx, row_idx]); cols.extend([int(S_indices[n]), int(Y_indices[n])]); data.extend([1.0, -1.0])
        b_ub.append(0.0); row_idx += 1
        if params.enforce_monotonicity and n < cap:
            rows.extend([row_idx, row_idx]); cols.extend([int(Y_indices[n]), int(Y_indices[n+1])]); data.extend([1.0, -1.0])
            b_ub.append(0.0); row_idx += 1

    A_tmp = csr_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                      shape=(row_idx, 2*Nvar))
    return A_tmp, np.array(b_ub, dtype=float), Nvar

def lp_analytical_and_feasibility_check():
    # prepare params with verbose_stats True so we can inspect lp_solution if present (not strictly required)
    default_pcs = [
        qkdmod.PulseTypeConfig("signal", 0.5, 0.7),
        qkdmod.PulseTypeConfig("decoy", 0.1, 0.15),
        qkdmod.PulseTypeConfig("vacuum", 0.0, 0.15),
    ]
    params_dict = {
        "num_bits": 1000,
        "pulse_configs": default_pcs,
        "distance_km": 10.0,
        "fiber_loss_db_km": 0.2,
        "det_eff": 1.0,
        "dark_rate": 0.0,
        "qber_intrinsic": 0.0,
        "misalignment": 0.0,
        "double_click_policy": qkdmod.DoubleClickPolicy.DISCARD,
        "basis_match_probability": 0.5,
        "f_error_correction": 1.1,
        "confidence_level": 0.99999999,
        "min_detections_for_stat": 1,
        "photon_number_cap": 2,
        "batch_size": 1000,
        "num_workers": 1,
        "force_sequential": True,
        "verbose_stats": True,
        "enforce_monotonicity": True,
        "assume_phase_equals_bit_error": True
    }
    params = qkdmod.QKDParams(**params_dict)

    final_stats, p_vecs = construct_final_stats_for_ideal_case(params)
    # Analytical Y1 lower bound calculation
    pdec = p_vecs["decoy"]; qdec = final_stats["decoy"].overall_sifted_gain
    y1_bound_decoy = (qdec - pdec[2]) / pdec[1] if pdec[1] > 0 else 0.0
    psig = p_vecs["signal"]; qsig = final_stats["signal"].overall_sifted_gain
    y1_bound_signal = (qsig - psig[2]) / psig[1] if psig[1] > 0 else 0.0
    analytical_y1_l = max(0.0, y1_bound_decoy, y1_bound_signal)

    # Run LP
    qs = qkdmod.QKDSystem(params)
    decoy_est = qs.estimate_Y1_e1_lp(final_stats)
    if decoy_est.get("status") != "OK":
        summary.mark_fail("lp_status", f"LP failed with status {decoy_est.get('status')}", {"decoy_est": decoy_est})
        return

    lp_y1 = float(decoy_est["Y1_sift_L"])
    # Check closeness to analytical
    if not np.isclose(lp_y1, analytical_y1_l, rtol=CONFIG["analytical_rel_tol"], atol=CONFIG["analytical_abs_tol"]):
        summary.mark_fail("lp_vs_analytic", f"LP Y1 ({lp_y1:.12g}) differs from analytical ({analytical_y1_l:.12g})",
                          {"lp_y1": lp_y1, "analytical_y1": analytical_y1_l,
                           "rel_tol": CONFIG["analytical_rel_tol"], "abs_tol": CONFIG["analytical_abs_tol"]})
    else:
        summary.mark_pass("lp_vs_analytic", f"LP Y1 matches analytical within tol ({lp_y1:.12g} ≈ {analytical_y1_l:.12g})")

    # Feasibility: reconstruct LP matrix and compute residual with the LP solution vector (if LP returns vector)
    # LP returns 'lp_solution' only when verbose_stats True; but it stores solution as unscaled vector (res_s1.x).
    lp_sol_vec = None
    if decoy_est.get("lp_solution") and isinstance(decoy_est["lp_solution"], dict):
        # expected keys "Y_n_sift" and "S_n_sift"
        Yn = decoy_est["lp_solution"].get("Y_n_sift")
        Sn = decoy_est["lp_solution"].get("S_n_sift")
        if Yn is not None and Sn is not None:
            lp_sol_vec = np.array(list(map(float, Yn)) + list(map(float, Sn)))
    # If not present, try to reconstruct by calling solve path (not possible without modifying module), so we skip.
    if lp_sol_vec is None:
        # We'll reconstruct feasibility by checking constraints using the reported Y1 and e1 values:
        # Quick feasibility check: ensure aggregated constraints are satisfied for Y1 and S1 bounds
        # (we still build A_tmp to compute full residual if possible)
        try:
            A_tmp, b_ub, Nvar = reconstruct_lp_matrix_and_b(final_stats, params)
            # Build an approximate solution vector using reported Y1 and e1 and heuristics:
            # set Y_n = Y1 for n=1, and for n>=2 set 1.0 (worst-case), for n=0 set 0.
            approx_x = np.zeros(2 * Nvar)
            approx_x[1] = lp_y1
            approx_x[:Nvar] = np.maximum(approx_x[:Nvar], 0.0)
            # set S_n similarly proportional to Y_n with zero errors in ideal case -> S ≈ 0
            approx_x[Nvar:] = 0.0
            residual = A_tmp.dot(approx_x) - b_ub
            # Compute scaled tolerance
            b_inf = max(1.0, np.linalg.norm(b_ub, np.inf))
            tol = max(CONFIG["lp_residual_abs_tol"], CONFIG["lp_residual_rel_tol"] * b_inf)
            if np.any(residual > tol + 1e-15):
                summary.mark_fail("lp_feasibility_approx", "Approximate LP feasibility check failed (residual too large)",
                                  {"residual_max": float(residual.max()), "tol": tol})
            else:
                summary.mark_pass("lp_feasibility_approx", "Approximate LP feasibility checks passed")
        except Exception as e:
            summary.mark_fail("lp_feasibility_error", f"Error while reconstructing LP for feasibility: {e}")
    else:
        # we have the solution vector from the LP
        try:
            A_tmp, b_ub, Nvar = reconstruct_lp_matrix_and_b(final_stats, params)
            residual = A_tmp.dot(lp_sol_vec) - b_ub
            b_inf = max(1.0, np.linalg.norm(b_ub, np.inf))
            tol = max(CONFIG["lp_residual_abs_tol"], CONFIG["lp_residual_rel_tol"] * b_inf)
            if np.any(residual > tol + 1e-15):
                summary.mark_fail("lp_feasibility", "LP solution violates constraints after reconstruction",
                                  {"residual_max": float(residual.max()), "tol": tol})
            else:
                summary.mark_pass("lp_feasibility", f"LP solution feasible (residual_max={float(residual.max()):.3e} <= tol={tol:.3e})")
        except Exception as e:
            summary.mark_fail("lp_feasibility_error", f"Error checking LP solution feasibility: {e}", {"exc": str(e)})

# ---------- 3) Repeatability over seeds ----------
def repeatability_check():
    params = qkdmod.QKDParams(
        num_bits=CONFIG["repeatability_num_bits"],
        pulse_configs=[
            qkdmod.PulseTypeConfig("signal", 0.5, 0.7),
            qkdmod.PulseTypeConfig("decoy", 0.1, 0.15),
            qkdmod.PulseTypeConfig("vacuum", 0.0, 0.15),
        ],
        distance_km=25.0,
        fiber_loss_db_km=0.2,
        det_eff=0.20,
        dark_rate=1e-6,
        qber_intrinsic=0.01,
        misalignment=0.015,
        double_click_policy=qkdmod.DoubleClickPolicy.DISCARD,
        basis_match_probability=0.5,
        f_error_correction=1.1,
        confidence_level=0.95,
        min_detections_for_stat=10,
        photon_number_cap=12,
        batch_size=50_000,
        num_workers=1,
        force_sequential=True,
        verbose_stats=False,
        enforce_monotonicity=True,
        assume_phase_equals_bit_error=True
    )
    skr_list = []
    seeds = list(range(CONFIG["repeatability_seeds"]))
    for seed in seeds:
        qs = qkdmod.QKDSystem(params, seed=seed)
        res = qs.run_simulation()
        if res.status != "OK":
            summary.mark_fail("repeat_run_status", f"Simulation failed with status {res.status} for seed {seed}",
                              {"seed": seed, "status": res.status})
            return
        skr = float(res.secure_key_rate) if res.secure_key_rate is not None else float("nan")
        skr_list.append(skr)
    skr_arr = np.array(skr_list, dtype=float)
    mean_skr = float(np.nanmean(skr_arr))
    std_skr = float(np.nanstd(skr_arr, ddof=0))
    if mean_skr == 0 or not np.isfinite(mean_skr):
        summary.mark_fail("repeatability_mean_zero", "Mean SKR is zero or non-finite, repeatability check failed", {"mean": mean_skr, "std": std_skr, "vals": skr_list})
    else:
        rel_std = std_skr / mean_skr
        if rel_std > CONFIG["reproducibility_rel_std_tol"]:
            summary.mark_fail("repeatability_variability", f"SKR variability across seeds too large (rel_std={rel_std:.3f})",
                              {"mean_skr": mean_skr, "std_skr": std_skr, "rel_std": rel_std, "vals": skr_list})
        else:
            summary.mark_pass("repeatability", f"SKR reproducible across seeds (mean={mean_skr:.3e}, std={std_skr:.3e}, rel_std={rel_std:.3f})",
                              {"mean_skr": mean_skr, "std_skr": std_skr, "vals": skr_list})

# ---------- 4) Monotonicity / Sensitivity checks ----------
def monotonicity_checks():
    # Central common params (do NOT include distance_km or det_eff here)
    params_common = {
        "num_bits": CONFIG["sweep_num_bits"],
        "pulse_configs": [
            qkdmod.PulseTypeConfig("signal", 0.5, 0.7),
            qkdmod.PulseTypeConfig("decoy", 0.1, 0.15),
            qkdmod.PulseTypeConfig("vacuum", 0.0, 0.15),
        ],
        "fiber_loss_db_km": 0.2,
        "dark_rate": 1e-6,
        "qber_intrinsic": 0.01,
        "misalignment": 0.015,
        "double_click_policy": qkdmod.DoubleClickPolicy.DISCARD,
        "basis_match_probability": 0.5,
        "f_error_correction": 1.1,
        "confidence_level": 0.95,
        "min_detections_for_stat": 10,
        "photon_number_cap": 12,
        "batch_size": 50_000,
        "num_workers": 1,
        "force_sequential": True,
        "verbose_stats": False,
        "enforce_monotonicity": True,
        "assume_phase_equals_bit_error": True
    }

    # 1) SKR vs distance: use fixed detector efficiency
    det_eff_fixed = 0.20
    distances = [5.0, 25.0, 50.0]
    skr_vs_dist = []
    for d in distances:
        params = qkdmod.QKDParams(distance_km=d, det_eff=det_eff_fixed, **params_common)
        qs = qkdmod.QKDSystem(params, seed=12345)
        res = qs.run_simulation()
        if res.status != "OK":
            summary.mark_fail("monotonicity_distance_run", f"Simulation failed for distance {d} with status {res.status}")
            return
        skr_vs_dist.append(float(res.secure_key_rate or 0.0))

    # Expect SKR to be non-increasing with distance
    ok = True
    for i in range(1, len(skr_vs_dist)):
        if skr_vs_dist[i] > skr_vs_dist[i-1] + CONFIG["monotonicity_tol"]:
            ok = False; break
    if not ok:
        summary.mark_fail("monotonicity_distance", "SKR increased with distance unexpectedly",
                          {"distances": distances, "skr": skr_vs_dist})
    else:
        summary.mark_pass("monotonicity_distance", "SKR behaved non-increasingly with distance",
                          {"distances": distances, "skr": skr_vs_dist})

    # 2) SKR vs detector efficiency: use fixed distance
    distance_fixed = 25.0
    dets = [0.05, 0.20, 0.40]
    skr_vs_det = []
    for de in dets:
        params = qkdmod.QKDParams(distance_km=distance_fixed, det_eff=de, **params_common)
        qs = qkdmod.QKDSystem(params, seed=54321)
        res = qs.run_simulation()
        if res.status != "OK":
            summary.mark_fail("monotonicity_de_run", f"Simulation failed for det_eff {de} with status {res.status}")
            return
        skr_vs_det.append(float(res.secure_key_rate or 0.0))

    ok = True
    for i in range(1, len(skr_vs_det)):
        if skr_vs_det[i] + CONFIG["monotonicity_tol"] < skr_vs_det[i-1]:
            ok = False; break
    if not ok:
        summary.mark_fail("monotonicity_det", "SKR decreased when detector efficiency increased unexpectedly",
                          {"det_eff": dets, "skr": skr_vs_det})
    else:
        summary.mark_pass("monotonicity_det", "SKR behaved non-decreasingly with detector efficiency",
                          {"det_eff": dets, "skr": skr_vs_det})


# ---------- 5) Finite-key presence check (we don't implement finite-key in the main code) ----------
def finite_key_presence_check():
    # Quick heuristic: look for function names or references in module indicating finite-key code
    text = None
    try:
        import inspect
        text = inspect.getsource(qkdmod)
    except Exception:
        # fallback: check module attributes
        text = " ".join(dir(qkdmod))
    keywords = ["finite", "finite_key", "finite-key", "finite_key_rate", "composable", "smooth", "security_parameter", "epsilon"]
    found = [k for k in keywords if k in text]
    if found:
        summary.mark_pass("finite_key_code_present", f"Potential finite-key related identifiers found: {found}", {"matches": found})
    else:
        summary.mark_fail("finite_key_code_missing", "No finite-key postprocessing code or identifiers found. Finite-key corrections required for practical security.", {"checked_keywords": keywords})

# ---------- 6) Quick smoke test (end-to-end) ----------
def smoke_test():
    params = qkdmod.QKDParams(
        num_bits=CONFIG["quick_run_num_bits"],
        pulse_configs=[
            qkdmod.PulseTypeConfig("signal", 0.5, 0.7),
            qkdmod.PulseTypeConfig("decoy", 0.1, 0.15),
            qkdmod.PulseTypeConfig("vacuum", 0.0, 0.15),
        ],
        distance_km=25.0,
        fiber_loss_db_km=0.2,
        det_eff=0.20,
        dark_rate=1e-6,
        qber_intrinsic=0.01,
        misalignment=0.015,
        double_click_policy=qkdmod.DoubleClickPolicy.DISCARD,
        basis_match_probability=0.5,
        f_error_correction=1.1,
        confidence_level=0.95,
        min_detections_for_stat=10,
        photon_number_cap=12,
        batch_size=50_000,
        num_workers=1,
        force_sequential=True,
        verbose_stats=False,
        enforce_monotonicity=True,
        assume_phase_equals_bit_error=True
    )
    try:
        qs = qkdmod.QKDSystem(params, seed=999)
        res = qs.run_simulation()
        if res.status != "OK":
            summary.mark_fail("smoke_run_status", f"Smoke run failed with status {res.status}")
        else:
            summary.mark_pass("smoke_run", f"Smoke run OK: SKR={res.secure_key_rate:.6e}, raw_sifted={res.raw_sifted_key_length}")
    except Exception as e:
        summary.mark_fail("smoke_exception", f"Smoke run raised exception: {e}")

# ---------- Runner ----------
def run_all():
    print("=== QKD Scientific Validation Suite ===")
    t0 = time.time()
    unit_checks()
    lp_analytical_and_feasibility_check()
    repeatability_check()
    monotonicity_checks()
    finite_key_presence_check()
    smoke_test()
    elapsed = time.time() - t0

    print("\n=== SUMMARY ===")
    print(f"Total time: {elapsed:.2f} s")
    print(f"Passed checks: {len(summary.passes)}")
    print(f"Failed checks: {len(summary.fails)}")
    if summary.fails:
        print("\nFailed items and details:")
        for k in summary.fails:
            entry = summary.details.get(k, {})
            print(f"- {k}: {entry.get('msg')}")
            if entry.get('details') is not None:
                print(f"  details: {entry.get('details')}")
        print("\nVERDICT: NOT READY — one or more validation checks failed.")
        print("Notes:")
        print("- Fix the failing items listed above, then re-run this validation.")
        print("- Passing all checks here still does not constitute formal security proof; you must perform finite-key analysis and an independent review.")
        sys.exit(1)
    else:
        print("\nAll validation checks passed according to the configured acceptance criteria.")
        print("VERDICT: READY FOR NEXT-STEP VALIDATION (but NOT a formal proof of security).")
        print("Recommended next steps:")
        print("- Implement / integrate finite-key corrections and re-run.")
        print("- Run larger-scale robustness sweeps (more pulses, more seeds).")
        print("- Obtain an independent domain review of the protocol assumptions and LP formulation.")
        sys.exit(0)

if __name__ == "__main__":
    run_all()
