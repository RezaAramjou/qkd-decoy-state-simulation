# -*- coding: utf-8 -*-
"""
Utility functions for Linear Programming solvers.

Moved from the monolithic script and made more robust.
"""
import logging
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import csr_matrix

from ..exceptions import LPFailureError
from ..constants import LP_CONSTRAINT_VIOLATION_TOL, LP_SOLVER_METHODS

__all__ = ["solve_lp"]

logger = logging.getLogger(__name__)


def solve_lp(cost_vector: np.ndarray, A_ub: csr_matrix, b_ub: np.ndarray, n_vars: int, method: str) -> Tuple[np.ndarray, Dict]:
    """
    Solves a linear program with robust error handling and fallback.
    """
    if cost_vector.size != n_vars:
        raise LPFailureError(f"LP cost_vector length ({cost_vector.size}) mismatches n_vars ({n_vars}).")

    bounds = [(0.0, 1.0)] * n_vars
    last_exc = None

    for solver in LP_SOLVER_METHODS:
        if solver != method and method in LP_SOLVER_METHODS:
            continue # Only try the specified method unless it fails, then try others

        try:
            res: OptimizeResult = linprog(
                cost_vector, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=solver, options={"presolve": True, "tol": 1e-9}
            )

            if not getattr(res, "success", False) or getattr(res, "x", None) is None:
                raise LPFailureError(f"LP solver '{solver}' failed: {getattr(res, 'message', 'No message')} (Status: {getattr(res, 'status', -1)})")

            sol = res.x.copy()
            residual = A_ub.dot(sol) - b_ub
            max_violation = float(np.max(np.maximum(0.0, residual))) if residual.size > 0 else 0.0

            tol = LP_CONSTRAINT_VIOLATION_TOL * max(1.0, np.max(np.abs(b_ub))) if b_ub.size > 0 else LP_CONSTRAINT_VIOLATION_TOL
            if max_violation > tol:
                logger.warning(f"LP solution from '{solver}' violates constraints by {max_violation:.3e} (tol={tol:.1e}).")
                if max_violation > 10 * tol:
                    raise LPFailureError(f"LP solution from '{solver}' violates constraints significantly.")

            diagnostics = {
                "method": str(solver),
                "status": int(getattr(res, "status", -1)),
                "message": str(getattr(res, "message", "")),
                "fun": float(res.fun),
                "nit": int(getattr(res, "nit", -1)),
                "max_violation": max_violation,
            }
            return sol, diagnostics

        except LPFailureError as e:
            last_exc = e
            logger.warning(f"Primary LP solver '{solver}' failed. Trying fallback.")
            continue

    raise LPFailureError(f"All LP solver methods failed. Last error: {last_exc}")
