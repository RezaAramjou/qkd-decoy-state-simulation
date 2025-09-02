# -*- coding: utf-8 -*-
"""
Abstract base class for finite-key security proofs.

Moved from the monolithic script. No logic changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

from ..utils.math import clopper_pearson_bounds, hoeffding_bounds, binary_entropy
from ..datatypes import ConfidenceBoundMethod, TallyCounts, EpsilonAllocation

if TYPE_CHECKING:
    from ..params import QKDParams

__all__ = ["FiniteKeyProof"]


class FiniteKeyProof(ABC):
    def __init__(self, params: "QKDParams"):
        self.p = params
        self.eps_alloc = self.allocate_epsilons()
        self.eps_alloc.validate()

    @abstractmethod
    def allocate_epsilons(self) -> EpsilonAllocation:
        raise NotImplementedError

    @abstractmethod
    def estimate_yields_and_errors(self, stats_map: Dict[str, TallyCounts]) -> Dict[str, any]:
        raise NotImplementedError

    @abstractmethod
    def calculate_key_length(self, decoy_estimates: Dict[str, any], stats_map: Dict[str, TallyCounts]) -> int:
        raise NotImplementedError

    def get_bounds(self, k: int, n: int, failure_prob: float) -> Tuple[float, float]:
        if self.p.ci_method == ConfidenceBoundMethod.CLOPPER_PEARSON:
            return clopper_pearson_bounds(k, n, failure_prob)
        elif self.p.ci_method == ConfidenceBoundMethod.HOEFFDING:
            return hoeffding_bounds(k, n, failure_prob)
        else:
            raise NotImplementedError(f"CI method {self.p.ci_method} not implemented.")

    @staticmethod
    def binary_entropy(p_err: float) -> float:
        return binary_entropy(p_err)
