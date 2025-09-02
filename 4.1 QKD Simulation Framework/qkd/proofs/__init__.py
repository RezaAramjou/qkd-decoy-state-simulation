# -*- coding: utf-8 -*-
"""
Security proofs sub-package for the QKD framework.

This package exposes the core abstract base class for all proofs and the
concrete implementations for different QKD protocols.
"""

# Import the base class to make it available for type hinting and extension.
from .base import FiniteKeyProof

# Import the concrete proof implementations to make them part of the public API.
from .lim2014 import Lim2014Proof
from .tight import BB84TightProof
from .mdi import MDIQKDProof

# Define the public API for this sub-package.
__all__ = [
    "FiniteKeyProof",
    "Lim2014Proof",
    "BB84TightProof",
    "MDIQKDProof",
]
