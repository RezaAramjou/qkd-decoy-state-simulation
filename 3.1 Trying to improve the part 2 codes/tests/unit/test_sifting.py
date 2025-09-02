# -*- coding: utf-8 -*-
"""
Unit tests for the sifting and error correction logic.
This version has been hardened based on expert review to be more robust and less brittle.
"""
import pytest
import copy

# Explicitly skip this entire module if scipy is not available, with a clear reason.
pytest.importorskip("scipy", reason="This test module depends on fixtures that may require SciPy.")

import numpy as np

import qkd_simulation_main
from qkd_simulation_main import QKDParams
from tests.helpers import make_random_side_effect

# NOTE on testing private functions:
# These tests call the private helper `_sifting_and_errors`. This is done to isolate
# the complex sifting logic. This approach is brittle and may break if the function's
# signature or behavior changes during refactoring.
# Expected signature: _sifting_and_errors(params, num_pulses, alice_bits, alice_bases,
#                                       signal_click, dark0, dark1, rng)
# Expected return schema: (sifted, errors, discarded_dc_mask)
# Expected types:
# - sifted: boolean mask of length num_pulses OR 1D array of sifted indices
# - errors: boolean or integer mask of length num_pulses
# - discarded_dc_mask: boolean mask of length num_pulses

class FakeRNG:
    """
    A deterministic fake RNG that mimics the numpy.random.Generator API
    for the specific methods used by the function under test.
    """
    def __init__(self, choice_out, random_seq):
        self._choice_out = np.asarray(choice_out)
        self._random_seq = np.asarray(random_seq)

    def choice(self, a, size=None, replace=True, p=None):
        # Normalize size to a tuple to be compatible with np.broadcast_to
        if size is None:
            return self._choice_out
        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        return np.broadcast_to(self._choice_out, shape)

    def random(self, size=None):
        seq = self._random_seq
        if size is None:
            return seq
        
        # Normalize size to a total element count and reshape.
        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        total_elements = int(np.prod(shape))

        if seq.size != total_elements:
            seq = np.resize(seq, total_elements)
        return seq.reshape(shape)
    
    def __getattr__(self, name):
        # Fail loudly if the production code starts using an unimplemented RNG method.
        raise AttributeError(
            f"FakeRNG does not implement '{name}'. If production code calls this, extend FakeRNG "
            "to include that method (e.g., integers, normal, etc.)."
        )


class TestSiftingLogic:
    """Dedicated tests for the corrected sifting and error model."""
    def test_misalignment_is_deterministic_with_mock_rng(self, small_params):
        """
        Verifies that the misalignment model produces a predictable number of
        errors when the random number generator is fully controlled.
        """
        num_pulses = 20
        params_dict = copy.deepcopy(small_params.to_dict())
        params_dict['misalignment'] = 0.1
        params_dict['qber_intrinsic'] = 0.0
        params = QKDParams.from_dict(params_dict)
        
        # Check for immutability as a defensive measure.
        params_before = copy.deepcopy(params_dict)
        
        real_rng = np.random.default_rng(42)
        alice_bits = real_rng.integers(0, 2, size=num_pulses, dtype=np.int8)
        alice_bases = np.zeros(num_pulses, dtype=np.int8)
        signal_click = np.ones(num_pulses, dtype=bool)
        dark0, dark1 = np.zeros_like(signal_click), np.zeros_like(signal_click)

        assert alice_bits.shape == (num_pulses,)

        # Setup the deterministic fake RNG with robust checks.
        choice_out = np.zeros(num_pulses, dtype=np.int8)
        # Use np.linspace directly to create a predictable sequence, as the helper is unreliable.
        random_seq = np.linspace(0.0, 1.0, num_pulses, endpoint=False)
        assert random_seq.shape == (num_pulses,), "Test helper must return 1D array."
        assert np.all((random_seq >= 0.0) & (random_seq < 1.0)), "RNG values must be in [0,1)"
        fake_rng = FakeRNG(choice_out=choice_out, random_seq=random_seq)

        sifted, errors, _ = qkd_simulation_main._sifting_and_errors(
            params, num_pulses, alice_bits, alice_bases,
            signal_click, dark0, dark1, fake_rng
        )
        
        # Ensure the params object was not mutated by the function call.
        assert params.to_dict() == params_before

        # Flexible post-condition checks for output shapes and types.
        assert isinstance(errors, np.ndarray)
        assert errors.shape == (num_pulses,)
        assert np.issubdtype(errors.dtype, np.bool_) or np.issubdtype(errors.dtype, np.integer)

        threshold = params.misalignment
        expected_errors = int(np.count_nonzero(random_seq < threshold))
        actual_errors = int(np.count_nonzero(errors))
        
        assert actual_errors == expected_errors, \
            f"Expected {expected_errors} errors, got {actual_errors}"

    def test_sifting_zero_pulses_returns_empty(self, small_params):
        """Ensures the function handles zero-pulse inputs gracefully."""
        params = QKDParams.from_dict(copy.deepcopy(small_params.to_dict()))
        empty_int = np.array([], dtype=np.int8)
        empty_bool = np.array([], dtype=bool)
        fake_rng = FakeRNG(choice_out=empty_int, random_seq=np.array([]))

        s, e, _ = qkd_simulation_main._sifting_and_errors(
            params, 0, empty_int, empty_int,
            empty_bool, empty_bool, empty_bool, fake_rng
        )
        assert s.size == 0 and e.size == 0

    def test_sifting_full_misalignment_flips_all(self, small_params):
        """Ensures that a 100% misalignment probability flips all sifted bits."""
        num_pulses = 10
        params_dict = copy.deepcopy(small_params.to_dict())
        # Use a value close to 1.0 to satisfy the validation constraint `misalignment < 1.0`.
        params_dict['misalignment'] = 0.999999999
        params_dict['qber_intrinsic'] = 0.0
        params = QKDParams.from_dict(params_dict)

        alice_bits = np.zeros(num_pulses, dtype=np.int8)
        alice_bases = np.zeros(num_pulses, dtype=np.int8)
        signal_click = np.ones(num_pulses, dtype=bool)
        dark0, dark1 = np.zeros(num_pulses, dtype=bool), np.zeros(num_pulses, dtype=bool)
        
        fake_rng = FakeRNG(
            choice_out=np.zeros(num_pulses, dtype=np.int8),
            random_seq=np.full(num_pulses, 0.5)
        )
        
        sifted, errors, _ = qkd_simulation_main._sifting_and_errors(
            params, num_pulses, alice_bits, alice_bases,
            signal_click, dark0, dark1, fake_rng
        )
        
        assert int(np.count_nonzero(sifted)) == num_pulses
        assert int(np.count_nonzero(errors)) == num_pulses

    def test_sifting_and_errors_contract(self, small_params):
        """An explicit contract test to verify the return schema of the private helper."""
        params = QKDParams.from_dict(copy.deepcopy(small_params.to_dict()))
        num_pulses = 5
        bits = np.zeros(num_pulses, dtype=np.int8)
        bases = np.zeros(num_pulses, dtype=np.int8)
        clicks = np.ones(num_pulses, dtype=bool)
        fake_rng = FakeRNG(choice_out=np.zeros(num_pulses, dtype=np.int8), random_seq=np.zeros(num_pulses))
        
        sifted, errors, discarded = qkd_simulation_main._sifting_and_errors(
            params, num_pulses, bits, bases, clicks, np.zeros_like(clicks), np.zeros_like(clicks), fake_rng
        )
        
        assert isinstance(sifted, np.ndarray)
        assert isinstance(errors, np.ndarray)
        assert isinstance(discarded, np.ndarray)
        assert sifted.shape == (num_pulses,)
        assert errors.shape == (num_pulses,)
        assert discarded.shape == (num_pulses,)
