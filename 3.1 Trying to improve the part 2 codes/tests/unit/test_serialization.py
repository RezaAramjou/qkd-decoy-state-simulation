# -*- coding: utf-8 -*-
"""
Unit tests for serialization, deserialization, and helper methods of QKD dataclasses.
This version has been hardened based on expert review to be more robust and less brittle.
"""
import pytest
import copy
from qkd_simulation_main import QKDParams, PulseTypeConfig, ParameterValidationError

def _clone(params_dict):
    """Helper to reduce repetitive deepcopy calls in tests."""
    return copy.deepcopy(params_dict)

class TestSerializationAndHelpers:
    """Tests for data serialization, deserialization, and helper methods."""

    def test_from_dict_normalizes_pulse_dicts(self, default_params_dict):
        """
        Ensures the from_dict classmethod correctly normalizes pulse
        configurations from dictionaries into PulseTypeConfig objects.
        """
        params = QKDParams.from_dict(default_params_dict)
        assert all(isinstance(pc, PulseTypeConfig) for pc in params.pulse_configs)

    def test_qkdparams_to_from_dict_roundtrip(self, default_params_dict):
        """
        Ensures that serializing QKDParams to a dictionary and back
        results in an equivalent object, using robust, order-independent comparison.
        """
        params1 = QKDParams.from_dict(default_params_dict)
        params_dict = params1.to_dict()
        params2 = QKDParams.from_dict(params_dict)

        # Use defensive copies for comparison to avoid side effects
        p1_dict = _clone(params1.to_dict())
        p2_dict = _clone(params2.to_dict())
        p1_dict.pop('pulse_configs', None)
        p2_dict.pop('pulse_configs', None)
        assert p1_dict == p2_dict

        # For pulse_configs, compare them in an order-independent way
        pc_map1 = {pc.name: pc for pc in params1.pulse_configs}
        pc_map2 = {pc.name: pc for pc in params2.pulse_configs}
        assert set(pc_map1.keys()) == set(pc_map2.keys())
        for name in pc_map1:
            assert pc_map1[name].mean_photon_number == pytest.approx(pc_map2[name].mean_photon_number)
            assert pc_map1[name].probability == pytest.approx(pc_map2[name].probability)

    def test_to_dict_produces_canonical_schema(self, default_params_dict):
        """
        Verifies that to_dict() produces a dictionary with the correct keys and types.
        """
        params = QKDParams.from_dict(default_params_dict)
        out = params.to_dict()
        
        assert isinstance(out['num_bits'], int)
        assert isinstance(out['pulse_configs'], list)
        assert out['pulse_configs'], "Serialized pulse_configs should not be empty"
        
        for pc in out['pulse_configs']:
            assert set(pc.keys()) == {"name", "mean_photon_number", "probability"}
            assert isinstance(pc['name'], str)
            assert isinstance(pc['mean_photon_number'], float)
            assert isinstance(pc['probability'], float)

    def test_get_pulse_config_by_name(self, default_params_dict):
        """
        Verifies get_pulse_config_by_name correctly retrieves a pulse
        configuration and does not perform partial matches.
        """
        assert default_params_dict.get('pulse_configs'), "Fixture must have pulse configs for this test."
        params = QKDParams.from_dict(default_params_dict)
        
        expected_name = default_params_dict['pulse_configs'][0]['name']
        assert len(expected_name) > 1, "Fixture pulse name must be >1 char for partial-match test"
        
        pc = params.get_pulse_config_by_name(expected_name)
        assert isinstance(pc, PulseTypeConfig)
        assert pc.name == expected_name
        
        assert params.get_pulse_config_by_name("non_existent_pulse") is None
        assert params.get_pulse_config_by_name(expected_name[:-1]) is None

    def test_get_pulse_config_by_name_handles_duplicates(self, default_params_dict):
        """Ensures get_pulse_config_by_name returns the first match if names are duplicated."""
        d = _clone(default_params_dict)
        assert d.get('pulse_configs'), "Fixture must have pulse configs for this test."
        
        # Add a duplicate pulse config with a different probability
        duplicate_pulse = _clone(d['pulse_configs'][0])
        duplicate_pulse['probability'] = 0.001
        # Adjust another probability to keep the sum at 1.0, avoiding a validation error
        d['pulse_configs'][1]['probability'] -= 0.001
        d['pulse_configs'].append(duplicate_pulse)
        
        params = QKDParams.from_dict(d)
        
        # Contract: The method should return the first encountered instance.
        retrieved_pc = params.get_pulse_config_by_name(duplicate_pulse['name'])
        assert retrieved_pc is not None
        assert retrieved_pc.probability == pytest.approx(d['pulse_configs'][0]['probability'])

    def test_from_dict_with_missing_key_raises(self, default_params_dict):
        """Ensures from_dict raises a specific error if a required key is missing."""
        bad_dict = _clone(default_params_dict)
        del bad_dict['num_bits']
        
        with pytest.raises(ParameterValidationError) as exc:
            QKDParams.from_dict(bad_dict)
        
        assert "Missing required parameter" in str(exc.value)
        assert "num_bits" in str(exc.value)

    def test_from_dict_with_unknown_key_raises(self, default_params_dict):
        """Ensures from_dict raises an error if an unknown key is provided."""
        bad_dict = _clone(default_params_dict)
        bad_dict['unknown_parameter'] = "some_value"
        
        with pytest.raises(ParameterValidationError) as exc:
            QKDParams.from_dict(bad_dict)
        assert "Unknown parameter" in str(exc.value)

    def test_from_dict_with_missing_key_in_pulse_config_raises(self, default_params_dict):
        """Ensures from_dict raises an error for a missing key in a nested pulse_config."""
        bad_dict = _clone(default_params_dict)
        assert bad_dict.get('pulse_configs'), "Fixture must have pulse configs for this test."
        del bad_dict['pulse_configs'][0]['probability']
        
        with pytest.raises(ParameterValidationError) as exc:
            QKDParams.from_dict(bad_dict)
        
        assert "is missing required key" in str(exc.value)
        assert "probability" in str(exc.value)

    def test_from_dict_with_invalid_value_in_pulse_config_raises(self, default_params_dict):
        """Ensures from_dict raises an error for an invalid value in a nested pulse_config."""
        bad_dict = _clone(default_params_dict)
        assert bad_dict.get('pulse_configs'), "Fixture must have pulse configs for this test."
        bad_dict['pulse_configs'][0]['mean_photon_number'] = -0.1
        with pytest.raises(ParameterValidationError, match="mean_photon_number must be non-negative"):
            QKDParams.from_dict(bad_dict)
            
    def test_from_dict_ignores_unknown_key_in_pulse_config(self, default_params_dict):
        """
        Ensures from_dict ignores unknown keys in nested pulse_configs,
        reflecting the current implementation's permissive behavior.
        """
        # Contract: The current implementation ignores unknown keys in nested objects.
        # This test verifies that behavior. A stricter API would require a change
        # in qkd_simulation_main.py, and this test would need to be updated.
        bad_dict = _clone(default_params_dict)
        assert bad_dict.get('pulse_configs'), "Fixture must have pulse configs for this test."
        bad_dict['pulse_configs'][0]['unexpected_field'] = 'test'
        
        # No exception should be raised.
        try:
            QKDParams.from_dict(bad_dict)
        except ParameterValidationError as e:
            pytest.fail(f"Unexpected ParameterValidationError was raised for an unknown nested key: {e}")
