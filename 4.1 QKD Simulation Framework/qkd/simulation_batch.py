# -*- coding: utf-8 -*-
"""
Low-level batch worker functions for the simulation engine.

This module contains the core logic for simulating a single batch of a QKD
protocol. It is designed to be executed in a separate worker process and
includes robust error handling, validation, and performance optimizations.
"""
import dataclasses
import logging
import os
import pickle
import traceback
from typing import Dict, Any

import numpy as np
# Use an alias for Generator for clarity
from numpy.random import Generator as RNG

from .params import QKDParams
from .datatypes import TallyCounts
# Import all necessary classes directly from the modules
from .protocols import (
    MDIQKDProtocol, BB84DecoyProtocol, DetectionResults, SiftingResults,
    MDIPreparedStates, BB84PreparedStates
)
from .exceptions import QKDSimulationError, ParameterValidationError
# Corrected constant import
from .constants import MAX_SEED_INT_PCG64 as MAX_SEED_INT

__all__ = ["_simulate_quantum_part_batch", "_top_level_worker_function", "_merge_batch_tallies"]

logger = logging.getLogger(__name__)

def _check_for_termination():
    """Checks if the global termination event has been set."""
    if 'globals' in locals() and '_qkd_terminate_event' in globals():
        if globals()['_qkd_terminate_event'].is_set():
            raise InterruptedError("Termination event received by worker.")

def _simulate_quantum_part_batch(p: QKDParams, num_pulses: int, rng: RNG) -> Dict[str, Any]:
    """
    Simulates one batch of the quantum part of the protocol.
    """
    if getattr(p, 'reset_detector_state_per_batch', True):
        p.detector.reset_state()

    _check_for_termination()
    # This call is now correct because p.protocol is an instance of a Protocol subclass
    prepared_states = p.protocol.prepare_states(num_pulses, rng)

    alice_bases = prepared_states.alice_bases
    alice_bits = prepared_states.alice_bits

    if isinstance(p.protocol, MDIQKDProtocol):
        if not isinstance(prepared_states, MDIPreparedStates):
             raise QKDSimulationError("MDIQKDProtocol did not return MDIPreparedStates.")
        photons_A = p.source.generate_photons(prepared_states.alice_pulse_type_indices, rng)
        photons_B = p.source.generate_photons(prepared_states.bob_pulse_type_indices, rng)
        photon_numbers = np.minimum(photons_A, photons_B)
        channel_transmittance = p.channel.transmittance**2
        ideal_outcomes_d0 = rng.random(num_pulses) < 0.5
    elif isinstance(p.protocol, BB84DecoyProtocol):
        if not isinstance(prepared_states, BB84PreparedStates):
            raise QKDSimulationError("BB84DecoyProtocol did not return BB84PreparedStates.")
        photon_numbers = p.source.generate_photons(prepared_states.alice_pulse_type_indices, rng)
        channel_transmittance = p.channel.transmittance
        bob_bases_temp = prepared_states.bob_bases
        basis_match_temp = alice_bases == bob_bases_temp
        ideal_outcomes_d0 = (alice_bits == 0)
        ideal_outcomes_d0 = np.where(basis_match_temp, ideal_outcomes_d0, rng.random(num_pulses) < 0.5)
    else:
        raise NotImplementedError(f"Simulation logic not implemented for protocol type: {type(p.protocol).__name__}")

    _check_for_termination()
    detection_result_tuple = p.detector.simulate_detection(
        channel_transmittance=channel_transmittance,
        photon_numbers=photon_numbers,
        rng=rng,
        ideal_outcomes_d0=ideal_outcomes_d0,
        return_diagnostics=True
    )

    detection_results_for_protocol = DetectionResults(
        num_pulses=num_pulses,
        click0=detection_result_tuple.click0,
        click1=detection_result_tuple.click1
    )

    sifting_results: SiftingResults = p.protocol.sift_results(prepared_states, detection_results_for_protocol, rng)
    sifted_mask = sifting_results.sifted_mask
    errors_mask = sifting_results.error_mask

    # --- Vectorized Tallying for Performance ---
    batch_tallies = {}
    indices = prepared_states.alice_pulse_type_indices
    num_pulse_types = len(p.source.pulse_configs)

    if indices.size > 0 and (np.min(indices) < 0 or np.max(indices) >= num_pulse_types):
        raise QKDSimulationError(f"Invalid pulse indices generated. Min: {np.min(indices)}, Max: {np.max(indices)}")

    z_basis_mask = (alice_bases == 0)
    sent_counts = np.bincount(indices, minlength=num_pulse_types).astype(np.int64)
    sifted_counts = np.bincount(indices[sifted_mask], minlength=num_pulse_types).astype(np.int64)
    errors_counts = np.bincount(indices[errors_mask], minlength=num_pulse_types).astype(np.int64)
    sent_z_counts = np.bincount(indices[z_basis_mask], minlength=num_pulse_types).astype(np.int64)
    sifted_z_counts = np.bincount(indices[sifted_mask & z_basis_mask], minlength=num_pulse_types).astype(np.int64)
    errors_z_counts = np.bincount(indices[errors_mask & z_basis_mask], minlength=num_pulse_types).astype(np.int64)
    
    for i, pc in enumerate(p.source.pulse_configs):
        t = TallyCounts(
            sent=int(sent_counts[i]),
            sent_z=int(sent_z_counts[i]),
            sent_x=int(sent_counts[i] - sent_z_counts[i]),
            sifted=int(sifted_counts[i]),
            sifted_z=int(sifted_z_counts[i]),
            sifted_x=int(sifted_counts[i] - sifted_z_counts[i]),
            errors_sifted=int(errors_counts[i]),
            errors_sifted_z=int(errors_z_counts[i]),
            errors_sifted_x=int(errors_counts[i] - errors_z_counts[i]),
            double_clicks_discarded=0,
        )
        batch_tallies[pc.name] = dataclasses.asdict(t)

    return {"tallies": batch_tallies, "sifted_count": int(np.sum(sifted_mask)), "num_pulses": num_pulses}


def _top_level_worker_function(serialized_params: Dict, num_pulses: int, seed: int) -> Dict:
    """ Top-level function executed by each worker process. """
    try:
        if "params_file" in serialized_params:
            with open(serialized_params["params_file"], 'rb') as f:
                params_dict = pickle.load(f)
        else:
            params_dict = serialized_params

        deserialized_params = QKDParams.from_dict(params_dict)
        safe_seed = int(seed) % MAX_SEED_INT or 1
        rng = np.random.default_rng(safe_seed)
        return _simulate_quantum_part_batch(deserialized_params, num_pulses, rng)
    except (KeyboardInterrupt, InterruptedError):
        raise
    except Exception as e:
        tb = traceback.format_exc(limit=20)
        logger.error(f"Error in worker process (seed={seed}): {type(e).__name__}: {e}\n{tb}")
        raise RuntimeError(f"Worker error (seed={seed}): {type(e).__name__}: {e}") from e


def _merge_batch_tallies(overall_stats: Dict[str, TallyCounts], batch_result: Dict):
    """ Merges tally counts from a worker batch into the main statistics object. """
    for name, tally_data in batch_result.get("tallies", {}).items():
        stats_obj = overall_stats.setdefault(name, TallyCounts())
        for key, val in tally_data.items():
            if hasattr(stats_obj, key):
                current_val = getattr(stats_obj, key)
                new_val = current_val + int(val)
                if new_val < current_val:
                    raise QKDSimulationError(f"Tally count overflow for '{name}.{key}'.")
                setattr(stats_obj, key, new_val)

