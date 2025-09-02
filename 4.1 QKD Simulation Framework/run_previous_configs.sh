#!/bin/bash

# ==============================================================================
# QKD Simulation Framework - Legacy Configuration Test Script (Debug Version)
# ==============================================================================
# This script creates and runs four specific configurations from a previous
# version of the model, adapted for the new framework structure.
#
# Instructions:
# 1. Save this file as `run_previous_configs.sh` in your project's root directory.
# 2. Make it executable: chmod +x run_previous_configs.sh
# 3. Run the script: ./run_previous_configs.sh
# ==============================================================================

# --- Configuration ---
# CORRECTED: Use python3 to ensure the correct interpreter is found.
PYTHON_CMD="python3 -m scripts.run_qkd"
TEST_DIR="legacy_tests"

# --- Helper Functions ---
print_header() {
  echo ""
  echo "=============================================================================="
  echo "$1"
  echo "=============================================================================="
}

run_simulation() {
  local test_name="$1"
  local config_file="$2"
  local output_file="${TEST_DIR}/output_${test_name}.json"

  echo "-> Running test: ${test_name}"
  echo "   Config: ${config_file}"
  
  # MODIFIED: Removed output redirection to show the full Python error traceback.
  $PYTHON_CMD "$config_file" -o "$output_file" --force

  if [ $? -eq 0 ]; then
    local key_length=$(jq '.secure_key_length' "$output_file")
    echo "   SUCCESS: Final Secure Key Length = ${key_length}"
  else
    echo "   ERROR: Simulation failed. See traceback above for details."
  fi
  echo ""
}

# --- Main Script ---
# 1. Check for dependencies (jq).
if ! command -v jq &> /dev/null; then
  echo "Error: 'jq' is not installed. Please run: sudo apt update && sudo apt install jq"
  exit 1
fi

# 2. Set up the test directory.
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
mkdir -p "${TEST_DIR}/configs"
echo "Created temporary directory for tests at: ./${TEST_DIR}/"

# 3. --- Create and Run Test Configurations ---

print_header "Creating and Running Legacy Test Configurations"

# --- Config 1: 50km Distance ---
CONFIG_1_PATH="${TEST_DIR}/configs/legacy_config_1.json"
cat << EOF > "$CONFIG_1_PATH"
{
  "protocol_name": "bb84-decoy",
  "protocol_config": { "alice_z_basis_prob": 0.5, "bob_z_basis_prob": 0.5 },
  "source_config": {
    "pulse_configs": [
      {"name": "signal", "mean_photon_number": 0.5, "probability": 0.7},
      {"name": "decoy", "mean_photon_number": 0.1, "probability": 0.15},
      {"name": "vacuum", "mean_photon_number": 0.0, "probability": 0.15}
    ]
  },
  "channel_config": { "distance_km": 50.0, "fiber_loss_db_km": 0.2 },
  "detector_config": {
    "det_eff_d0": 0.2, "det_eff_d1": 0.2, "dark_rate": 1e-7,
    "qber_intrinsic": 0.01, "misalignment": 0.015,
    "double_click_policy": "DISCARD"
  },
  "num_bits": 1000000, "photon_number_cap": 12, "batch_size": 100000,
  "num_workers": 4, "f_error_correction": 1.1,
  "eps_sec": 1e-9, "eps_cor": 1e-15, "eps_pe": 1e-10, "eps_smooth": 1e-10,
  "security_proof": "lim-2014", "ci_method": "clopper-pearson",
  "enforce_monotonicity": true, "assume_phase_equals_bit_error": false
}
EOF
run_simulation "legacy_config_1_50km" "$CONFIG_1_PATH"

# --- Config 2, 3, 4: 20km Distance (Identical configs, so we only run it once) ---
CONFIG_2_PATH="${TEST_DIR}/configs/legacy_config_2.json"
cat << EOF > "$CONFIG_2_PATH"
{
  "protocol_name": "bb84-decoy",
  "protocol_config": { "alice_z_basis_prob": 0.5, "bob_z_basis_prob": 0.5 },
  "source_config": {
    "pulse_configs": [
      {"name": "signal", "mean_photon_number": 0.5, "probability": 0.7},
      {"name": "decoy", "mean_photon_number": 0.1, "probability": 0.15},
      {"name": "vacuum", "mean_photon_number": 0.0, "probability": 0.15}
    ]
  },
  "channel_config": { "distance_km": 20.0, "fiber_loss_db_km": 0.2 },
  "detector_config": {
    "det_eff_d0": 0.2, "det_eff_d1": 0.2, "dark_rate": 1e-7,
    "qber_intrinsic": 0.01, "misalignment": 0.015,
    "double_click_policy": "DISCARD"
  },
  "num_bits": 1000000, "photon_number_cap": 12, "batch_size": 100000,
  "num_workers": 4, "f_error_correction": 1.1,
  "eps_sec": 1e-9, "eps_cor": 1e-15, "eps_pe": 1e-10, "eps_smooth": 1e-10,
  "security_proof": "lim-2014", "ci_method": "clopper-pearson",
  "enforce_monotonicity": true, "assume_phase_equals_bit_error": false
}
EOF
run_simulation "legacy_config_2_20km" "$CONFIG_2_PATH"


print_header "All legacy tests completed."
echo "Generated configs and outputs are located in the '${TEST_DIR}' directory."

