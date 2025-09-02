#!/bin/bash

# ==============================================================================
# QKD Simulation Framework - Final Comprehensive Test Script
# ==============================================================================
# This script automates four key performance analyses: SKL vs. Distance,
# Detector Efficiency, Block Size, and Error Rates. It uses a larger number
# of bits for more accurate results.
#
# Instructions:
# 1. Save this file as `run_final_tests.sh` in your project's root directory.
# 2. Make it executable: chmod +x run_final_tests.sh
# 3. Run the script: ./run_final_tests.sh
# ==============================================================================

# --- Configuration ---
PYTHON_CMD="python3 -m scripts.run_qkd"
TEST_DIR="final_comprehensive_tests"
BASE_CONFIG_PATH="${TEST_DIR}/configs/base_config.json"

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
  
  # We run the command without redirecting output to see any errors.
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

# 2. Set up the test directory and create a corrected base configuration file.
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
mkdir -p "${TEST_DIR}/configs"
echo "Created temporary directory for tests at: ./${TEST_DIR}/"

# CORRECTED: Added the missing eps_pe and eps_smooth parameters.
cat << EOF > "$BASE_CONFIG_PATH"
{
  "protocol_name": "bb84-decoy",
  "protocol_config": { "alice_z_basis_prob": 0.5, "bob_z_basis_prob": 0.5 },
  "source_config": {
    "pulse_configs": [
      {"name": "signal", "mean_photon_number": 0.5, "probability": 1.0}
    ]
  },
  "channel_config": { "distance_km": 10.0, "fiber_loss_db_km": 0.2 },
  "detector_config": {
    "det_eff_d0": 0.9, "det_eff_d1": 0.9, "dark_rate": 1e-8,
    "qber_intrinsic": 0.001, "misalignment": 0.005,
    "afterpulse_prob": 0.01, "afterpulse_memory": 5, "dead_time_ns": 50.0,
    "double_click_policy": "DISCARD"
  },
  "num_bits": 10000000, "photon_number_cap": 12, "batch_size": 1000000,
  "num_workers": 4, "f_error_correction": 1.16,
  "eps_sec": 1e-9, "eps_cor": 1e-15, "eps_pe": 1e-10, "eps_smooth": 1e-10,
  "security_proof": "tight-proof", "ci_method": "clopper-pearson",
  "enforce_monotonicity": true, "assume_phase_equals_bit_error": false
}
EOF

# --- TEST SUITE 1: SKL vs. Distance ---
print_header "Test Suite 1: SKL vs. Distance"
DISTANCES=(25 50 75 100 125)
for dist in "${DISTANCES[@]}"; do
  test_name="distance_${dist}km"
  new_config="${TEST_DIR}/configs/${test_name}.json"
  jq --argjson d "$dist" '.channel_config.distance_km = $d' "$BASE_CONFIG_PATH" > "$new_config"
  run_simulation "$test_name" "$new_config"
done

# --- TEST SUITE 2: SKL vs. Detector Efficiency ---
print_header "Test Suite 2: SKL vs. Detector Efficiency (at 25km)"
EFFICIENCIES=(0.9 0.5 0.2 0.1)
for eff in "${EFFICIENCIES[@]}"; do
  test_name="efficiency_${eff//./p}" # Replace dot with 'p' for filename
  new_config="${TEST_DIR}/configs/${test_name}.json"
  jq --argjson e "$eff" '.detector_config.det_eff_d0 = $e | .detector_config.det_eff_d1 = $e | .channel_config.distance_km = 25' "$BASE_CONFIG_PATH" > "$new_config"
  run_simulation "$test_name" "$new_config"
done

# --- TEST SUITE 3: SKL vs. Block Size (num_bits) ---
print_header "Test Suite 3: SKL vs. Block Size (at 50km)"
BLOCK_SIZES=(1000000 10000000 100000000) # 1M, 10M, 100M bits
for bits in "${BLOCK_SIZES[@]}"; do
  test_name="block_size_${bits}"
  new_config="${TEST_DIR}/configs/${test_name}.json"
  jq --argjson b "$bits" '.num_bits = $b | .channel_config.distance_km = 50' "$BASE_CONFIG_PATH" > "$new_config"
  run_simulation "$test_name" "$new_config"
done

# --- TEST SUITE 4: SKL vs. Error Rates ---
print_header "Test Suite 4: SKL vs. Error Rates (at 10km)"
# Test with higher intrinsic QBER
test_name="high_qber"
new_config="${TEST_DIR}/configs/${test_name}.json"
jq '.detector_config.qber_intrinsic = 0.02' "$BASE_CONFIG_PATH" > "$new_config"
run_simulation "$test_name" "$new_config"

# Test with higher dark count rate
test_name="high_dark_rate"
new_config="${TEST_DIR}/configs/${test_name}.json"
jq '.detector_config.dark_rate = 1e-6' "$BASE_CONFIG_PATH" > "$new_config"
run_simulation "$test_name" "$new_config"

print_header "All tests completed."
echo "Generated configs and outputs are located in the '${TEST_DIR}' directory."

