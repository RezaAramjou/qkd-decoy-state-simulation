#!/bin/bash

# ==============================================================================
# QKD Simulation Framework - Automated Test Script
# ==============================================================================
# This script automates a series of tests by modifying a base JSON configuration
# file and running the QKD simulation for each case.
#
# Instructions:
# 1. Save this file as `run_tests.sh` in your project's root directory.
# 2. Open your terminal in the same directory.
# 3. Make the script executable by running: chmod +x run_tests.sh
# 4. Run the script: ./run_tests.sh
# ==============================================================================

# --- Configuration ---
# The base configuration file to use for all tests.
BASE_CONFIG="configs/test_stateful_detector.json"
# The Python command to execute the simulation.
PYTHON_CMD="python -m scripts.run_qkd"
# Directory to store generated test configurations and outputs.
TEST_DIR="automated_tests"


# --- Helper Functions ---
# Function to print a formatted header for each test section.
print_header() {
  echo ""
  echo "=============================================================================="
  echo "$1"
  echo "=============================================================================="
}

# Function to run a single simulation and report the result.
run_simulation() {
  local test_name="$1"
  local config_file="$2"
  local output_file="${TEST_DIR}/output_${test_name}.json"

  echo "-> Running test: ${test_name}"
  echo "   Config: ${config_file}"
  
  # Execute the simulation command. The --force flag overwrites previous results.
  # We redirect stdout and stderr to /dev/null to keep the output clean.
  $PYTHON_CMD "$config_file" -o "$output_file" --force > /dev/null 2>&1

  # Check if the simulation was successful (exit code 0).
  if [ $? -eq 0 ]; then
    # If successful, parse the output JSON to find the secure key length.
    local key_length=$(jq '.secure_key_length' "$output_file")
    echo "   SUCCESS: Final Secure Key Length = ${key_length}"
  else
    # If the simulation failed, report an error.
    echo "   ERROR: Simulation failed. Check logs for details."
  fi
  echo ""
}


# --- Main Script ---

# 1. Check for dependencies (jq).
if ! command -v jq &> /dev/null; then
  echo "Error: 'jq' is not installed. It is required to modify the JSON config files."
  echo "Please install it by running: sudo apt update && sudo apt install jq"
  exit 1
fi

# 2. Set up the test directory.
rm -rf "$TEST_DIR" # Clean up previous runs
mkdir -p "$TEST_DIR"
mkdir -p "${TEST_DIR}/configs"
echo "Created temporary directory for tests at: ./${TEST_DIR}/"

# 3. --- TEST SUITE 1: Impact of Distance ---
print_header "Test Suite 1: Analyzing the Impact of Distance"
DISTANCES=(25 50 75 100)
for dist in "${DISTANCES[@]}"; do
  test_name="distance_${dist}km"
  new_config="${TEST_DIR}/configs/${test_name}.json"
  # Use jq to modify the distance_km field and create a new config file.
  jq --argjson d "$dist" '.channel_config.distance_km = $d' "$BASE_CONFIG" > "$new_config"
  run_simulation "$test_name" "$new_config"
done

# 4. --- TEST SUITE 2: Impact of Detector Noise ---
print_header "Test Suite 2: Analyzing the Impact of Detector Noise (at 10km)"
# Test with higher dark count rate
test_name="high_dark_rate"
new_config="${TEST_DIR}/configs/${test_name}.json"
jq '.detector_config.dark_rate = 1e-6' "$BASE_CONFIG" > "$new_config"
run_simulation "$test_name" "$new_config"

# Test with higher intrinsic QBER
test_name="high_qber"
new_config="${TEST_DIR}/configs/${test_name}.json"
jq '.detector_config.qber_intrinsic = 0.01' "$BASE_CONFIG" > "$new_config"
run_simulation "$test_name" "$new_config"

# 5. --- TEST SUITE 3: Re-enabling Decoy-State Protocol ---
print_header "Test Suite 3: Testing Full Decoy-State Protocol (at 50km)"
test_name="decoy_protocol_50km"
new_config="${TEST_DIR}/configs/${test_name}.json"
# Use jq to replace the entire pulse_configs array and set the distance.
jq '.source_config.pulse_configs = [
      {"name": "signal", "mean_photon_number": 0.5, "probability": 0.8},
      {"name": "decoy", "mean_photon_number": 0.1, "probability": 0.1},
      {"name": "vacuum", "mean_photon_number": 0.0, "probability": 0.1}
    ] | .channel_config.distance_km = 50' "$BASE_CONFIG" > "$new_config"
run_simulation "$test_name" "$new_config"


print_header "All tests completed."
echo "Generated configs and outputs are located in the '${TEST_DIR}' directory."

