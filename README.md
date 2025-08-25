# Decoy-State Quantum Key Distribution Simulator

*A Python-based simulation framework for analyzing the security and performance of the decoy-state Bennett-Brassard 1984 (BB84) Quantum Key Distribution (QKD) protocol.*

This project provides a link-level physics and security simulator designed to model a full QKD system, including a transmitter (Alice), a receiver (Bob), and an eavesdropper (Eve).¹ It specifically evaluates the system's resilience against the Photon-Number-Splitting (PNS) attack by implementing the decoy-state method.

The simulation is structured to support security analysis in both the idealized asymptotic regime (infinite key length) and the more practical finite-key regime, which accounts for real-world statistical limitations.²

---

## Key Features

* **BB84 Protocol Simulation**: Implements the core logic of the polarization-based BB84 protocol.
* **Decoy-State Method**: Models the use of signal, decoy, and vacuum states to detect and defeat the PNS attack.³
* **PNS Attack Modeling**: Explicitly simulates an eavesdropper attempting a Photon-Number-Splitting attack to compromise the secret key.⁵
* **Dual Security Analysis Regimes**:
    * **Asymptotic Analysis**: Calculates the secure key rate assuming an infinite number of transmitted signals.
    * **Finite-Key Analysis**: Accounts for statistical fluctuations from a finite number of signals, providing more realistic security bounds for practical systems.²
* **Performance Metrics**: Calculates critical parameters such as the Quantum Bit Error Rate (QBER) and the final Secure Key Rate (SKR).
* **Parametric Sweeps**: The structure is designed to facilitate sweeping through parameters (e.g., distance, detector efficiency) to generate performance curves.²

---

## Theoretical Background

### The Challenge of Practical QKD

The original BB84 protocol assumes an ideal single-photon source. However, practical QKD systems use attenuated lasers, which produce Weak Coherent Pulses (WCPs). These sources have a non-zero probability of emitting multiple photons in a single pulse.⁴

### Photon-Number-Splitting (PNS) Attack

This multi-photon vulnerability allows an eavesdropper (Eve) to "split" one photon from any multi-photon pulse she intercepts. She can store her copy and forward the rest to Bob. After Alice and Bob publicly reveal their basis choices, Eve can measure her stored photon in the correct basis to learn a bit of the key without introducing any detectable errors, rendering the protocol insecure.⁵

### The Decoy-State Solution

To counter the PNS attack, Alice randomly varies the intensity (mean photon number) of her laser pulses. She uses a primary "signal" state, a weak "decoy" state, and a "vacuum" state (zero intensity). Since Eve cannot distinguish between these states, any attempt to selectively target multi-photon pulses will create a statistical discrepancy in the detection rates (gain) and error rates (QBER) of the different states. By comparing these statistics, Alice and Bob can accurately estimate the channel's properties for the secure single-photon pulses and detect Eve's presence.³

---

## Project Structure

The repository is organized into distinct modules that reflect the different stages of QKD security analysis.²
```
├── 1.Initial codes/      # Simulation for the asymptotic (infinite-key) regime.
├── 2.Finite key/         # Simulation including finite-key effects for practical security.
├── Sweep results/        # Directory for storing output data and plots from simulations.
├── requirements.in       # Project dependencies.
└── requirements.txt      # Pinned versions of all dependencies.
```

---

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/RezaAramjou/qkd-decoy-state-simulation.git](https://github.com/RezaAramjou/qkd-decoy-state-simulation.git)
    cd qkd-decoy-state-simulation
    ```
2.  It is highly recommended to create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required dependencies from the `requirements.txt` file:²
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage (Inferred)

*This guide is inferred from the project's structure, as detailed documentation is not yet available.*

1.  **Navigate to the desired analysis directory.**
2.  **Configure Parameters**: Open the main Python script (e.g., `main.py` or `simulation.py`). Manually edit the physical and protocol parameters defined within the script, such as:
    * Channel length (distance in km)
    * Fiber attenuation (dB/km)
    * Detector efficiency and dark count rate
    * Optical system misalignment error (`e_d`)
    * Mean photon numbers for signal (`mu`) and decoy (`nu`) states
    * Total number of pulses to simulate (especially for finite-key analysis)
3.  **Run the Simulation**:
    * To run the **asymptotic analysis**:
        ```bash
        cd "1.Initial codes"
        python main.py  # Or the relevant script name
        ```
    * To run the **finite-key analysis**:
        ```bash
        cd "2.Finite key"
        python main.py  # Or the relevant script name
        ```
4.  **View Results**: The simulation will likely generate output data files (e.g., `.csv`) and plots (e.g., `.png`) in the `Sweep results/` directory. These may include plots of Secure Key Rate vs. Distance.

---

## Core Formula: Secure Key Rate

The ultimate goal of the simulation is to compute the Secure Key Rate ($R$). In the asymptotic limit, this is often calculated using a formula derived from the GLLP security proof.⁸

$R \ge q \{ Y_1^L [1 - H_2(e_1^U)] - Q_\mu f(E_\mu) H_2(E_\mu) \}$

Where:

* **$q$**: The sifting factor, which accounts for basis mismatch (typically 1/2 for BB84).
* **$Y_1^L$**: The lower bound on the yield of single-photon pulses (estimated using decoy states).
* **$e_1^U$**: The upper bound on the error rate of single-photon pulses (estimated using decoy states).
* **$Q_\mu$**: The overall gain (detection rate) of the signal state (a measured observable).
* **$E_\mu$**: The overall QBER of the signal state (a measured observable).
* **$H_2(x)$**: The binary Shannon entropy function: $H_2(x) = -x \log_2(x) - (1-x) \log_2(1-x)$.
* **$f(E_\mu)$**: The inefficiency of the classical error correction protocol (a factor $\ge 1$).

The finite-key analysis uses a more complex version of this formula that includes additional penalty terms for the finite size of the transmitted data.

---

## Contributing

Contributions to improve this project are welcome. Key areas for development include:
* **Documentation**: Enhancing this README, adding a user guide, and providing extensive inline code comments.
* **Usability**: Implementing a command-line interface (CLI) or configuration files to manage parameters.
* **Verification**: Developing a test suite to verify the simulation's output against results from seminal papers in the field.
* **Feature Expansion**:
    * Modeling other QKD protocols (e.g., MDI-QKD).
    * Simulating more advanced eavesdropping attacks or channel imperfections.

---

## Citation

This project appears to be associated with the thesis research described in:

*A Study of the Quantum Key Distribution Decoy State Protocol Using Model Based Systems Engineering Processes, Methods, and Tools* ¹

If you use this code in your research, please consider citing the repository.
