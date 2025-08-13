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
