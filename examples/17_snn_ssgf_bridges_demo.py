# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Cross-repo bridge demos: SNN adapter + SSGF quantum loop."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.snn_adapter import SNNQuantumBridge, spike_train_to_rotations
from scpn_quantum_control.bridge.ssgf_adapter import (
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)
from scpn_quantum_control.identity.binding_spec import (
    orchestrator_to_quantum_phases,
    quantum_to_orchestrator_phases,
)


def snn_demo() -> None:
    print("SNN Adapter Demo (pure numpy, no sc-neurocore)")
    print("-" * 50)
    rng = np.random.default_rng(42)
    spikes = (rng.random((20, 3)) > 0.6).astype(float)
    angles = spike_train_to_rotations(spikes, window=10)
    print(f"  Spike history: {spikes.shape}")
    print(f"  Rotation angles: {angles}")

    bridge = SNNQuantumBridge(n_neurons=2, n_inputs=3, seed=42)
    out = bridge.forward(spikes)
    print(f"  Quantum output currents: {out}")


def ssgf_demo() -> None:
    print("\nSSGF Adapter Demo (standalone functions)")
    print("-" * 50)
    W = np.array([[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]])
    omega = np.array([1.0, 1.5, 2.0])
    H = ssgf_w_to_hamiltonian(W, omega)
    print(f"  W matrix: {W.shape}, {int(np.count_nonzero(W))} nonzero entries")
    print(f"  Hamiltonian: {H.num_qubits} qubits, {len(H)} terms")

    theta = np.array([0.5, 1.2, 2.8])
    qc = ssgf_state_to_quantum({"theta": theta})
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(qc)
    recovered = quantum_to_ssgf_state(sv, 3)
    print(f"  Input θ:     {theta}")
    print(f"  Recovered θ: {recovered['theta']}")
    print(f"  R_global:    {recovered['R_global']:.4f}")


def orchestrator_demo() -> None:
    print("\nOrchestrator Phase Mapping Demo")
    print("-" * 50)
    theta_18 = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    orch = quantum_to_orchestrator_phases(theta_18)
    print(f"  Quantum oscillators: {len(theta_18)}")
    print(f"  Orchestrator oscillators: {len(orch)}")
    theta_back = orchestrator_to_quantum_phases(orch)
    diff = np.angle(np.exp(1j * (theta_back - theta_18)))
    print(f"  Roundtrip max error: {np.max(np.abs(diff)):.2e}")


def main() -> None:
    print("Cross-Repo Bridge Demos")
    print("=" * 50)
    snn_demo()
    ssgf_demo()
    orchestrator_demo()


if __name__ == "__main__":
    main()
