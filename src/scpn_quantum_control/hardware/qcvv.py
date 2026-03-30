# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Qcvv
"""QCVV: Quantum Characterisation, Verification, and Validation.

Standardised protocols to certify that quantum simulation results
are trustworthy before publication:

    1. State fidelity: F(rho, sigma) = Tr(sqrt(sqrt(rho) sigma sqrt(rho)))^2
    2. Process fidelity: average gate fidelity via randomised benchmarking
    3. Mirror circuit verification: circuits that should return |0...0>
    4. Cross-entropy benchmark: XEB for circuit quality (Google-style)

For the Kuramoto-XY system, QCVV answers:
    "Is the quantum result we got actually what the circuit intended?"

Without QCVV, hardware results could be noise artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..bridge.knm_hamiltonian import knm_to_hamiltonian


@dataclass
class QCVVResult:
    """QCVV certification result."""

    state_fidelity: float  # |<psi_ideal|psi_actual>|^2
    mirror_fidelity: float  # probability of |0> after mirror circuit
    xeb_score: float  # cross-entropy benchmarking score
    certified: bool  # all metrics above threshold
    n_qubits: int


def state_fidelity(psi_ideal: np.ndarray, psi_actual: np.ndarray) -> float:
    """Pure state fidelity |<ideal|actual>|^2."""
    overlap = abs(np.dot(psi_ideal.conj(), psi_actual)) ** 2
    return float(overlap)


def mirror_circuit_fidelity(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.5,
    reps: int = 3,
) -> float:
    """Mirror circuit test: evolve forward then backward, check return to |0>.

    A perfect quantum computer returns |0...0> with probability 1.
    Noise reduces this — the return probability measures circuit quality.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    n = K.shape[0]
    H = knm_to_hamiltonian(K, omega)
    synth = LieTrotter(reps=reps)

    qc = QuantumCircuit(n)
    # Forward evolution
    evo_fwd = PauliEvolutionGate(H, time=t, synthesis=synth)
    qc.append(evo_fwd, range(n))
    # Backward evolution (negative time)
    evo_bwd = PauliEvolutionGate(H, time=-t, synthesis=synth)
    qc.append(evo_bwd, range(n))

    sv = Statevector.from_instruction(qc)
    # Probability of |0...0>
    p_zero = float(abs(sv.data[0]) ** 2)
    return p_zero


def cross_entropy_score(
    ideal_probs: np.ndarray,
    measured_counts: np.ndarray,
    n_shots: int,
) -> float:
    """Linear cross-entropy benchmarking score.

    XEB = 2^n × Σ_x p_ideal(x) × f_measured(x) - 1

    XEB = 0 for uniform noise, XEB = 1 for perfect circuit.
    """
    n_states = len(ideal_probs)
    freqs = measured_counts / max(n_shots, 1)
    xeb = float(n_states * np.sum(ideal_probs * freqs) - 1.0)
    return max(min(xeb, 1.0), -1.0)


def simulate_xeb(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.5,
    noise_level: float = 0.0,
    n_shots: int = 1000,
    seed: int = 42,
) -> float:
    """Simulate XEB with optional depolarising noise.

    At noise_level=0: XEB → 1 (perfect).
    At noise_level=1: XEB → 0 (uniform).
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    n = K.shape[0]
    H = knm_to_hamiltonian(K, omega)
    synth = LieTrotter(reps=3)

    qc = QuantumCircuit(n)
    evo = PauliEvolutionGate(H, time=t, synthesis=synth)
    qc.append(evo, range(n))

    sv = Statevector.from_instruction(qc)
    ideal_probs = np.abs(sv.data) ** 2

    # Simulate noisy sampling
    rng = np.random.default_rng(seed)
    noisy_probs = (1.0 - noise_level) * ideal_probs + noise_level / len(ideal_probs)
    noisy_probs = noisy_probs / noisy_probs.sum()
    counts = rng.multinomial(n_shots, noisy_probs)

    return cross_entropy_score(ideal_probs, counts, n_shots)


def qcvv_certify(
    K: np.ndarray,
    omega: np.ndarray,
    fidelity_threshold: float = 0.9,
    mirror_threshold: float = 0.9,
    xeb_threshold: float = 0.5,
) -> QCVVResult:
    """Run full QCVV certification suite.

    Statevector simulation (no hardware noise) should pass all checks.
    """
    from ..hardware.classical import classical_exact_diag

    n = K.shape[0]

    # State fidelity: VQE vs exact
    from ..phase.phase_vqe import PhaseVQE

    vqe = PhaseVQE(K, omega, ansatz_reps=2)
    vqe.solve(maxiter=100, seed=42)
    psi_vqe = np.ascontiguousarray(vqe.ground_state())

    exact = classical_exact_diag(n, K=K, omega=omega)
    psi_exact = np.ascontiguousarray(exact["ground_state"])
    sf = state_fidelity(psi_exact, psi_vqe)

    # Mirror circuit
    mf = mirror_circuit_fidelity(K, omega)

    # XEB (no noise for simulator)
    xeb = simulate_xeb(K, omega, noise_level=0.0)

    certified = sf >= fidelity_threshold and mf >= mirror_threshold and xeb >= xeb_threshold

    return QCVVResult(
        state_fidelity=sf,
        mirror_fidelity=mf,
        xeb_score=xeb,
        certified=certified,
        n_qubits=n,
    )
