# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Evs
"""Quantum-enhanced EVS (Emergent Value Signature) for CCW.

The EVS in CCW Standalone uses classical oscillator synchronisation
to detect emergent cognitive signatures. The quantum enhancement maps
EVS features through the Kuramoto-XY Hamiltonian to amplify:

    1. Phase coherence: quantum R_global more sensitive to weak coupling
    2. Topological features: H1 persistence from quantum TCBO
    3. Entanglement structure: quantum correlations beyond classical

Pipeline:
    EVS features (from CCW) → encode as coupling modulation
    → quantum kernel evaluation → enhanced feature vector
    → return to CCW for classification

This bridges the quantum-control module back to the original
CCW consciousness detection framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from ..bridge.ssgf_adapter import (
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)


@dataclass
class QuantumEVSResult:
    """Quantum-enhanced EVS feature vector."""

    classical_features: np.ndarray  # input EVS features
    quantum_features: np.ndarray  # enhanced features
    r_global: float  # quantum synchronisation
    p_h1_proxy: float  # topological feature
    enhancement_factor: float  # ||quantum|| / ||classical||


def _evs_to_coupling(features: np.ndarray, n_osc: int) -> np.ndarray:
    """Map EVS feature vector to coupling modulation.

    Features modulate the base K_nm: stronger features → stronger coupling.
    """
    K_base = build_knm_paper27(L=n_osc)
    n_features = len(features)

    # Modulate upper triangle by cycling through features
    modulation = np.ones_like(K_base)
    pair_idx = 0
    for i in range(n_osc):
        for j in range(i + 1, n_osc):
            feat_idx = pair_idx % n_features
            mod = 0.5 + 0.5 * float(np.clip(features[feat_idx], -1, 1))
            modulation[i, j] = modulation[j, i] = mod
            pair_idx += 1

    K_modulated: np.ndarray = K_base * modulation
    return K_modulated


def quantum_evs_enhance(
    features: np.ndarray,
    n_osc: int = 8,
    dt: float = 0.1,
    trotter_reps: int = 3,
) -> QuantumEVSResult:
    """Enhance EVS features through quantum Kuramoto evolution.

    Args:
        features: classical EVS feature vector (any length)
        n_osc: number of quantum oscillators
        dt: evolution time
        trotter_reps: Trotter repetitions
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.synthesis import LieTrotter

    K = _evs_to_coupling(features, n_osc)
    omega = OMEGA_N_16[:n_osc]

    # Encode features as initial phases
    theta_init = np.zeros(n_osc)
    for i in range(min(len(features), n_osc)):
        theta_init[i] = float(features[i]) * np.pi

    H = ssgf_w_to_hamiltonian(K, omega)
    qc_init = ssgf_state_to_quantum({"theta": theta_init})
    sv = Statevector.from_instruction(qc_init)

    synth = LieTrotter(reps=trotter_reps)
    evo = PauliEvolutionGate(H, time=dt, synthesis=synth)
    step_qc = QuantumCircuit(n_osc)
    step_qc.append(evo, range(n_osc))
    sv = sv.evolve(step_qc)

    state = quantum_to_ssgf_state(sv, n_osc)
    r_global = state["R_global"]
    theta_out = state["theta"]

    # Quantum features: R, phases, pairwise correlators
    q_features: list[float] = [r_global]
    for t in theta_out:
        q_features.append(float(np.cos(t)))
        q_features.append(float(np.sin(t)))

    # Pairwise XY correlators (sampled)
    for i in range(min(n_osc, 4)):
        for j in range(i + 1, min(n_osc, 4)):
            xx = ["I"] * n_osc
            xx[i] = "X"
            xx[j] = "X"
            op = SparsePauliOp("".join(reversed(xx)))
            q_features.append(float(sv.expectation_value(op).real))

    q_arr = np.array(q_features)
    c_norm = float(np.linalg.norm(features)) if len(features) > 0 else 1.0
    q_norm = float(np.linalg.norm(q_arr))
    enhancement = q_norm / max(c_norm, 1e-15)

    # p_h1 proxy: fraction of phases with |theta| > pi/2
    p_h1 = float(np.mean(np.abs(theta_out) > np.pi / 2))

    return QuantumEVSResult(
        classical_features=features,
        quantum_features=q_arr,
        r_global=r_global,
        p_h1_proxy=p_h1,
        enhancement_factor=enhancement,
    )
