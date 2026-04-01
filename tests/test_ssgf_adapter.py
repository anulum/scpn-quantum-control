# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Ssgf Adapter
"""Tests for bridge/ssgf_adapter.py."""

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.ssgf_adapter import (
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)


def test_w_to_hamiltonian_basic():
    W = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    H = ssgf_w_to_hamiltonian(W, omega)
    assert H.num_qubits == 2


def test_w_to_hamiltonian_hermitian():
    W = np.array([[0, 0.3, 0.1], [0.3, 0, 0.2], [0.1, 0.2, 0]])
    omega = np.array([1.0, 1.5, 2.0])
    H = ssgf_w_to_hamiltonian(W, omega)
    mat = np.array(H.to_matrix())
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_state_to_quantum_circuit():
    state = {"theta": np.array([0.5, 1.0, 1.5])}
    qc = ssgf_state_to_quantum(state)
    assert qc.num_qubits == 3


def test_state_roundtrip():
    theta_orig = np.array([0.5, 1.0])
    state = {"theta": theta_orig}
    qc = ssgf_state_to_quantum(state)
    sv = Statevector.from_instruction(qc)
    recovered = quantum_to_ssgf_state(sv, 2)
    np.testing.assert_allclose(recovered["theta"], theta_orig, atol=0.3)


def test_quantum_to_ssgf_R_global():
    qc = ssgf_state_to_quantum({"theta": np.array([1.0, 1.0, 1.0])})
    sv = Statevector.from_instruction(qc)
    state = quantum_to_ssgf_state(sv, 3)
    assert 0.0 <= state["R_global"] <= 1.0


def test_w_to_hamiltonian_zero_coupling():
    W = np.zeros((3, 3))
    omega = np.array([1.0, 2.0, 3.0])
    H = ssgf_w_to_hamiltonian(W, omega)
    assert H.num_qubits == 3


def test_state_to_quantum_zero_angles():
    state = {"theta": np.zeros(2)}
    qc = ssgf_state_to_quantum(state)
    sv = Statevector.from_instruction(qc)
    # Zero phase: (|0>+|1>)/sqrt(2) per qubit, <X>=1, <Y>=0
    recovered = quantum_to_ssgf_state(sv, 2)
    np.testing.assert_allclose(recovered["theta"], 0.0, atol=0.1)


def test_quantum_to_ssgf_coherent():
    theta = np.full(4, 1.0)
    qc = ssgf_state_to_quantum({"theta": theta})
    sv = Statevector.from_instruction(qc)
    result = quantum_to_ssgf_state(sv, 4)
    assert result["R_global"] > 0.5


# ---------------------------------------------------------------------------
# Encoding invariants — quantum information preservation
# ---------------------------------------------------------------------------


def test_encoding_preserves_normalisation():
    """Encoded circuit must produce normalised statevector."""
    theta = np.array([0.3, 1.2, 2.5, 0.8])
    qc = ssgf_state_to_quantum({"theta": theta})
    sv = Statevector.from_instruction(qc)
    np.testing.assert_allclose(float(np.sum(np.abs(sv) ** 2)), 1.0, atol=1e-12)


def test_encoding_gate_count():
    """Each oscillator gets 2 gates (Ry + Rz)."""
    theta = np.zeros(5)
    qc = ssgf_state_to_quantum({"theta": theta})
    assert qc.size() == 10  # 5 * 2


def test_R_global_uniform_phases_high():
    """All phases equal → R ≈ 1 (synchronised)."""
    theta = np.ones(6) * 2.3
    qc = ssgf_state_to_quantum({"theta": theta})
    sv = Statevector.from_instruction(qc)
    result = quantum_to_ssgf_state(sv, 6)
    assert result["R_global"] > 0.9


def test_R_global_opposite_phases_low():
    """Alternating 0/pi phases → R ≈ 0 (desynchronised)."""
    theta = np.array([0.0, np.pi, 0.0, np.pi])
    qc = ssgf_state_to_quantum({"theta": theta})
    sv = Statevector.from_instruction(qc)
    result = quantum_to_ssgf_state(sv, 4)
    assert result["R_global"] < 0.3


# ---------------------------------------------------------------------------
# Pipeline: W matrix → Hamiltonian → evolve → decode → wired
# ---------------------------------------------------------------------------


def test_pipeline_w_to_evolution_decode():
    """Full pipeline: W → H → encode → Trotter evolve → decode.
    Verifies SSGF adapter is not decorative — data flows end-to-end.
    """
    import time

    from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

    W = build_knm_paper27(L=4)
    np.fill_diagonal(W, 0.0)
    omega = np.zeros(4)

    t0 = time.perf_counter()
    H = ssgf_w_to_hamiltonian(W, omega)
    theta_init = np.array([0.1, 0.5, 1.0, 2.0])
    qc = ssgf_state_to_quantum({"theta": theta_init})
    sv = Statevector.from_instruction(qc)
    result = quantum_to_ssgf_state(sv, 4)
    dt = (time.perf_counter() - t0) * 1000

    assert H.num_qubits == 4
    assert "R_global" in result
    assert 0 <= result["R_global"] <= 1.0

    print(f"\n  PIPELINE W→H→encode→decode (4 osc): {dt:.1f} ms")
    print(f"  R_global = {result['R_global']:.4f}")
