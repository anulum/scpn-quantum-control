# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Ssgf 16Osc
"""16-oscillator SSGF quantum loop tests — elite multi-angle coverage."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.bridge.ssgf_adapter import (
    SSGFQuantumLoop,
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CanonicalNodeSpace:
    def __init__(self, L: int = 16) -> None:
        self.W = build_knm_paper27(L=L).copy()
        np.fill_diagonal(self.W, 0.0)
        self.theta = np.random.default_rng(42).uniform(0, 2 * np.pi, L)


class CanonicalSSGFEngine:
    def __init__(self, L: int = 16) -> None:
        self.ns = CanonicalNodeSpace(L)


# ---------------------------------------------------------------------------
# ssgf_w_to_hamiltonian
# ---------------------------------------------------------------------------


class TestSSGFWToHamiltonian:
    def test_16q_construction(self):
        W = build_knm_paper27(L=16)
        H = ssgf_w_to_hamiltonian(W, OMEGA_N_16)
        assert H.num_qubits == 16

    @pytest.mark.parametrize("L", [2, 4, 8])
    def test_various_sizes(self, L):
        W = build_knm_paper27(L=L)
        omega = OMEGA_N_16[:L]
        H = ssgf_w_to_hamiltonian(W, omega)
        assert H.num_qubits == L

    def test_hermitian(self):
        W = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H = ssgf_w_to_hamiltonian(W, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# ssgf_state_to_quantum
# ---------------------------------------------------------------------------


class TestStateToQuantum:
    def test_16q_circuit(self):
        theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 16)
        qc = ssgf_state_to_quantum({"theta": theta})
        assert qc.num_qubits == 16
        assert qc.size() == 32  # 16 Ry + 16 Rz

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_various_sizes(self, n):
        theta = np.zeros(n)
        qc = ssgf_state_to_quantum({"theta": theta})
        assert qc.num_qubits == n
        assert qc.size() == 2 * n

    def test_normalised_output(self):
        theta = np.array([0.5, 1.0, 2.0, 3.0])
        qc = ssgf_state_to_quantum({"theta": theta})
        sv = Statevector.from_instruction(qc)
        np.testing.assert_allclose(float(np.sum(np.abs(sv) ** 2)), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# quantum_to_ssgf_state
# ---------------------------------------------------------------------------


class TestQuantumToSSGFState:
    def test_8osc_roundtrip(self):
        theta = np.random.default_rng(42).uniform(-np.pi, np.pi, 8)
        qc = ssgf_state_to_quantum({"theta": theta})
        sv = Statevector.from_instruction(qc)
        recovered = quantum_to_ssgf_state(sv, 8)
        diff = np.angle(np.exp(1j * (recovered["theta"] - theta)))
        np.testing.assert_allclose(diff, 0.0, atol=1e-6)

    def test_returns_R_global(self):
        theta = np.zeros(4)
        qc = ssgf_state_to_quantum({"theta": theta})
        sv = Statevector.from_instruction(qc)
        result = quantum_to_ssgf_state(sv, 4)
        assert "R_global" in result
        assert 0.0 <= result["R_global"] <= 1.0

    def test_uniform_phases_high_R(self):
        """All phases equal → R ≈ 1."""
        theta = np.ones(4) * 1.5
        qc = ssgf_state_to_quantum({"theta": theta})
        sv = Statevector.from_instruction(qc)
        result = quantum_to_ssgf_state(sv, 4)
        assert result["R_global"] > 0.95


# ---------------------------------------------------------------------------
# Canonical W properties
# ---------------------------------------------------------------------------


class TestCanonicalWProperties:
    def test_symmetric(self):
        W = build_knm_paper27(L=16)
        np.fill_diagonal(W, 0.0)
        np.testing.assert_allclose(W, W.T, atol=1e-12)

    def test_non_negative(self):
        W = build_knm_paper27(L=16)
        np.fill_diagonal(W, 0.0)
        assert np.all(W >= 0)

    def test_zero_diagonal(self):
        W = build_knm_paper27(L=16)
        np.fill_diagonal(W, 0.0)
        np.testing.assert_allclose(np.diag(W), 0.0)


# ---------------------------------------------------------------------------
# SSGFQuantumLoop
# ---------------------------------------------------------------------------


class TestSSGFQuantumLoop:
    @pytest.mark.slow
    def test_16osc_quantum_step(self):
        engine = CanonicalSSGFEngine(16)
        theta_before = engine.ns.theta.copy()
        loop = SSGFQuantumLoop(engine, dt=0.05, trotter_reps=1)
        result = loop.quantum_step()

        assert "theta" in result
        assert "R_global" in result
        assert len(result["theta"]) == 16
        assert 0.0 <= result["R_global"] <= 1.0
        assert not np.allclose(engine.ns.theta, theta_before, atol=1e-10)

    def test_4osc_quantum_step(self):
        engine = CanonicalSSGFEngine(4)
        loop = SSGFQuantumLoop(engine, dt=0.1, trotter_reps=2)
        result = loop.quantum_step()
        assert len(result["theta"]) == 4
        assert np.all(np.isfinite(result["theta"]))

    def test_writes_back_to_engine(self):
        engine = CanonicalSSGFEngine(4)
        loop = SSGFQuantumLoop(engine, dt=0.1, trotter_reps=1)
        result = loop.quantum_step()
        np.testing.assert_array_equal(engine.ns.theta, result["theta"])
