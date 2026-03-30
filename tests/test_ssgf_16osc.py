# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Ssgf 16Osc
"""16-oscillator SSGF quantum loop tests with canonical SCPN parameters."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.bridge.ssgf_adapter import (
    SSGFQuantumLoop,
    quantum_to_ssgf_state,
    ssgf_state_to_quantum,
    ssgf_w_to_hamiltonian,
)


class CanonicalNodeSpace:
    """16-oscillator NodeSpace using canonical SCPN K_nm as geometry W."""

    def __init__(self) -> None:
        self.W = build_knm_paper27(L=16).copy()
        np.fill_diagonal(self.W, 0.0)
        self.theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 16)


class CanonicalSSGFEngine:
    """16-oscillator mock SSGFEngine with canonical SCPN parameters."""

    def __init__(self) -> None:
        self.ns = CanonicalNodeSpace()


class TestSSGF16Oscillator:
    def test_w_to_hamiltonian_16q(self):
        """Canonical K_nm compiles to a 16-qubit Hamiltonian."""
        W = build_knm_paper27(L=16)
        omega = OMEGA_N_16.copy()
        H = ssgf_w_to_hamiltonian(W, omega)
        assert H.num_qubits == 16

    def test_phase_encode_16q(self):
        """16 phases encode to a 16-qubit circuit."""
        theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 16)
        qc = ssgf_state_to_quantum({"theta": theta})
        assert qc.num_qubits == 16
        assert qc.size() == 32  # 16 Ry + 16 Rz

    @pytest.mark.slow
    def test_quantum_loop_16osc(self):
        """Full quantum loop on 16 oscillators (statevector, ~30s)."""
        engine = CanonicalSSGFEngine()
        theta_before = engine.ns.theta.copy()
        loop = SSGFQuantumLoop(engine, dt=0.05, trotter_reps=1)
        result = loop.quantum_step()

        assert "theta" in result
        assert "R_global" in result
        assert len(result["theta"]) == 16
        assert 0.0 <= result["R_global"] <= 1.0
        # Theta should change (non-trivial coupling)
        assert not np.allclose(engine.ns.theta, theta_before, atol=1e-10)

    def test_phase_roundtrip_8osc(self):
        """8-oscillator encode→decode roundtrip."""
        theta = np.random.default_rng(42).uniform(-np.pi, np.pi, 8)
        qc = ssgf_state_to_quantum({"theta": theta})
        from qiskit.quantum_info import Statevector

        sv = Statevector.from_instruction(qc)
        recovered = quantum_to_ssgf_state(sv, 8)
        diff = np.angle(np.exp(1j * (recovered["theta"] - theta)))
        np.testing.assert_allclose(diff, 0.0, atol=1e-6)

    def test_canonical_w_properties(self):
        """Canonical K_nm used as W satisfies SSGF constraints."""
        W = build_knm_paper27(L=16)
        np.fill_diagonal(W, 0.0)
        np.testing.assert_allclose(W, W.T, atol=1e-12)  # symmetric
        assert np.all(W >= 0)  # non-negative
        assert np.allclose(np.diag(W), 0.0)  # zero diagonal
