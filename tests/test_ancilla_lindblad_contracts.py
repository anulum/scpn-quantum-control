# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Ancilla Lindblad contract tests
"""Contract tests for ancilla Lindblad circuit structure, statistics, and damping parameters."""

from __future__ import annotations

import numpy as np
import pytest


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _zero_coupling(n: int = 4):
    """Decoupled system — K=0, eigenstates are product states."""
    K = np.zeros((n, n))
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


class TestAncillaLindblad:
    """Tests for single-ancilla open-system circuit."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_circuit_has_correct_qubit_count(self, n):
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(n)
        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.1,
            trotter_reps=2,
            n_dissipation_steps=2,
        )
        assert qc.num_qubits == n + 1  # system + 1 ancilla

    @pytest.mark.parametrize("n_steps", [1, 2, 3, 5])
    def test_reset_count_scales_with_steps(self, n_steps):
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            t=0.1,
            n_dissipation_steps=n_steps,
        )
        reset_count = sum(1 for i in qc.data if i.operation.name == "reset")
        # Each dissipation step resets the ancilla after interacting with each system qubit
        assert reset_count == 3 * n_steps  # n_system * n_dissipation_steps

    def test_circuit_stats_all_keys(self):
        from scpn_quantum_control.phase.ancilla_lindblad import ancilla_circuit_stats

        _, K, omega = _system(4)
        stats = ancilla_circuit_stats(K, omega)
        assert stats["n_qubits"] == 5
        assert stats["n_system"] == 4
        assert stats["n_ancilla"] == 1
        assert stats["n_resets"] > 0
        assert stats["n_cx_gates"] > 0
        assert stats["total_gates"] > 0

    def test_circuit_has_measurements(self):
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega)
        assert any(i.operation.name == "measure" for i in qc.data)

    def test_stats_consistent_with_circuit(self):
        """Stats should match actual circuit properties."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            ancilla_circuit_stats,
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(3)
        kwargs = {"t": 0.1, "trotter_reps": 3, "gamma": 0.05, "n_dissipation_steps": 2}
        qc = build_ancilla_lindblad_circuit(K, omega, **kwargs)
        stats = ancilla_circuit_stats(K, omega, **kwargs)

        assert stats["n_qubits"] == qc.num_qubits
        actual_resets = sum(1 for i in qc.data if i.operation.name == "reset")
        assert stats["n_resets"] == actual_resets

    @pytest.mark.parametrize("gamma", [0.0, 0.01, 0.05, 0.1, 0.5])
    def test_gamma_range(self, gamma):
        """Circuit should build for all valid gamma values."""
        from scpn_quantum_control.phase.ancilla_lindblad import (
            build_ancilla_lindblad_circuit,
        )

        _, K, omega = _system(2)
        qc = build_ancilla_lindblad_circuit(K, omega, gamma=gamma)
        assert qc.num_qubits == 3
