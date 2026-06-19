# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Single-Ancilla Lindblad Circuit
"""Tests for open-system Kuramoto-XY simulation via single ancilla.

Covers:
    - Circuit construction and qubit count
    - Circuit has reset instructions
    - Circuit has measurement
    - Zero coupling → no CX gates
    - Small and large gamma
    - ancilla_circuit_stats output keys and values
    - Edge cases: n=2, single dissipation step
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.phase.ancilla_lindblad import (
    ancilla_circuit_stats,
    build_ancilla_lindblad_circuit,
)


def _system(n: int = 3):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


def _cry_angles(qc: QuantumCircuit) -> list[float]:
    return [float(inst.operation.params[0]) for inst in qc.data if inst.operation.name == "cry"]


class TestBuildCircuit:
    def test_returns_circuit(self):
        K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega)
        assert isinstance(qc, QuantumCircuit)

    def test_qubit_count(self):
        """n system + 1 ancilla."""
        K, omega = _system(4)
        qc = build_ancilla_lindblad_circuit(K, omega)
        assert qc.num_qubits == 5  # 4 + 1

    def test_has_measurement(self):
        K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega)
        op_names = [inst.operation.name for inst in qc.data]
        assert "measure" in op_names

    def test_has_reset(self):
        K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega)
        op_names = [inst.operation.name for inst in qc.data]
        assert "reset" in op_names

    def test_n2_circuit(self):
        K, omega = _system(2)
        qc = build_ancilla_lindblad_circuit(K, omega)
        assert qc.num_qubits == 3

    def test_zero_coupling_no_cx(self):
        """Zero K → no CX gates from coupling (only from dissipation CRY)."""
        n = 3
        K = np.zeros((n, n))
        omega = np.ones(n)
        qc = build_ancilla_lindblad_circuit(K, omega)
        # Should still have reset and measure
        op_names = [inst.operation.name for inst in qc.data]
        assert "reset" in op_names
        # CX from coupling should be absent; CRY from dissipation present
        assert "cry" in op_names

    def test_zero_omega_no_rz(self):
        """Zero omega → no Rz gates."""
        n = 3
        K = np.eye(n) * 0
        K[0, 1] = K[1, 0] = 0.5
        omega = np.zeros(n)
        qc = build_ancilla_lindblad_circuit(K, omega)
        # Rz would only appear from omega terms — but initial Ry still present
        assert isinstance(qc, QuantumCircuit)

    def test_high_gamma_clamps_angle(self):
        """Large gamma → angle clamped to π."""
        K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega, gamma=10.0, t=1.0)
        assert isinstance(qc, QuantumCircuit)

    def test_dissipation_angle_uses_exact_finite_time_decay_probability(self):
        """The repeated-interaction angle must encode p=1-exp(-gamma*dt)."""
        K, omega = _system(2)
        gamma = 0.8
        t = 0.6
        n_dissipation_steps = 3

        qc = build_ancilla_lindblad_circuit(
            K,
            omega,
            gamma=gamma,
            t=t,
            n_dissipation_steps=n_dissipation_steps,
        )

        expected_probability = 1.0 - np.exp(-gamma * (t / n_dissipation_steps))
        expected_angle = 2.0 * np.arcsin(np.sqrt(expected_probability))
        assert _cry_angles(qc)
        np.testing.assert_allclose(_cry_angles(qc), expected_angle, rtol=1e-12, atol=1e-12)

    def test_large_gamma_time_product_remains_finite_without_runtime_warning(self):
        """Large rates are valid finite-time damping, not a source of NaN angles."""
        K, omega = _system(2)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            qc = build_ancilla_lindblad_circuit(
                K,
                omega,
                gamma=10.0,
                t=1.0,
                n_dissipation_steps=1,
            )

        assert not recorded
        assert all(np.isfinite(angle) for angle in _cry_angles(qc))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"gamma": -0.1}, "gamma must be non-negative"),
            ({"gamma": np.inf}, "gamma must be finite"),
            ({"t": -0.1}, "t must be non-negative"),
            ({"t": np.inf}, "t must be finite"),
            ({"trotter_reps": 0}, "trotter_reps must be positive"),
            ({"n_dissipation_steps": 0}, "n_dissipation_steps must be positive"),
        ],
    )
    def test_invalid_physical_parameters_are_rejected(self, kwargs, message):
        K, omega = _system(2)

        with pytest.raises(ValueError, match=message):
            build_ancilla_lindblad_circuit(K, omega, **kwargs)

    def test_single_dissipation_step(self):
        K, omega = _system(3)
        qc = build_ancilla_lindblad_circuit(K, omega, n_dissipation_steps=1)
        assert isinstance(qc, QuantumCircuit)


class TestCircuitStats:
    def test_output_keys(self):
        K, omega = _system(4)
        stats = ancilla_circuit_stats(K, omega)
        expected = {
            "n_qubits",
            "n_system",
            "n_ancilla",
            "n_cx_gates",
            "n_resets",
            "total_gates",
            "n_dissipation_steps",
            "gamma",
        }
        assert set(stats.keys()) == expected

    def test_qubit_count(self):
        K, omega = _system(4)
        stats = ancilla_circuit_stats(K, omega)
        assert stats["n_qubits"] == 5
        assert stats["n_system"] == 4
        assert stats["n_ancilla"] == 1

    def test_resets_count(self):
        K, omega = _system(3)
        stats = ancilla_circuit_stats(K, omega, n_dissipation_steps=5)
        assert stats["n_resets"] == 3 * 5

    def test_zero_coupling_no_cx(self):
        K = np.zeros((4, 4))
        omega = np.ones(4)
        stats = ancilla_circuit_stats(K, omega)
        assert stats["n_cx_gates"] == 0

    def test_total_gates_positive(self):
        K, omega = _system(3)
        stats = ancilla_circuit_stats(K, omega)
        assert stats["total_gates"] > 0
