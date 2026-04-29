# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto core facade tests
"""Behavioural tests for the public Kuramoto core facade."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control import (
    KuramotoProblem,
    build_kuramoto_problem,
    compile_dense_hamiltonian,
    compile_hamiltonian,
    compile_trotter_circuit,
    measure_order_parameter,
)


def _problem() -> KuramotoProblem:
    K_nm = np.array(
        [
            [0.0, 0.4, 0.1],
            [0.4, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    omega = np.array([0.3, -0.1, 0.5], dtype=np.float64)
    return build_kuramoto_problem(K_nm, omega, metadata={"domain": "unit-test"})


def test_problem_copies_inputs_and_exports_serialisable_metadata() -> None:
    K_nm = np.array([[7.0, 0.25], [0.25, 9.0]], dtype=np.float64)
    omega = np.array([0.1, -0.2], dtype=np.float64)

    problem = build_kuramoto_problem(K_nm, omega, metadata={"source": "arbitrary"})
    K_nm[0, 1] = 99.0
    omega[0] = 99.0

    assert problem.n_oscillators == 2
    assert problem.K_nm[0, 0] == 0.0
    assert problem.K_nm[0, 1] == 0.25
    assert problem.omega[0] == 0.1
    assert problem.to_metadata() == {
        "n_oscillators": 2,
        "metadata": {"source": "arbitrary"},
        "K_nm_shape": [2, 2],
        "omega_shape": [2],
    }
    with pytest.raises(ValueError):
        problem.K_nm[0, 1] = 0.5
    with pytest.raises(TypeError):
        problem.metadata["new"] = "blocked"


def test_problem_rejects_invalid_inputs_and_metadata() -> None:
    with pytest.raises(ValueError, match="K_nm must be a square matrix"):
        build_kuramoto_problem(np.zeros((2, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="omega must have shape"):
        build_kuramoto_problem(np.zeros((3, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="K_nm must be symmetric"):
        build_kuramoto_problem(np.array([[0.0, 1.0], [0.5, 0.0]]), np.zeros(2))
    with pytest.raises(ValueError, match="omega must contain only finite values"):
        build_kuramoto_problem(np.zeros((2, 2)), np.array([0.0, np.nan]))
    with pytest.raises(TypeError, match="metadata must be JSON-serialisable"):
        build_kuramoto_problem(np.zeros((2, 2)), np.zeros(2), metadata={"bad": object()})


def test_facade_compiles_hamiltonians_and_circuit_for_arbitrary_problem() -> None:
    problem = _problem()

    sparse = compile_hamiltonian(problem)
    dense = compile_dense_hamiltonian(problem)
    circuit = compile_trotter_circuit(problem, time=0.2, trotter_steps=2, trotter_order=2)

    assert sparse.num_qubits == 3
    assert dense.shape == (8, 8)
    np.testing.assert_allclose(dense, dense.conj().T)
    assert circuit.num_qubits == 3
    assert circuit.size() > 0


def test_facade_measures_order_parameter_from_statevector() -> None:
    problem = _problem()
    qc = QuantumCircuit(problem.n_oscillators)
    qc.h(range(problem.n_oscillators))
    state = Statevector.from_instruction(qc)

    R, psi = measure_order_parameter(problem, state)

    assert pytest.approx(1.0) == R
    assert psi == pytest.approx(0.0)
