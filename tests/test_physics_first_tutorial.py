# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Physics-first tutorial tests
"""Behavioural coverage for the physics-first Kuramoto-XY tutorial path."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control import (
    build_kuramoto_problem,
    compile_hamiltonian,
    compile_trotter_circuit,
    measure_order_parameter,
)


def test_physics_first_tutorial_workflow_compiles_and_measures() -> None:
    K_nm = np.array(
        [
            [0.0, 0.7, 0.0, 0.2],
            [0.7, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.4],
            [0.2, 0.0, 0.4, 0.0],
        ],
        dtype=float,
    )
    omega = np.array([0.9, 1.1, 0.8, 1.2], dtype=float)

    problem = build_kuramoto_problem(
        K_nm,
        omega,
        metadata={"domain": "physics-first-tutorial", "source": "inline-example"},
    )
    hamiltonian = compile_hamiltonian(problem)
    circuit = compile_trotter_circuit(problem, time=0.25, trotter_steps=3)

    initial = QuantumCircuit(problem.n_oscillators)
    initial.h(range(problem.n_oscillators))
    state = Statevector.from_instruction(initial)
    R, psi = measure_order_parameter(problem, state)

    assert problem.n_oscillators == 4
    assert problem.to_metadata()["metadata"]["domain"] == "physics-first-tutorial"
    assert hamiltonian.num_qubits == 4
    assert len(hamiltonian) > 0
    assert circuit.num_qubits == 4
    assert circuit.depth() > 0
    assert pytest.approx(1.0) == R
    assert pytest.approx(0.0) == psi
