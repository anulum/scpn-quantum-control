# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Execution Integration Tests
"""Integration tests for Phase-QNode statevector and density execution."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDensityCircuit,
    PhaseQNodeDensityExecutionResult,
    PhaseQNodeNoiseChannel,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    execute_phase_qnode_density_matrix,
    registered_phase_qnode_gates,
    registered_phase_qnode_observables,
)


def test_phase_qnode_registered_gate_family_executes_with_pauli_observables() -> None:
    params = np.linspace(0.11, 0.91, 10)
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("x", (1,)),
            ("y", (0,)),
            ("z", (1,)),
            ("s", (0,)),
            ("t", (1,)),
            ("sx", (0,)),
            ("rx", (0,), 0),
            ("ry", (1,), 1),
            ("rz", (0,), 2),
            ("phase", (1,), 3),
            ("cnot", (0, 1)),
            ("cz", (1, 0)),
            ("cy", (0, 1)),
            ("swap", (0, 1)),
            ("crx", (0, 1), 4),
            ("cry", (1, 0), 5),
            ("crz", (0, 1), 6),
            ("rxx", (0, 1), 7),
            ("ryy", (0, 1), 8),
            ("rzz", (0, 1), 9),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "x"),)),
                PauliTerm(-0.25, ((1, "y"),)),
                PauliTerm(0.75, ((0, "z"), (1, "z"))),
            )
        ),
    )

    result = execute_phase_qnode_circuit(circuit, params)

    assert np.isfinite(result.value)
    assert result.state.shape == (4,)
    np.testing.assert_allclose(np.vdot(result.state, result.state).real, 1.0, atol=1e-12)
    assert result.support_report.supported
    assert set(registered_phase_qnode_gates()) >= {
        "rx",
        "ry",
        "rz",
        "phase",
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "sx",
        "cnot",
        "cz",
        "cy",
        "swap",
        "ch",
        "cs",
        "ct",
        "ccnot",
        "ccz",
        "cswap",
        "crx",
        "cry",
        "crz",
        "rxx",
        "ryy",
        "rzz",
    }
    assert set(registered_phase_qnode_observables()) >= {
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "weighted_pauli_sum",
        "pauli_product",
        "pauli_covariance",
        "dense_hermitian",
        "sparse_pauli_hamiltonian",
    }


def test_phase_qnode_covariance_observable_matches_bell_reference() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("h", (0,)), ("cnot", (0, 1))),
        observable=PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((1, "z"),)),
        ),
    )

    result = execute_phase_qnode_circuit(circuit, np.array([], dtype=float))

    np.testing.assert_allclose(result.value, 1.0, atol=1e-12)
    assert result.support_report.observable_kind == "pauli_covariance"


def test_phase_qnode_dense_hermitian_observable_matches_matrix_reference() -> None:
    observable = DenseHermitianObservable(
        np.array(
            [
                [0.7, 0.2 - 0.1j],
                [0.2 + 0.1j, -0.3],
            ],
            dtype=np.complex128,
        )
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=observable,
    )
    params = np.array([0.41], dtype=float)

    result = execute_phase_qnode_circuit(circuit, params)
    state = result.state
    expected = np.vdot(state, observable.matrix @ state).real

    np.testing.assert_allclose(result.value, expected, atol=1e-12)
    assert result.support_report.observable_kind == "dense_hermitian"


def test_phase_qnode_dense_hermitian_observable_fails_closed_on_invalid_matrix() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        DenseHermitianObservable(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))
    with pytest.raises(ValueError, match="dimension"):
        PhaseQNodeCircuit(
            n_qubits=2,
            operations=(("h", (0,)),),
            observable=DenseHermitianObservable(np.eye(2, dtype=np.complex128)),
        )


def test_phase_qnode_controlled_single_qubit_gates_match_references() -> None:
    ch_circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("x", (0,)), ("ch", (0, 1))),
        observable=PauliTerm(1.0, ((1, "x"),)),
    )
    cs_circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("x", (0,)), ("h", (1,)), ("cs", (0, 1))),
        observable=PauliTerm(1.0, ((1, "y"),)),
    )
    ct_circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("x", (0,)), ("h", (1,)), ("ct", (0, 1))),
        observable=PauliTerm(1.0, ((1, "x"),)),
    )

    np.testing.assert_allclose(
        execute_phase_qnode_circuit(ch_circuit, np.array([], dtype=float)).value,
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        execute_phase_qnode_circuit(cs_circuit, np.array([], dtype=float)).value,
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        execute_phase_qnode_circuit(ct_circuit, np.array([], dtype=float)).value,
        np.sqrt(0.5),
        atol=1e-12,
    )


def test_phase_qnode_density_matrix_unitary_route_matches_statevector() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rz", (1,), 1)),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.19], dtype=float)

    pure = execute_phase_qnode_circuit(circuit, params)
    mixed = execute_phase_qnode_density_matrix(circuit, params)

    assert isinstance(mixed, PhaseQNodeDensityExecutionResult)
    np.testing.assert_allclose(mixed.value, pure.value, atol=1e-12)
    np.testing.assert_allclose(mixed.density_matrix, np.outer(pure.state, pure.state.conj()))
    np.testing.assert_allclose(mixed.trace, 1.0, atol=1e-12)
    np.testing.assert_allclose(mixed.purity, 1.0, atol=1e-12)
    assert mixed.support_report.supported
    assert "density-matrix" in mixed.claim_boundary


def test_phase_qnode_density_matrix_noise_channels_match_references() -> None:
    bit_flip = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(PhaseQNodeNoiseChannel("bit_flip", (0,), 1.0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    phase_flip = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(("h", (0,)), ("phase_flip", (0,), 0.5)),
        observable=PauliTerm(1.0, ((0, "x"),)),
    )
    amplitude_damping = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(("x", (0,)), ("amplitude_damping", (0,), 0.25)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    depolarizing = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(("depolarizing", (0,), 0.75),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    bit_flip_result = execute_phase_qnode_density_matrix(bit_flip, np.array([], dtype=float))
    phase_flip_result = execute_phase_qnode_density_matrix(phase_flip, np.array([], dtype=float))
    damping_result = execute_phase_qnode_density_matrix(
        amplitude_damping,
        np.array([], dtype=float),
    )
    depolarizing_result = execute_phase_qnode_density_matrix(
        depolarizing,
        np.array([], dtype=float),
    )

    np.testing.assert_allclose(bit_flip_result.value, -1.0, atol=1e-12)
    np.testing.assert_allclose(phase_flip_result.value, 0.0, atol=1e-12)
    np.testing.assert_allclose(damping_result.value, -0.5, atol=1e-12)
    np.testing.assert_allclose(depolarizing_result.value, 0.0, atol=1e-12)
    for result in (bit_flip_result, phase_flip_result, damping_result, depolarizing_result):
        np.testing.assert_allclose(result.trace, 1.0, atol=1e-12)
        assert result.purity <= 1.0 + 1e-12


def test_phase_qnode_density_matrix_covariance_and_dense_observables() -> None:
    bell_with_noise = PhaseQNodeDensityCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("cnot", (0, 1)),
            PhaseQNodeNoiseChannel("phase_flip", (0,), 0.5),
        ),
        observable=PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((1, "z"),)),
        ),
    )
    dense = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(("amplitude_damping", (0,), 1.0),),
        observable=DenseHermitianObservable(
            np.array([[0.4, 0.1], [0.1, -0.8]], dtype=np.complex128)
        ),
    )

    covariance = execute_phase_qnode_density_matrix(bell_with_noise, np.array([], dtype=float))
    dense_result = execute_phase_qnode_density_matrix(dense, np.array([], dtype=float))

    np.testing.assert_allclose(covariance.value, 1.0, atol=1e-12)
    np.testing.assert_allclose(dense_result.value, 0.4, atol=1e-12)
