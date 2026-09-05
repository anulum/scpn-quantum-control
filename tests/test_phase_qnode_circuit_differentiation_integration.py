# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Differentiation Integration Tests
"""Integration tests for Phase-QNode gradients and information metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeClassicalFisherResult,
    PhaseQNodeMetricTensorResult,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_computational_basis_fisher_information,
    phase_qnode_computational_basis_fisher_support_report,
    phase_qnode_natural_gradient_metric,
    phase_qnode_quantum_fisher_information,
)


@pytest.mark.parametrize("swapped", [False, True])
def test_fisher_replays_qiskit_counts_with_explicit_measurement_wires(swapped: bool) -> None:
    """Replay real local samples with both classical-bit measurement assignments."""
    parameters = [0.7, 1.2]
    quantum_circuit = QuantumCircuit(2, 2)
    quantum_circuit.ry(parameters[0], 0)
    quantum_circuit.ry(parameters[1], 1)
    quantum_circuit.measure([0, 1], [1, 0] if swapped else [0, 1])
    raw = (
        StatevectorSampler(seed=73)
        .run([quantum_circuit], shots=4096)
        .result()[0]
        .data.c.get_counts()
    )
    original = dict(raw)
    wires = [0, 1] if swapped else [1, 0]  # Qiskit strings display c1, c0.
    circuit = PhaseQNodeCircuit(
        n_qubits=2, operations=(("ry", (0,), 0), ("ry", (1,), 1)), observable="pauli_z"
    )
    mapped = phase_qnode_computational_basis_fisher_information(
        circuit, parameters, observed_counts=raw, observed_count_wires=wires, shot_count=4096
    )
    labels = ["00", "01", "10", "11"] if swapped else ["00", "10", "01", "11"]
    vector = phase_qnode_computational_basis_fisher_information(
        circuit, parameters, observed_counts=[original[label] for label in labels], shot_count=4096
    )
    mapped_record = json.loads(json.dumps(mapped.to_dict()))
    mapping = mapped_record.pop("count_mapping")
    assert mapped_record == json.loads(json.dumps(vector.to_dict()))
    assert mapping == {
        "schema": "phase_qnode.computational_basis_count_mapping.v1",
        "raw_counts": original,
        "bit_wires": wires,
        "basis_order": "qubit_zero_most_significant",
    }
    assert "count_mapping" not in vector.to_dict()
    np.testing.assert_allclose(mapped.classical_fisher_information, np.eye(2), atol=1e-12)
    assert mapped.empirical_probabilities is not None
    np.testing.assert_allclose(mapped.empirical_probabilities, mapped.probabilities, atol=0.025)
    raw.clear()
    wires.reverse()
    mapping["raw_counts"].clear()
    assert mapped.count_mapping is not None
    assert mapped.count_mapping.to_dict()["raw_counts"] == original
    assert mapped.count_mapping.bit_wires == ((0, 1) if swapped else (1, 0))


def test_fisher_replays_three_qubit_measurement_permutation() -> None:
    """Map a cyclic classical-register assignment, not just reversed strings."""
    parameters = [0.7, 1.2, 1.5]
    quantum_circuit = QuantumCircuit(3, 3)
    for wire, angle in enumerate(parameters):
        quantum_circuit.ry(angle, wire)
    quantum_circuit.measure([0, 1, 2], [1, 2, 0])
    raw = (
        StatevectorSampler(seed=73)
        .run([quantum_circuit], shots=8192)
        .result()[0]
        .data.c.get_counts()
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(("ry", (0,), 0), ("ry", (1,), 1), ("ry", (2,), 2)),
        observable="pauli_z",
    )
    result = phase_qnode_computational_basis_fisher_information(
        circuit, parameters, observed_counts=raw, observed_count_wires=[1, 0, 2], shot_count=8192
    )
    # Canonical q0 q1 q2 outcomes -> displayed c2 c1 c0 = q1 q0 q2.
    assert result.count_record == tuple(
        raw[label] for label in ("000", "001", "100", "101", "010", "011", "110", "111")
    )
    assert result.empirical_probabilities is not None
    np.testing.assert_allclose(result.empirical_probabilities, result.probabilities, atol=0.025)
    np.testing.assert_allclose(result.classical_fisher_information, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    ("counts", "wires", "message"),
    [
        ({"00": 1, "01": 2, "10": 3, "11": 4}, None, "require observed_count_wires"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [0, 0], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [0], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [0, 2], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [False, 1], "permutation"),
        ({"00": 1, "01": 2, "10": 3, "11": 4}, [0.0, 1], "permutation"),
        ({"00": 1}, [0, 1], "every computational-basis"),
        ({"0": 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({"0 0": 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({"ab": 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({0: 1, "01": 2, "10": 3, "11": 4}, [0, 1], "full-width binary"),
        ({"00": True, "01": 2, "10": 3, "11": 4}, [0, 1], "positive integers"),
        ({"00": 1.5, "01": 2, "10": 3, "11": 4}, [0, 1], "positive integers"),
        ({"00": 0, "01": 2, "10": 3, "11": 4}, [0, 1], "positive integers"),
        ({"00": -1, "01": 2, "10": 3, "11": 4}, [0, 1], "positive integers"),
    ],
)
def test_fisher_refuses_ambiguous_bitstring_records(
    counts: object, wires: object, message: str
) -> None:
    """Reject incomplete, nonbinary or undeclared measurement evidence at the public API."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2, operations=(("ry", (0,), 0), ("ry", (1,), 1)), observable="pauli_z"
    )
    with pytest.raises(ValueError, match=message):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            [0.7, 1.2],
            observed_counts=cast(Mapping[str, int], counts),
            observed_count_wires=cast(Sequence[int] | None, wires),
        )


@pytest.mark.parametrize("counts", [None, [1, 2, 3, 4]])
def test_fisher_refuses_wire_metadata_without_bitstring_counts(counts: list[int] | None) -> None:
    """Prevent meaningless mapping provenance on vector or expected-count replay."""
    circuit = PhaseQNodeCircuit(n_qubits=2, operations=(("h", (0,)),), observable="pauli_z")
    with pytest.raises(ValueError, match="requires bitstring mapping counts"):
        phase_qnode_computational_basis_fisher_information(
            circuit, [], observed_counts=counts, observed_count_wires=[0, 1]
        )


@pytest.mark.parametrize("count", [2**80, 10**1000])
def test_fisher_mapping_preserves_integer_counts_with_finite_statistics_boundary(
    count: int,
) -> None:
    """Retain arbitrary integer mapping counts but refuse unrepresentable statistics."""
    circuit = PhaseQNodeCircuit(n_qubits=1, operations=(("ry", (0,), 0),), observable="pauli_z")
    if count == 10**1000:
        with pytest.raises(ValueError, match="finite float64"):
            phase_qnode_computational_basis_fisher_information(
                circuit,
                [np.pi / 2],
                observed_counts={"0": count, "1": count},
                observed_count_wires=[0],
            )
        return
    result = phase_qnode_computational_basis_fisher_information(
        circuit, [np.pi / 2], observed_counts={"0": count, "1": count}, observed_count_wires=[0]
    )
    assert result.count_record == (count, count)
    assert result.shot_count == 2 * count
    with pytest.raises(ValueError, match="sum must equal"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            [np.pi / 2],
            observed_counts={"0": count, "1": count},
            observed_count_wires=[0],
            shot_count=1,
        )


def test_phase_qnode_parameter_shift_matches_finite_difference_for_registered_generators() -> None:
    """Compare registered generator shifts with central differences of the executed circuit."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("cnot", (0, 1)),
            ("rzz", (0, 1), 1),
            ("rx", (1,), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"), (1, "x"))),)),
    )
    params = np.array([0.31, -0.27, 0.43], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    finite_difference = np.zeros_like(params)
    eps = 1e-6
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference[index] = (
            execute_phase_qnode_circuit(circuit, plus).value
            - execute_phase_qnode_circuit(circuit, minus).value
        ) / (2.0 * eps)

    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)
    assert gradient.support_report.differentiable_parameters == (0, 1, 2)
    assert gradient.parameter_shift_evaluations == 6
    assert gradient.evaluation_plan is not None
    assert gradient.evaluation_plan.planned_shifted_evaluations == 6


def test_phase_qnode_covariance_gradient_uses_product_rule() -> None:
    """Check the covariance value and product-rule derivative against analytic sine terms."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1))),
        observable=PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((1, "z"),)),
        ),
    )
    params = np.array([0.37], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)

    np.testing.assert_allclose(gradient.value, np.sin(params[0]) ** 2, atol=1e-12)
    np.testing.assert_allclose(gradient.gradient, [np.sin(2.0 * params[0])], atol=1e-12)
    assert gradient.parameter_shift_evaluations == 2


def test_phase_qnode_dense_hermitian_gradient_matches_finite_difference() -> None:
    """Compare dense-observable gradients against perturbed circuit execution."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=DenseHermitianObservable(
            np.array([[0.7, 0.2], [0.2, -0.3]], dtype=np.complex128)
        ),
    )
    params = np.array([0.41], dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(circuit, params)
    eps = 1e-6
    plus = params + eps
    minus = params - eps
    finite_difference = (
        execute_phase_qnode_circuit(circuit, plus).value
        - execute_phase_qnode_circuit(circuit, minus).value
    ) / (2.0 * eps)

    np.testing.assert_allclose(gradient.gradient, [finite_difference], atol=1e-6)


def test_phase_qnode_quantum_fisher_information_matches_ry_reference() -> None:
    """Check unit quantum Fisher information for a single Ry rotation."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    result = phase_qnode_quantum_fisher_information(circuit, np.array([0.31], dtype=float))

    assert isinstance(result, PhaseQNodeMetricTensorResult)
    np.testing.assert_allclose(result.fubini_study_metric, [[0.25]], atol=1e-12)
    np.testing.assert_allclose(result.quantum_fisher_information, [[1.0]], atol=1e-12)
    assert result.support_report.differentiable_parameters == (0,)
    assert result.claim_boundary.startswith("pure-state local Phase-QNode")


def test_phase_qnode_quantum_fisher_information_is_gauge_invariant_psd() -> None:
    """Check metric symmetry, semidefiniteness and the QFI scaling relation."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("cnot", (0, 1)),
            ("ry", (0,), 0),
            ("rz", (1,), 1),
            ("rxx", (0, 1), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"),)),)),
    )

    result = phase_qnode_quantum_fisher_information(
        circuit,
        np.array([0.17, -0.23, 0.41], dtype=float),
    )

    np.testing.assert_allclose(result.fubini_study_metric, result.fubini_study_metric.T)
    np.testing.assert_allclose(result.quantum_fisher_information, 4.0 * result.fubini_study_metric)
    assert np.min(np.linalg.eigvalsh(result.fubini_study_metric)) >= -1e-12
    assert result.derivative_norms.shape == (3,)
    assert np.all(result.derivative_norms > 0.0)


def test_phase_qnode_computational_basis_fisher_matches_ry_reference() -> None:
    """Check Ry basis probabilities and their classical Fisher information."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        np.array([0.31], dtype=float),
    )

    assert isinstance(result, PhaseQNodeClassicalFisherResult)
    np.testing.assert_allclose(
        result.probabilities, [np.cos(0.31 / 2.0) ** 2, np.sin(0.31 / 2.0) ** 2]
    )
    np.testing.assert_allclose(result.classical_fisher_information, [[1.0]], atol=1e-12)
    assert result.measurement == "computational_basis"
    assert "finite-shot" in result.claim_boundary


def test_phase_qnode_computational_basis_fisher_is_bounded_by_qfi() -> None:
    """Verify that the measurement Fisher matrix does not exceed pure-state QFI."""
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("h", (0,)),
            ("cnot", (0, 1)),
            ("ry", (0,), 0),
            ("rz", (1,), 1),
            ("rxx", (0, 1), 2),
        ),
        observable=SparsePauliHamiltonian((PauliTerm(1.0, ((0, "z"),)),)),
    )
    params = np.array([0.17, -0.23, 0.41], dtype=float)

    classical = phase_qnode_computational_basis_fisher_information(circuit, params)
    quantum = phase_qnode_quantum_fisher_information(circuit, params)

    np.testing.assert_allclose(
        classical.classical_fisher_information,
        classical.classical_fisher_information.T,
        atol=1e-12,
    )
    gap = quantum.quantum_fisher_information - classical.classical_fisher_information
    assert np.min(np.linalg.eigvalsh(gap)) >= -1e-10


def test_phase_qnode_computational_basis_fisher_reports_finite_shot_uncertainty() -> None:
    """Check expected-count uncertainty and confidence-radius serialisation."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        np.array([0.31], dtype=float),
        shot_count=4096,
        confidence_z=2.0,
    )

    assert result.shot_count == 4096
    assert result.count_record is None
    assert result.sampling_model == "multinomial_delta_method_expected_counts"
    finite_shot = result.finite_shot_classical_fisher_information
    assert finite_shot is not None
    np.testing.assert_allclose(
        finite_shot,
        result.classical_fisher_information,
        atol=1e-12,
    )
    standard_error = result.fisher_standard_error
    confidence_radius = result.fisher_confidence_radius
    assert standard_error is not None
    assert confidence_radius is not None
    assert standard_error.shape == (1, 1)
    assert standard_error[0, 0] > 0.0
    np.testing.assert_allclose(
        confidence_radius,
        2.0 * standard_error,
        atol=1e-12,
    )
    payload = result.to_dict()
    assert payload["shot_count"] == 4096
    assert payload["count_record"] is None
    assert payload["sampling_model"] == "multinomial_delta_method_expected_counts"
    assert payload["fisher_standard_error"] == standard_error.tolist()
    assert "finite-shot" in result.claim_boundary


def test_phase_qnode_computational_basis_fisher_replays_raw_count_record() -> None:
    """Replay positive raw counts without replacing the analytic reference."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )
    params = np.array([0.31], dtype=float)
    counts = np.array([3900, 196], dtype=np.int64)

    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        params,
        observed_counts=counts,
        confidence_z=1.5,
    )

    empirical_probabilities = counts / counts.sum()
    exact = phase_qnode_computational_basis_fisher_information(circuit, params)
    expected = (
        exact.probability_derivatives
        @ (exact.probability_derivatives / empirical_probabilities[np.newaxis, :]).T
    )

    assert result.shot_count == int(counts.sum())
    assert result.count_record == (3900, 196)
    assert result.sampling_model == "multinomial_delta_method_raw_count_replay"
    replay_probabilities = result.empirical_probabilities
    finite_shot = result.finite_shot_classical_fisher_information
    standard_error = result.fisher_standard_error
    confidence_radius = result.fisher_confidence_radius
    assert replay_probabilities is not None
    assert finite_shot is not None
    assert standard_error is not None
    assert confidence_radius is not None
    np.testing.assert_allclose(replay_probabilities, empirical_probabilities)
    np.testing.assert_allclose(finite_shot, expected)
    np.testing.assert_allclose(
        confidence_radius,
        1.5 * standard_error,
        atol=1e-12,
    )


@pytest.mark.parametrize("count,dtype", [(2**62, "int64"), (2**63 + 5, "uint64")])
@pytest.mark.parametrize("declare_shots", [False, True])
def test_fisher_replay_preserves_large_integer_counts(
    count: int, dtype: str, declare_shots: bool
) -> None:
    """Preserve raw counts and totals beyond signed NumPy accumulation limits."""
    circuit = PhaseQNodeCircuit(1, (("ry", (0,), 0),), "pauli_z")
    counts = np.array([count, count], dtype=dtype)
    result = phase_qnode_computational_basis_fisher_information(
        circuit,
        [np.pi / 2],
        observed_counts=counts,
        shot_count=2 * count if declare_shots else None,
    )
    assert result.count_record == (count, count)
    assert result.shot_count == 2 * count
    assert result.empirical_probabilities is not None
    assert result.finite_shot_classical_fisher_information is not None
    np.testing.assert_allclose(result.empirical_probabilities, [0.5, 0.5])
    np.testing.assert_allclose(result.finite_shot_classical_fisher_information, [[1.0]])
    assert result.fisher_standard_error is not None
    assert np.all(np.isfinite(result.fisher_standard_error))
    assert result.to_dict()["count_record"] == [count, count]


def test_fisher_replay_uses_qubit_zero_as_most_significant_basis_axis() -> None:
    """Bind an asymmetric two-qubit record to the executed basis ordering."""
    circuit = PhaseQNodeCircuit(2, (("ry", (0,), 0), ("ry", (1,), 1)), "pauli_z")
    angles = np.array([0.4, 1.1])
    local = [np.array([np.cos(angle / 2) ** 2, np.sin(angle / 2) ** 2]) for angle in angles]
    result = phase_qnode_computational_basis_fisher_information(
        circuit, angles, observed_counts=np.array([61, 29, 7, 3])
    )
    np.testing.assert_allclose(result.probabilities, np.kron(local[0], local[1]))
    assert result.empirical_probabilities is not None
    np.testing.assert_allclose(result.empirical_probabilities, [0.61, 0.29, 0.07, 0.03])
    assert result.count_record == (61, 29, 7, 3)


def test_fisher_rejects_shot_total_outside_float_representation() -> None:
    """Reject shot totals that cannot enter the floating uncertainty model."""
    circuit = PhaseQNodeCircuit(1, (("ry", (0,), 0),), "pauli_z")
    with pytest.raises(ValueError, match="shot_count"):
        phase_qnode_computational_basis_fisher_information(circuit, [0.4], shot_count=10**1000)


@pytest.mark.parametrize("shots", [True, 1.5, "4", [4]])
def test_fisher_rejects_non_integer_scalar_shot_declarations(shots: object) -> None:
    """Keep strict scalar-integer admission while allowing large Python totals."""
    circuit = PhaseQNodeCircuit(1, (("ry", (0,), 0),), "pauli_z")
    with pytest.raises(ValueError, match="shot_count"):
        phase_qnode_computational_basis_fisher_information(
            circuit, [0.4], shot_count=cast(int, shots)
        )


def test_fisher_accepts_numpy_integer_shot_scalar() -> None:
    """Retain the existing NumPy scalar shot-count interface."""
    circuit = PhaseQNodeCircuit(1, (("ry", (0,), 0),), "pauli_z")
    result = phase_qnode_computational_basis_fisher_information(
        circuit, [0.4], shot_count=cast(int, np.uint64(16))
    )
    assert result.shot_count == 16


def test_phase_qnode_computational_basis_fisher_validates_finite_shot_inputs() -> None:
    """Reject invalid shot declarations, count shapes and singular replay outcomes."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )
    params = np.array([0.31], dtype=float)

    with pytest.raises(ValueError, match="shot_count must be a positive integer"):
        phase_qnode_computational_basis_fisher_information(circuit, params, shot_count=0)
    with pytest.raises(ValueError, match="observed_counts must have shape"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            observed_counts=np.array([1, 2, 3]),
        )
    with pytest.raises(ValueError, match="observed_counts must be integer counts"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            observed_counts=np.array([1.5, 2.5]),
        )
    with pytest.raises(ValueError, match="observed_counts sum must equal shot_count"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            shot_count=12,
            observed_counts=np.array([5, 6]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        phase_qnode_computational_basis_fisher_information(
            circuit,
            params,
            observed_counts=np.array([4096, 0]),
        )


def test_phase_qnode_computational_basis_fisher_fails_closed_at_singular_probability() -> None:
    """Propagate the support refusal at a zero-probability boundary."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable="pauli_z",
    )

    report = phase_qnode_computational_basis_fisher_support_report(
        circuit,
        np.array([0.0], dtype=float),
    )

    assert not report.supported
    assert "zero-probability" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError, match="zero-probability") as exc_info:
        phase_qnode_computational_basis_fisher_information(circuit, np.array([0.0], dtype=float))
    assert exc_info.value.report == report


def test_phase_qnode_natural_gradient_metric_provider_returns_fubini_study_metric() -> None:
    """Evaluate the natural-gradient callback against the single-rotation metric."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("rx", (0,), 0),),
        observable="pauli_z",
    )
    metric = phase_qnode_natural_gradient_metric(circuit)

    np.testing.assert_allclose(metric(np.array([0.4], dtype=float)), [[0.25]], atol=1e-12)


def test_phase_qnode_quantum_fisher_information_fails_closed_for_unsupported_routes() -> None:
    """Reject quantum Fisher evaluation for an unregistered gate."""
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        phase_qnode_quantum_fisher_information(circuit, np.array([0.2], dtype=float))
    assert "unsupported gates" in exc_info.value.report.failure_reason
