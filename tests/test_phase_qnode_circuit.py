# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Circuit Registry
"""Tests for phase/qnode_circuit.py registered circuit execution."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control import phase
from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeClassicalFisherResult,
    PhaseQNodeDensityCircuit,
    PhaseQNodeDensityExecutionResult,
    PhaseQNodeDepthProfile,
    PhaseQNodeMetricTensorResult,
    PhaseQNodeNoiseChannel,
    PhaseQNodeRegisteredCircuitSpec,
    PhaseQNodeSupportError,
    PhaseQNodeTemplateSpec,
    SparsePauliHamiltonian,
    build_phase_qnode_template,
    build_registered_phase_qnode_circuit,
    decompose_phase_qnode_controlled_gate,
    execute_phase_qnode_circuit,
    execute_phase_qnode_density_matrix,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_computational_basis_fisher_information,
    phase_qnode_computational_basis_fisher_support_report,
    phase_qnode_density_support_report,
    phase_qnode_depth_profile,
    phase_qnode_gradient_support_report,
    phase_qnode_metric_support_report,
    phase_qnode_natural_gradient_metric,
    phase_qnode_quantum_fisher_information,
    phase_qnode_support_report,
    registered_phase_qnode_decompositions,
    registered_phase_qnode_gates,
    registered_phase_qnode_noise_channels,
    registered_phase_qnode_observables,
    registered_phase_qnode_templates,
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


def test_phase_qnode_parameter_shift_matches_finite_difference_for_registered_generators() -> None:
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


def test_phase_qnode_covariance_gradient_uses_product_rule() -> None:
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


def test_phase_qnode_dense_hermitian_gradient_matches_finite_difference() -> None:
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


def test_phase_qnode_dense_hermitian_observable_fails_closed_on_invalid_matrix() -> None:
    with pytest.raises(ValueError, match="Hermitian"):
        DenseHermitianObservable(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128))
    with pytest.raises(ValueError, match="dimension"):
        PhaseQNodeCircuit(
            n_qubits=2,
            operations=(("h", (0,)),),
            observable=DenseHermitianObservable(np.eye(2, dtype=np.complex128)),
        )


def test_phase_qnode_quantum_fisher_information_matches_ry_reference() -> None:
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


def test_phase_qnode_computational_basis_fisher_fails_closed_at_singular_probability() -> None:
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
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("rx", (0,), 0),),
        observable="pauli_z",
    )
    metric = phase_qnode_natural_gradient_metric(circuit)

    np.testing.assert_allclose(metric(np.array([0.4], dtype=float)), [[0.25]], atol=1e-12)


def test_phase_qnode_quantum_fisher_information_fails_closed_for_unsupported_routes() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        phase_qnode_quantum_fisher_information(circuit, np.array([0.2], dtype=float))
    assert "unsupported gates" in exc_info.value.report.failure_reason


def test_phase_qnode_route_support_reports_block_density_noise_for_pure_metrics() -> None:
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=1,
        operations=(
            ("ry", (0,), 0),
            PhaseQNodeNoiseChannel("amplitude_damping", (0,), 0.2),
        ),
        observable="pauli_z",
    )
    params = np.array([0.31], dtype=float)

    gradient_report = phase_qnode_gradient_support_report(circuit, params)
    metric_report = phase_qnode_metric_support_report(circuit, params)
    fisher_report = phase_qnode_computational_basis_fisher_support_report(circuit, params)

    for report in (gradient_report, metric_report, fisher_report):
        assert not report.supported
        assert "PhaseQNodeCircuit" in report.failure_reason
        assert "amplitude_damping" in report.failure_reason
        assert report.differentiable_parameters == (0,)
        assert report.to_dict()["alternatives"]
    with pytest.raises(PhaseQNodeSupportError) as gradient_error:
        parameter_shift_phase_qnode_gradient(circuit, params)
    with pytest.raises(PhaseQNodeSupportError) as metric_error:
        phase_qnode_quantum_fisher_information(circuit, params)
    with pytest.raises(PhaseQNodeSupportError) as fisher_error:
        phase_qnode_computational_basis_fisher_information(circuit, params)
    assert gradient_error.value.report == gradient_report
    assert metric_error.value.report == metric_report
    assert fisher_error.value.report == fisher_report


def test_phase_qnode_unsupported_routes_fail_with_structured_support_report() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("u3", (0,), 0),),
        observable="pauli_z",
    )

    report = phase_qnode_support_report(circuit, np.array([0.2], dtype=float))

    assert not report.supported
    assert report.unsupported_gates == ("u3",)
    assert "u3" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        execute_phase_qnode_circuit(circuit, np.array([0.2], dtype=float))
    assert exc_info.value.report == report


def test_phase_qnode_ghz_chain_template_executes_multi_qubit_parity_reference() -> None:
    template = build_phase_qnode_template(
        "ghz_chain",
        4,
        observable="z_parity",
    )

    result = execute_phase_qnode_circuit(template.circuit(), np.array([], dtype=float))

    assert isinstance(template, PhaseQNodeTemplateSpec)
    assert template.name == "ghz_chain"
    assert template.parameter_count == 0
    assert [operation.gate for operation in template.operations] == [
        "h",
        "cnot",
        "cnot",
        "cnot",
    ]
    assert result.support_report.supported
    np.testing.assert_allclose(result.value, 1.0, atol=1e-12)
    assert "hardware" in template.claim_boundary
    payload = template.to_dict()
    assert payload["parameter_count"] == 0
    assert payload["operations"]


def test_phase_qnode_hardware_efficient_template_gradient_matches_finite_difference() -> None:
    template = build_phase_qnode_template(
        "hardware_efficient_ryrz",
        3,
        n_layers=2,
        entangler="ring",
    )
    params = np.linspace(0.13, 0.97, template.parameter_count, dtype=float)

    gradient = parameter_shift_phase_qnode_gradient(template.circuit(), params)
    finite_difference = np.zeros_like(params)
    eps = 1e-6
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference[index] = (
            execute_phase_qnode_circuit(template.circuit(), plus).value
            - execute_phase_qnode_circuit(template.circuit(), minus).value
        ) / (2.0 * eps)

    assert template.parameter_count == 12
    assert template.entangler == "ring"
    assert gradient.support_report.differentiable_parameters == tuple(range(12))
    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)


def test_phase_qnode_template_registry_validates_boundaries() -> None:
    assert set(registered_phase_qnode_templates()) == {
        "ghz_chain",
        "hardware_efficient_ry",
        "hardware_efficient_ryrz",
    }
    with pytest.raises(ValueError, match="at least two qubits"):
        build_phase_qnode_template("hardware_efficient_ry", 1)
    with pytest.raises(ValueError, match="ring entanglement"):
        build_phase_qnode_template("hardware_efficient_ry", 2, entangler="ring")
    with pytest.raises(ValueError, match="only supports chain"):
        build_phase_qnode_template("ghz_chain", 3, entangler="ring")
    with pytest.raises(ValueError, match="template observable strings"):
        build_phase_qnode_template(
            "hardware_efficient_ry",
            2,
            observable="not_a_template_observable",
        )


def test_phase_qnode_template_exports_are_public() -> None:
    assert phase.build_phase_qnode_template is build_phase_qnode_template
    assert phase.registered_phase_qnode_templates is registered_phase_qnode_templates
    assert phase.PhaseQNodeTemplateSpec is PhaseQNodeTemplateSpec


def test_phase_qnode_registered_depth_builder_profiles_arbitrary_circuit() -> None:
    spec = build_registered_phase_qnode_circuit(
        n_qubits=3,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
            ("rz", (2,), 2),
            ("rzz", (1, 2), 3),
            ("ry", (0,), 4),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "z"),)),
                PauliTerm(-0.25, ((1, "x"), (2, "z"))),
            )
        ),
        max_depth=4,
        max_operations=6,
    )

    assert isinstance(spec, PhaseQNodeRegisteredCircuitSpec)
    assert isinstance(spec.depth_profile, PhaseQNodeDepthProfile)
    assert spec.support_report.supported
    assert spec.depth_profile.operation_layers == (1, 1, 2, 1, 3, 3)
    assert spec.depth_profile.depth == 3
    assert spec.depth_profile.operation_count == 6
    assert spec.depth_profile.parameter_count == 5
    assert spec.depth_profile.differentiable_parameters == (0, 1, 2, 3, 4)
    assert spec.depth_profile.gate_counts == {
        "cnot": 1,
        "rx": 1,
        "ry": 2,
        "rz": 1,
        "rzz": 1,
    }
    assert spec.depth_profile.two_qubit_gate_count == 2
    assert spec.depth_profile.entangling_pairs == ((0, 1), (1, 2))
    assert "hardware" in spec.claim_boundary
    assert spec.to_dict()["depth_profile"]


def test_phase_qnode_registered_depth_builder_executes_and_gradients() -> None:
    spec = build_registered_phase_qnode_circuit(
        n_qubits=3,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
            ("rz", (2,), 2),
            ("rxx", (0, 2), 3),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"), (2, "x"))),
    )
    params = np.array([0.21, -0.33, 0.17, 0.44], dtype=float)

    value = execute_phase_qnode_circuit(spec.circuit, params)
    gradient = parameter_shift_phase_qnode_gradient(spec.circuit, params)
    finite_difference = np.zeros_like(params)
    eps = 1e-6
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += eps
        minus[index] -= eps
        finite_difference[index] = (
            execute_phase_qnode_circuit(spec.circuit, plus).value
            - execute_phase_qnode_circuit(spec.circuit, minus).value
        ) / (2.0 * eps)

    assert np.isfinite(value.value)
    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)
    assert gradient.parameter_shift_evaluations == 8


def test_phase_qnode_depth_profile_matches_template_circuit() -> None:
    template = build_phase_qnode_template(
        "hardware_efficient_ry",
        3,
        n_layers=2,
        entangler="chain",
    )

    profile = phase_qnode_depth_profile(template.circuit())

    assert profile.depth == 6
    assert profile.operation_count == 10
    assert profile.parameter_count == template.parameter_count
    assert profile.two_qubit_gate_count == 4
    assert profile.max_operation_arity == 2


def test_phase_qnode_registered_depth_builder_fails_closed_for_budgets() -> None:
    operations = (
        ("ry", (0,), 0),
        ("cnot", (0, 1)),
        ("rzz", (0, 1), 1),
    )
    with pytest.raises(ValueError, match="max_operations"):
        build_registered_phase_qnode_circuit(
            2,
            operations,
            PauliTerm(1.0, ((0, "z"),)),
            max_operations=2,
        )
    with pytest.raises(ValueError, match="max_depth"):
        build_registered_phase_qnode_circuit(
            2,
            operations,
            PauliTerm(1.0, ((0, "z"),)),
            max_depth=2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        build_registered_phase_qnode_circuit(
            2,
            operations,
            PauliTerm(1.0, ((0, "z"),)),
            max_depth=0,
        )


def test_phase_qnode_registered_depth_builder_fails_closed_for_unsupported_routes() -> None:
    with pytest.raises(PhaseQNodeSupportError) as exc_info:
        build_registered_phase_qnode_circuit(
            1,
            (("u3", (0,), 0),),
            PauliTerm(1.0, ((0, "z"),)),
        )
    assert exc_info.value.report.unsupported_gates == ("u3",)


def test_phase_qnode_registered_depth_exports_are_public() -> None:
    assert phase.build_registered_phase_qnode_circuit is build_registered_phase_qnode_circuit
    assert phase.phase_qnode_depth_profile is phase_qnode_depth_profile
    assert phase.PhaseQNodeDepthProfile is PhaseQNodeDepthProfile
    assert phase.PhaseQNodeRegisteredCircuitSpec is PhaseQNodeRegisteredCircuitSpec


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


def test_phase_qnode_toffoli_and_fredkin_decompositions_are_exact() -> None:
    ccnot_prefix = (
        ("x", (0,)),
        ("x", (1,)),
    )
    ccnot_native = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(*ccnot_prefix, ("ccnot", (0, 1, 2))),
        observable=PauliTerm(1.0, ((2, "z"),)),
    )
    ccnot_decomposed = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            *ccnot_prefix,
            *decompose_phase_qnode_controlled_gate(("ccnot", (0, 1, 2))),
        ),
        observable=PauliTerm(1.0, ((2, "z"),)),
    )

    native_ccnot = execute_phase_qnode_circuit(ccnot_native, np.array([], dtype=float))
    decomposed_ccnot = execute_phase_qnode_circuit(ccnot_decomposed, np.array([], dtype=float))
    np.testing.assert_allclose(native_ccnot.value, -1.0, atol=1e-12)
    np.testing.assert_allclose(native_ccnot.state, decomposed_ccnot.state, atol=1e-12)

    cswap_prefix = (
        ("x", (0,)),
        ("x", (1,)),
    )
    cswap_native = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(*cswap_prefix, ("cswap", (0, 1, 2))),
        observable=PauliTerm(1.0, ((2, "z"),)),
    )
    cswap_decomposed = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            *cswap_prefix,
            *decompose_phase_qnode_controlled_gate(("cswap", (0, 1, 2))),
        ),
        observable=PauliTerm(1.0, ((2, "z"),)),
    )

    native_cswap = execute_phase_qnode_circuit(cswap_native, np.array([], dtype=float))
    decomposed_cswap = execute_phase_qnode_circuit(cswap_decomposed, np.array([], dtype=float))
    np.testing.assert_allclose(native_cswap.value, -1.0, atol=1e-12)
    np.testing.assert_allclose(native_cswap.state, decomposed_cswap.state, atol=1e-12)


def test_phase_qnode_controlled_gate_support_reports_validate_arity() -> None:
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(("ccnot", (0, 1)),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    report = phase_qnode_support_report(circuit, np.array([], dtype=float))

    assert not report.supported
    assert "gate arity mismatches" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError):
        execute_phase_qnode_circuit(circuit, np.array([], dtype=float))


def test_phase_qnode_controlled_decomposition_registry_validates_inputs() -> None:
    assert set(registered_phase_qnode_decompositions()) == {"ccnot", "cswap"}
    with pytest.raises(ValueError, match="do not accept trainable"):
        decompose_phase_qnode_controlled_gate(("ccnot", (0, 1, 2), 0))
    with pytest.raises(ValueError, match="expects 3 qubits"):
        decompose_phase_qnode_controlled_gate(("cswap", (0, 1)))
    with pytest.raises(ValueError, match="no registered"):
        decompose_phase_qnode_controlled_gate(("ch", (0, 1)))


def test_phase_qnode_controlled_gate_exports_are_public() -> None:
    assert phase.decompose_phase_qnode_controlled_gate is decompose_phase_qnode_controlled_gate
    assert phase.registered_phase_qnode_decompositions is registered_phase_qnode_decompositions


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


def test_phase_qnode_density_support_reports_fail_closed() -> None:
    circuit = PhaseQNodeDensityCircuit(
        n_qubits=2,
        operations=(("ry", (0,)), ("bit_flip", (0, 1), 0.2)),
        observable="pauli_z",
    )

    report = phase_qnode_density_support_report(circuit, np.array([], dtype=float))

    assert not report.supported
    assert "missing parameter" in report.failure_reason
    assert "noise channel arity" in report.failure_reason
    with pytest.raises(PhaseQNodeSupportError):
        execute_phase_qnode_density_matrix(circuit, np.array([], dtype=float))
    with pytest.raises(ValueError, match="between 0 and 1"):
        PhaseQNodeNoiseChannel("bit_flip", (0,), 1.01)


def test_phase_qnode_density_exports_are_public() -> None:
    assert set(registered_phase_qnode_noise_channels()) == {
        "amplitude_damping",
        "bit_flip",
        "depolarizing",
        "phase_flip",
    }
    assert phase.PhaseQNodeDensityCircuit is PhaseQNodeDensityCircuit
    assert phase.PhaseQNodeDensityExecutionResult is PhaseQNodeDensityExecutionResult
    assert phase.PhaseQNodeNoiseChannel is PhaseQNodeNoiseChannel
    assert phase.execute_phase_qnode_density_matrix is execute_phase_qnode_density_matrix
    assert phase.phase_qnode_density_support_report is phase_qnode_density_support_report
    assert phase.phase_qnode_gradient_support_report is phase_qnode_gradient_support_report
    assert phase.phase_qnode_metric_support_report is phase_qnode_metric_support_report
    assert (
        phase.phase_qnode_computational_basis_fisher_support_report
        is phase_qnode_computational_basis_fisher_support_report
    )
    assert phase.registered_phase_qnode_noise_channels is registered_phase_qnode_noise_channels
