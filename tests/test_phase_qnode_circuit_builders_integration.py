# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Builder Integration Tests
"""Integration tests for Phase-QNode circuit and template builders."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control import phase
from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeDepthProfile,
    PhaseQNodeRegisteredCircuitSpec,
    PhaseQNodeSupportError,
    PhaseQNodeTemplateSpec,
    SparsePauliHamiltonian,
    build_phase_qnode_template,
    build_registered_phase_qnode_circuit,
    build_sparse_ising_chain_hamiltonian,
    decompose_phase_qnode_controlled_gate,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_depth_profile,
    registered_phase_qnode_decompositions,
    registered_phase_qnode_templates,
)


def test_phase_qnode_sparse_ising_chain_builder_matches_manual_hamiltonian() -> None:
    x_field = np.linspace(0.05, 0.15, 6)
    z_field = np.linspace(0.2, 0.7, 6)
    zz_coupling = np.linspace(0.4, 0.9, 6)
    observable = build_sparse_ising_chain_hamiltonian(
        6,
        x_field=x_field,
        z_field=z_field,
        zz_coupling=zz_coupling,
        periodic=True,
    )
    manual = SparsePauliHamiltonian(
        (
            *(PauliTerm(float(weight), ((index, "x"),)) for index, weight in enumerate(x_field)),
            *(PauliTerm(float(weight), ((index, "z"),)) for index, weight in enumerate(z_field)),
            *(
                PauliTerm(float(weight), ((index, "z"), ((index + 1) % 6, "z")))
                for index, weight in enumerate(zz_coupling)
            ),
        )
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=6,
        operations=tuple(("ry", (index,), index) for index in range(6)),
        observable=observable,
    )
    manual_circuit = PhaseQNodeCircuit(
        n_qubits=6,
        operations=tuple(("ry", (index,), index) for index in range(6)),
        observable=manual,
    )
    params = np.linspace(0.11, 0.61, 6, dtype=float)

    result = execute_phase_qnode_circuit(circuit, params)
    manual_result = execute_phase_qnode_circuit(manual_circuit, params)
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

    assert len(observable.terms) == 18
    np.testing.assert_allclose(result.value, manual_result.value, atol=1e-12)
    np.testing.assert_allclose(gradient.gradient, finite_difference, atol=1e-6)
    assert result.support_report.observable_kind == "sparse_pauli_hamiltonian"


def test_phase_qnode_sparse_ising_chain_builder_validates_coefficients() -> None:
    open_chain = build_sparse_ising_chain_hamiltonian(4, zz_coupling=0.25)

    assert len(open_chain.terms) == 3
    assert phase.build_sparse_ising_chain_hamiltonian is build_sparse_ising_chain_hamiltonian
    with pytest.raises(ValueError, match="at least two qubits"):
        build_sparse_ising_chain_hamiltonian(1)
    with pytest.raises(ValueError, match="periodic must be a boolean"):
        build_sparse_ising_chain_hamiltonian(4, periodic=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"zz_coupling must have shape \(3,\)"):
        build_sparse_ising_chain_hamiltonian(4, zz_coupling=np.ones(4))
    with pytest.raises(ValueError, match="at least one non-zero term"):
        build_sparse_ising_chain_hamiltonian(4, x_field=0.0, z_field=0.0, zz_coupling=0.0)


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


def test_phase_qnode_controlled_decomposition_registry_validates_inputs() -> None:
    assert set(registered_phase_qnode_decompositions()) == {"ccnot", "cswap"}
    with pytest.raises(ValueError, match="do not accept trainable"):
        decompose_phase_qnode_controlled_gate(("ccnot", (0, 1, 2), 0))
    with pytest.raises(ValueError, match="expects 3 qubits"):
        decompose_phase_qnode_controlled_gate(("cswap", (0, 1)))
    with pytest.raises(ValueError, match="no registered"):
        decompose_phase_qnode_controlled_gate(("ch", (0, 1)))
