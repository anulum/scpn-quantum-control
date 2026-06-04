# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Gradient Support Matrix
"""Tests for phase/gradient_support_matrix.py support decisions."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_quantum_control.phase import (
    GradientSupportCapability,
    GradientSupportMatrixAuditResult,
    assert_gradient_support,
    gradient_support_capability,
    list_gradient_support_capabilities,
    plan_gradient_support,
    run_gradient_support_matrix_audit,
)


def test_gradient_support_matrix_routes_core_local_gradients() -> None:
    plan = plan_gradient_support(
        gate="ry",
        observable="pauli_expectation",
        backend="statevector",
        transform="grad",
        adapter="native",
        n_params=3,
    )

    assert plan.supported
    assert not plan.fail_closed
    assert plan.recommended_method == "parameter_shift"
    assert plan.evaluation_mode == "deterministic_local"
    assert plan.backend_plan.evaluations == 6
    assert not plan.requires_finite_shot_variance
    assert assert_gradient_support(plan) is plan


def test_gradient_support_matrix_routes_finite_shot_and_host_bridge_paths() -> None:
    finite_shot = plan_gradient_support(
        gate="rz",
        observable="kuramoto_xy_energy",
        backend="qasm_simulator",
        transform="grad",
        adapter="native",
        n_params=2,
        shots=400,
    )
    jax_plan = plan_gradient_support(
        gate="ry",
        observable="z",
        backend="statevector",
        transform="value_grad",
        adapter="jax",
        n_params=2,
    )
    qiskit_plan = plan_gradient_support(
        gate="rx",
        observable="hamiltonian",
        backend="exact",
        transform="grad",
        adapter="qiskit",
        n_params=1,
    )

    assert finite_shot.supported
    assert finite_shot.recommended_method == "stochastic_parameter_shift"
    assert finite_shot.requires_finite_shot_variance
    assert "variance" in finite_shot.warnings[0]

    assert jax_plan.supported
    assert jax_plan.observable == "pauli_expectation"
    assert jax_plan.transform == "value_and_grad"
    assert jax_plan.recommended_method == "jax_host_callback_parameter_shift"
    assert jax_plan.evaluation_mode == "host_bridge"

    assert qiskit_plan.supported
    assert qiskit_plan.observable == "sparse_pauli_sum"
    assert qiskit_plan.backend == "statevector"
    assert qiskit_plan.recommended_method == "qiskit_shifted_circuit_parameter_shift"


def test_gradient_support_matrix_fails_closed_for_unsupported_components() -> None:
    unsupported_gate = plan_gradient_support(
        gate="arbitrary_unitary",
        observable="pauli_expectation",
        backend="statevector",
        transform="grad",
        adapter="native",
        n_params=2,
    )
    unsupported_observable = plan_gradient_support(
        gate="ry",
        observable="arbitrary_povm",
        backend="statevector",
        transform="grad",
        adapter="native",
        n_params=2,
    )
    unsupported_transform = plan_gradient_support(
        gate="ry",
        observable="pauli_expectation",
        backend="statevector",
        transform="vmap",
        adapter="jax",
        n_params=2,
    )

    assert unsupported_gate.fail_closed
    assert (
        "gate has no registered parameter-shift generator spectrum"
        in unsupported_gate.blocked_reasons
    )
    assert "ry" in unsupported_gate.alternatives

    assert unsupported_observable.fail_closed
    assert (
        "observable has no registered expectation-gradient contract"
        in unsupported_observable.blocked_reasons
    )

    assert unsupported_transform.fail_closed
    assert (
        "transform is outside the bounded quantum-gradient algebra"
        in unsupported_transform.blocked_reasons
    )
    assert "native" in unsupported_transform.alternatives
    with pytest.raises(ValueError, match="bounded quantum-gradient algebra"):
        assert_gradient_support(unsupported_transform)


def test_gradient_support_matrix_blocks_hardware_and_finite_shot_hessian() -> None:
    hardware = plan_gradient_support(
        gate="ry",
        observable="pauli_expectation",
        backend="hardware",
        transform="grad",
        adapter="native",
        n_params=2,
        shots=1024,
    )
    finite_shot_hessian = plan_gradient_support(
        gate="ry",
        observable="pauli_expectation",
        backend="qasm_simulator",
        transform="hessian",
        adapter="native",
        n_params=2,
        shots=400,
    )

    assert hardware.fail_closed
    assert hardware.requires_hardware_policy
    assert (
        "hardware gradient execution requires explicit hardware policy approval"
        in hardware.blocked_reasons
    )

    assert finite_shot_hessian.fail_closed
    assert (
        "hessian support is limited to deterministic local backends"
        in finite_shot_hessian.blocked_reasons
    )
    assert "hessian route is a local curvature diagnostic" in finite_shot_hessian.warnings


def test_gradient_support_capability_registry_and_payloads_are_explicit() -> None:
    gate = gradient_support_capability("gate", "y_rotation")
    unknown_adapter = gradient_support_capability("adapter", "new_ml_stack")
    backends = list_gradient_support_capabilities("backend")
    all_capabilities = list_gradient_support_capabilities()

    assert isinstance(gate, GradientSupportCapability)
    assert gate.name == "ry"
    assert gate.supported
    assert not unknown_adapter.supported
    assert "adapter has no registered quantum-gradient bridge" in unknown_adapter.blocked_reasons
    assert {backend.name for backend in backends} == {
        "statevector_simulator",
        "finite_shot_simulator",
        "hardware_qpu",
    }
    assert len(all_capabilities) > len(backends)

    payload = gate.to_dict()
    assert payload["category"] == "gate"
    assert payload["name"] == "ry"


def test_gradient_support_matrix_audit_records_expected_supported_and_blocked_routes() -> None:
    audit = run_gradient_support_matrix_audit()
    payload = audit.to_dict()
    plans = cast(list[dict[str, object]], payload["plans"])

    assert isinstance(audit, GradientSupportMatrixAuditResult)
    assert audit.passed
    assert len(audit.plans) == 9
    assert len(audit.supported_plans) == 4
    assert len(audit.blocked_plans) == 5
    assert audit.failing_plans == ()
    assert payload["passed"] is True
    assert plans[0]["supported"] is True
    assert plans[-1]["supported"] is False
    assert "gradient support matrix audit only" in cast(str, payload["claim_boundary"])


def test_gradient_support_matrix_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_params"):
        plan_gradient_support(gate="ry", observable="pauli_expectation", n_params=0)
    with pytest.raises(ValueError, match="shift_terms"):
        plan_gradient_support(
            gate="ry",
            observable="pauli_expectation",
            n_params=1,
            shift_terms=0,
        )
    with pytest.raises(ValueError, match="gate name"):
        plan_gradient_support(gate=" ", observable="pauli_expectation")
