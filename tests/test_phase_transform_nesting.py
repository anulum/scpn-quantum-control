# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Transform Nesting
"""Tests for phase/transform_nesting.py fail-closed transform rules."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_quantum_control.phase import (
    GradientTransformNestingAuditResult,
    assert_gradient_transform_nesting_supported,
    plan_gradient_transform_nesting,
    run_gradient_transform_nesting_audit,
)


def test_transform_nesting_supports_first_order_and_value_grad_routes() -> None:
    grad_plan = plan_gradient_transform_nesting("grad", n_params=2)
    value_grad_plan = plan_gradient_transform_nesting("value_grad", n_params=2)

    assert grad_plan.supported
    assert grad_plan.strategy == "parameter_shift"
    assert grad_plan.differentiation_order == 1
    assert not grad_plan.requires_deterministic_backend
    assert assert_gradient_transform_nesting_supported(grad_plan) is grad_plan

    assert value_grad_plan.supported
    assert value_grad_plan.transforms == ("value_and_grad",)
    assert value_grad_plan.strategy == "parameter_shift"
    assert value_grad_plan.claim_boundary.startswith("bounded first-order")


def test_transform_nesting_supports_deterministic_native_curvature_routes() -> None:
    hessian = plan_gradient_transform_nesting("hessian", n_params=2)
    nested_grad = plan_gradient_transform_nesting(("grad", "grad"), n_params=2)

    assert hessian.supported
    assert hessian.strategy == "native_parameter_shift_hessian"
    assert hessian.differentiation_order == 2
    assert hessian.requires_deterministic_backend
    assert "second-order routes are deterministic local diagnostics" in hessian.warnings

    assert nested_grad.supported
    assert nested_grad.strategy == "native_hessian_via_nested_grad"
    assert nested_grad.support_plan.transform == "hessian"
    assert nested_grad.requires_deterministic_backend


def test_transform_nesting_supports_single_adapter_and_tape_routes() -> None:
    jax_plan = plan_gradient_transform_nesting("value_and_grad", adapter="jax", n_params=2)
    tape_plan = plan_gradient_transform_nesting("tape", n_params=2)

    assert jax_plan.supported
    assert jax_plan.requires_host_boundary
    assert jax_plan.strategy == "jax_host_callback_parameter_shift"
    assert "single-transform bridge support" in jax_plan.claim_boundary

    assert tape_plan.supported
    assert tape_plan.transforms == ("gradient_tape",)
    assert tape_plan.strategy == "record_supported_parameter_shift_tape"
    assert "record/replay" in tape_plan.warnings[0]


def test_transform_nesting_blocks_vectorized_and_unimplemented_algebra() -> None:
    vmap_grad = plan_gradient_transform_nesting(("vmap", "grad"), adapter="jax", n_params=2)
    jvp = plan_gradient_transform_nesting("jvp", n_params=2)
    jacrev = plan_gradient_transform_nesting("jacrev", n_params=2)

    assert vmap_grad.fail_closed
    assert "vmap over quantum-gradient executions is not implemented" in vmap_grad.blocked_reasons
    assert (
        "ML/provider adapters support only declared single-transform bridge surfaces"
        in vmap_grad.blocked_reasons
    )
    assert "manual loop" in vmap_grad.alternatives

    assert jvp.fail_closed
    assert "quantum-gradient jvp/vjp transform execution is not implemented" in jvp.blocked_reasons

    assert jacrev.fail_closed
    assert "jacfwd/jacrev quantum transform algebra is not implemented" in jacrev.blocked_reasons
    with pytest.raises(ValueError, match="jacfwd/jacrev"):
        assert_gradient_transform_nesting_supported(jacrev)


def test_transform_nesting_blocks_unsafe_curvature_and_tape_nesting() -> None:
    finite_shot_hessian = plan_gradient_transform_nesting(
        "hessian",
        backend="qasm_simulator",
        n_params=2,
        shots=400,
    )
    pytorch_nested_grad = plan_gradient_transform_nesting(
        ("grad", "grad"),
        adapter="pytorch",
        n_params=2,
    )
    tape_grad = plan_gradient_transform_nesting(("gradient_tape", "grad"), n_params=2)

    assert finite_shot_hessian.fail_closed
    assert (
        "second-order transform nesting requires deterministic local expectations"
        in finite_shot_hessian.blocked_reasons
    )
    assert (
        "hessian support is limited to deterministic local backends"
        in finite_shot_hessian.blocked_reasons
    )

    assert pytorch_nested_grad.fail_closed
    assert (
        "nested grad-of-grad is supported only on the native local route"
        in pytorch_nested_grad.blocked_reasons
    )
    assert (
        "ML/provider adapters support only declared single-transform bridge surfaces"
        in pytorch_nested_grad.blocked_reasons
    )

    assert tape_grad.fail_closed
    assert (
        "gradient tape records supported evaluations but is not itself nestable"
        in tape_grad.blocked_reasons
    )


def test_transform_nesting_blocks_hardware_and_unknown_transforms() -> None:
    hardware = plan_gradient_transform_nesting(
        "grad",
        backend="hardware",
        n_params=2,
        shots=1024,
    )
    unknown = plan_gradient_transform_nesting("magic_transform", n_params=2)

    assert hardware.fail_closed
    assert hardware.support_plan.requires_hardware_policy
    assert (
        "hardware gradient execution requires explicit hardware policy approval"
        in hardware.blocked_reasons
    )

    assert unknown.fail_closed
    assert "unknown transform in nesting stack" in unknown.blocked_reasons
    assert "grad" in unknown.alternatives


def test_transform_nesting_audit_records_supported_and_blocked_routes() -> None:
    audit = run_gradient_transform_nesting_audit()
    payload = audit.to_dict()
    plans = cast(list[dict[str, object]], payload["plans"])

    assert isinstance(audit, GradientTransformNestingAuditResult)
    assert audit.passed
    assert len(audit.plans) == 13
    assert len(audit.supported_plans) == 6
    assert len(audit.blocked_plans) == 7
    assert audit.failing_plans == ()
    assert payload["passed"] is True
    assert plans[0]["supported"] is True
    assert plans[-1]["supported"] is False
    assert "transform-nesting audit only" in cast(str, payload["claim_boundary"])


def test_transform_nesting_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="at least one transform"):
        plan_gradient_transform_nesting(())
    with pytest.raises(ValueError, match="transform names"):
        plan_gradient_transform_nesting(("grad", " "))
