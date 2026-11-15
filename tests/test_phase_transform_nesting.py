# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Transform Nesting
"""Tests for phase/transform_nesting.py fail-closed transform rules."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from scpn_quantum_control.phase import (
    GradientTransformNestingAuditResult,
    assert_gradient_transform_nesting_supported,
    plan_gradient_transform_nesting,
    run_gradient_transform_nesting_audit,
)


def test_transform_nesting_supports_first_order_and_value_grad_routes() -> None:
    """First-order grad and value_grad are supported through parameter shift."""
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
    """A native Hessian and grad-of-grad are supported as deterministic second-order routes."""
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
    """A single JAX transform and the gradient tape are supported with their strategies."""
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


def test_transform_nesting_supports_local_directional_and_scalar_jacobian_routes() -> None:
    """JVP, VJP and the scalar Jacobians are supported as local diagnostics."""
    jvp = plan_gradient_transform_nesting("jvp", n_params=2)
    vjp = plan_gradient_transform_nesting("vjp", n_params=2)
    jacfwd = plan_gradient_transform_nesting("jacfwd", n_params=2)
    jacrev = plan_gradient_transform_nesting("jacrev", n_params=2)

    assert jvp.supported
    assert jvp.strategy == "native_parameter_shift_jvp"
    assert "directional/Jacobian routes are scalar local QNode diagnostics" in jvp.warnings
    assert assert_gradient_transform_nesting_supported(jvp) is jvp

    assert vjp.supported
    assert vjp.strategy == "native_parameter_shift_vjp"
    assert jacfwd.supported
    assert jacfwd.strategy == "native_parameter_shift_scalar_jacobian"
    assert jacrev.supported
    assert jacrev.strategy == "native_parameter_shift_scalar_jacobian"


def test_transform_nesting_supports_native_manual_vmap_grad() -> None:
    """Native vmap(grad) is supported as a manual batch loop."""
    vmap_grad = plan_gradient_transform_nesting(("vmap", "grad"), n_params=2)

    assert vmap_grad.supported
    assert vmap_grad.strategy == "native_manual_vmap_parameter_shift_grad"
    assert vmap_grad.requires_deterministic_backend
    assert "manual batch loop" in vmap_grad.warnings[0]


def test_transform_nesting_blocks_framework_vectorized_transform_stacks() -> None:
    """vmap(grad) through a framework adapter is blocked."""
    vmap_grad = plan_gradient_transform_nesting(("vmap", "grad"), adapter="jax", n_params=2)

    assert vmap_grad.fail_closed
    assert (
        "vmap over quantum-gradient executions is supported only for native manual vmap(grad)"
        in vmap_grad.blocked_reasons
    )
    assert (
        "ML/provider adapters support only declared single-transform bridge surfaces"
        in vmap_grad.blocked_reasons
    )
    assert "manual loop" in vmap_grad.alternatives


def test_transform_nesting_blocks_unsafe_curvature_and_tape_nesting() -> None:
    """A finite-shot Hessian, an adapter grad-of-grad and grad over a tape are blocked."""
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
    """Hardware without a policy and an unknown transform are blocked."""
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
    """The audit holds eleven supported and six blocked stacks and no failures."""
    audit = run_gradient_transform_nesting_audit()
    payload = audit.to_dict()
    plans = cast(list[dict[str, object]], payload["plans"])

    assert isinstance(audit, GradientTransformNestingAuditResult)
    assert audit.passed
    assert len(audit.plans) == 17
    assert len(audit.supported_plans) == 11
    assert len(audit.blocked_plans) == 6
    assert audit.failing_plans == ()
    assert payload["passed"] is True
    assert plans[0]["supported"] is True
    assert plans[-1]["supported"] is False
    assert "transform-nesting audit only" in cast(str, payload["claim_boundary"])


def test_transform_nesting_rejects_invalid_input() -> None:
    """An empty stack and a blank transform name are refused."""
    with pytest.raises(ValueError, match="at least one transform"):
        plan_gradient_transform_nesting(())
    with pytest.raises(ValueError, match="transform names"):
        plan_gradient_transform_nesting(("grad", " "))


@pytest.mark.parametrize(
    ("transforms", "reason"),
    [
        pytest.param(
            ("vmap", "grad"),
            "native vmap(grad) hardware execution is policy-blocked",
            id="vmap-grad",
        ),
        pytest.param(
            ("grad", "grad"),
            "second-order hardware-gradient execution is policy-blocked",
            id="grad-grad",
        ),
        pytest.param(
            ("hessian",),
            "second-order hardware-gradient execution is policy-blocked",
            id="hessian",
        ),
    ],
)
def test_transform_nesting_blocks_vectorised_and_second_order_stacks_on_hardware(
    transforms: tuple[str, ...], reason: str
) -> None:
    """Hardware execution of vmap(grad) or of a second-order stack is policy-blocked by name."""
    plan = plan_gradient_transform_nesting(
        transforms, backend="ibm_quantum", allow_hardware=True, shots=1024
    )

    assert plan.fail_closed
    assert reason in plan.blocked_reasons


def test_transform_nesting_audit_flags_unexpected_and_regressed_routes() -> None:
    """A supported stack outside the expected set, or an expected stack now blocked, is a failure."""
    audit = run_gradient_transform_nesting_audit()
    supported = next(plan for plan in audit.plans if plan.supported)
    unexpected = replace(supported, adapter="unlisted_adapter")
    withdrawn = replace(supported, supported=False, blocked_reasons=("withdrawn",))

    assert replace(audit, plans=(*audit.plans, unexpected)).failing_plans == (unexpected,)
    assert replace(audit, plans=(withdrawn,)).failing_plans == (withdrawn,)
