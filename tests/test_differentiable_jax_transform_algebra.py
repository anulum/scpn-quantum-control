# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real native transform composition contracts
"""Compare real jax custom/structured compositions with analytic derivatives."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.differentiable_jax_transform_algebra import (
    run_jax_transform_algebra_cases,
)

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_native_custom_curvature_and_structured_mixed_partials(batch_axis: int) -> None:
    """Check nonlinear cross terms and shared-bias mixed derivatives independently.

    Parameters
    ----------
    batch_axis:
        Required native batch axis, including the non-leading route.

    """
    matrix = np.array([[0.0, -0.4], [0.7, 0.3]], dtype=np.float64)
    import jax

    previous_x64 = jax.config.x64_enabled
    original = matrix.copy()
    cases = {
        case.category: case
        for case in run_jax_transform_algebra_cases(matrix, batch_axis=batch_axis)
    }
    gradient = 4.0 * matrix[0] ** 3
    hessian = np.diag(12.0 * matrix[0] ** 2)
    phase = np.sum(matrix[0] ** 4)
    expected_custom = np.cos(phase) * hessian - np.sin(phase) * np.outer(gradient, gradient)
    custom = cases["framework_custom_rules"]
    # The final block is the independently executed reverse-over-reverse Hessian.
    np.testing.assert_allclose(
        np.array(custom.lhs[-4:]).reshape(2, 2), expected_custom, atol=1e-10, rtol=1e-8
    )
    mixed = cases["framework_structured"]
    shifted_diagonal = 12.0 * (matrix[0] + 0.3) ** 2
    np.testing.assert_allclose(mixed.lhs[-5:-3], shifted_diagonal, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(mixed.lhs[-3:-1], shifted_diagonal, atol=1e-10, rtol=1e-8)
    assert mixed.lhs[-1] == pytest.approx(np.sum(shifted_diagonal), abs=1e-10)
    assert all(case.status in {"passed", "blocked"} for case in cases.values())
    assert cases["framework_quantum_bridge_boundary"].status == "blocked"
    assert cases["framework_quantum_bridge_boundary"].blocked_reasons
    assert any("version=" in item for item in custom.evidence)
    np.testing.assert_array_equal(matrix, original)
    assert jax.config.x64_enabled == previous_x64


def test_qnode_native_hessian_matches_trigonometric_oracle() -> None:
    """Compare actual public QNode lowering with analytic mixed quantum curvature."""
    import jax

    from scpn_quantum_control.phase import (
        PauliTerm,
        PhaseQNodeCircuit,
        jax_phase_qnode_native_transform_audit,
    )

    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rz", (0,), 1)),
        observable=PauliTerm(1.0, ((0, "x"),)),
    )
    a, b = 0.37, -0.21
    previous_x64 = jax.config.x64_enabled
    try:
        with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
            result = jax_phase_qnode_native_transform_audit(circuit, [a, b])
    finally:
        # The existing QNode API deliberately enables global x64; restore test custody.
        jax.config.update("jax_enable_x64", previous_x64)
    expected_value = np.sin(a) * np.cos(b)
    expected_cross = -np.cos(a) * np.sin(b)
    np.testing.assert_allclose(result.value, expected_value, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(
        result.gradient, [np.cos(a) * np.cos(b), -np.sin(a) * np.sin(b)], atol=1e-10, rtol=1e-8
    )
    np.testing.assert_allclose(
        result.hessian,
        [[-expected_value, expected_cross], [expected_cross, -expected_value]],
        atol=1e-10,
        rtol=1e-8,
    )
    assert result.passed
    assert not result.host_callback


def test_native_audit_refuses_undeclared_quantum_support_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not promote a host bridge if its public planner starts accepting undeclared orders.

    Parameters
    ----------
    monkeypatch:
        Scoped fault returning a genuine first-order plan for a third-order query.

    """
    import scpn_quantum_control.phase.transform_nesting as planner
    from scpn_quantum_control.differentiable_transform_algebra import (
        run_native_transform_algebra_audit,
    )

    first_order = planner.plan_gradient_transform_nesting(("grad",), adapter="jax")
    assert first_order.supported
    monkeypatch.setattr(
        planner, "plan_gradient_transform_nesting", lambda *args, **kwargs: first_order
    )
    with pytest.raises(RuntimeError, match="undeclared third-order"):
        run_native_transform_algebra_audit("jax", [0.2, -0.3])
