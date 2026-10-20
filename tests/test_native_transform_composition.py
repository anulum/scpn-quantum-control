# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real native transform composition contracts
"""Exercise framework transform qualification through the public audit owner."""

from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable_transform_algebra import (
    run_native_transform_algebra_audit,
)
from scpn_quantum_control.phase.transform_nesting import (
    assert_gradient_transform_nesting_supported,
    plan_gradient_transform_nesting,
)


@pytest.mark.framework_imports
@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
def test_native_transform_composition_01(framework: str) -> None:
    """Compare actual gradients and Hessians with analytic quartic derivatives.

    Parameters
    ----------
    framework:
        Required CPU framework; absence is a failure of this owning cohort.

    """
    values = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    original = values.copy()

    audit = run_native_transform_algebra_audit(framework, values)
    cases = {case.category: case for case in audit.cases}

    from scpn_quantum_control.differentiable_api import differentiable_api

    report = differentiable_api("transform_algebra_report", adapter=framework, values=values)
    assert report.supported
    assert report.method == f"{framework}_native_transform_algebra"
    assert report.payload == audit.to_dict()
    assert audit.passed
    assert audit.missing_categories == ()
    assert audit.missing_support_rows == ()
    np.testing.assert_allclose(cases["framework_grad"].lhs, 4.0 * values**3, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        np.array(cases["framework_hessian"].lhs).reshape(3, 3),
        np.diag(12.0 * values**2),
        rtol=1e-8,
        atol=1e-10,
    )
    assert all(framework in case.case_id for case in audit.cases)
    assert "framework-native" in audit.claim_boundary
    np.testing.assert_array_equal(values, original)


@pytest.mark.framework_imports
@pytest.mark.parametrize("framework", ["jax", "pytorch", "tensorflow"])
@pytest.mark.parametrize("batch_axis", [0, 1])
def test_native_transform_composition_02(framework: str, batch_axis: int) -> None:
    """Exercise genuine native batching with both admitted batch axes.

    Parameters
    ----------
    framework:
        Required CPU framework selected by the public audit.
    batch_axis:
        Axis containing the independent quartic input samples.

    """
    values = np.array([[0.2, -0.4], [0.5, 0.7], [-0.6, 0.3]], dtype=np.float64)
    inputs = values if batch_axis == 0 else values.T.copy()
    original = inputs.copy()

    audit = run_native_transform_algebra_audit(framework, inputs, batch_axis=batch_axis)
    cases = {case.category: case for case in audit.cases}

    assert audit.passed
    np.testing.assert_allclose(
        np.array(cases["framework_vmap_grad"].lhs).reshape(values.shape),
        4.0 * values**3,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.array(cases["framework_grad_vmap"].lhs).reshape(values.shape),
        4.0 * values**3,
        rtol=1e-8,
        atol=1e-10,
    )
    shifted = 4.0 * (values + 0.3) ** 3
    np.testing.assert_allclose(
        cases["framework_structured_vmap"].lhs,
        np.concatenate([shifted.ravel(), shifted.sum(axis=1)]),
        rtol=1e-8,
        atol=1e-10,
    )
    custom_expected = (np.cos(np.sum(values**4, axis=1))[:, None] * 4.0 * values**3).ravel()
    copies = 2 if framework == "jax" else 1
    np.testing.assert_allclose(
        np.ravel(cases["framework_custom_vmap"].lhs),
        np.tile(custom_expected, copies),
        rtol=1e-8,
        atol=1e-10,
    )
    assert "manual_batching=False" in cases["framework_vmap_grad"].evidence
    assert any(
        "vmap" in row.transform_stack[0] or "vectorized_map" in row.transform_stack[0]
        for row in audit.support_matrix
    )
    np.testing.assert_array_equal(inputs, original)


@pytest.mark.parametrize("adapter", ["jax", "pytorch", "tensorflow"])
def test_native_transform_composition_03(adapter: str) -> None:
    """Refuse undeclared nested quantum bridges before derivative execution.

    Parameters
    ----------
    adapter:
        Quantum adapter whose generic bridge does not admit arbitrary nesting.

    """
    plan = plan_gradient_transform_nesting(("grad", "grad", "grad"), adapter=adapter)

    assert plan.fail_closed
    assert plan.blocked_reasons
    with pytest.raises(ValueError, match="nesting|nested|single-transform"):
        assert_gradient_transform_nesting_supported(plan)


@pytest.mark.parametrize(
    "values",
    [np.array([]), np.array([np.nan]), np.array([np.inf]), np.array([1.0 + 1.0j])],
)
def test_native_transform_rejection_preserves_inputs(
    values: NDArray[np.float64] | NDArray[np.complex128],
) -> None:
    """Reject inadmissible numeric inputs without modifying caller storage.

    Parameters
    ----------
    values:
        Empty, non-finite or complex input outside the real smooth domain.

    """
    original = values.copy()

    with pytest.raises(ValueError):
        run_native_transform_algebra_audit("jax", values)

    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize(
    "values,batch_axis,match",
    [
        ([True], 0, "real numeric"),
        (["0.3"], 0, "real numeric"),
        (0.3, 0, "vector or matrix"),
        (np.zeros((1, 1, 1)), 0, "vector or matrix"),
        (np.zeros(17), 0, "between one and sixteen"),
        (np.empty((2, 0)), 0, "between one and sixteen"),
        ([0.3], 1, "vector values"),
        ([0.3], True, "zero or one"),
        ([0.3], -1, "zero or one"),
        ([0.3], 2, "zero or one"),
    ],
)
def test_native_audit_refuses_invalid_shape_axis_or_dtype(
    values: Any,
    batch_axis: int,
    match: str,
) -> None:
    """Reject invalid runtime inputs before importing any optional dependency.

    Parameters
    ----------
    values:
        Malformed numeric input.
    batch_axis:
        Requested map axis, including invalid selectors.
    match:
        Observable validation diagnostic.

    """
    with pytest.raises(ValueError, match=match):
        run_native_transform_algebra_audit("jax", values, batch_axis=batch_axis)


def test_unknown_native_runtime_refuses_before_optional_import() -> None:
    """Do not silently route an unknown framework to an available backend."""
    with pytest.raises(ValueError, match="framework must be"):
        run_native_transform_algebra_audit("unknown", [0.3])


@pytest.mark.parametrize(
    "framework,package", [("jax", "jax"), ("pytorch", "torch"), ("tensorflow", "tensorflow")]
)
def test_required_native_runtime_import_failure_is_visible(
    framework: str,
    package: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a missing required library without replacing positive native tests.

    Parameters
    ----------
    framework:
        Public runtime selector.
    package:
        Actual import root whose failure is injected.
    monkeypatch:
        Scoped import fault, removed after the boundary check.

    """
    original_import = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        """Refuse the selected framework while preserving ordinary imports."""
        if name == package:
            raise ModuleNotFoundError(f"required framework unavailable: {package}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ModuleNotFoundError, match="required framework unavailable"):
        run_native_transform_algebra_audit(framework, [0.3])
