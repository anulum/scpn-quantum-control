# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Local derivative composition qualification
"""Compare real local transform compositions against independent coupled derivatives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

import scpn_quantum_control.differentiable_transform_algebra as algebra
from scpn_quantum_control.differentiable_canonical_api import grad
from scpn_quantum_control.differentiable_consistency import check_parameter_shift_consistency
from scpn_quantum_control.differentiable_exact_modes import reverse_mode_gradient
from scpn_quantum_control.differentiable_finite_difference import (
    ScalarObjective,
    VectorObjective,
    jacobian,
    jvp,
    vjp,
)
from scpn_quantum_control.differentiable_local_transform_composition import (
    adjoint_inner_product_pair,
    chain_rule_jacobian_pair,
    gradient_linear_difference_pair,
    reduced_mapped_gradient_pair,
)
from scpn_quantum_control.differentiable_local_transform_primitives import (
    JacobianProvider,
    LocalTransformPrimitives,
    ProductProvider,
)
from scpn_quantum_control.differentiable_parameter_contracts import Parameter
from scpn_quantum_control.differentiable_vmap import vmap
from scpn_quantum_control.metamorphic_ad_verification import get_metamorphic_law


@pytest.mark.parametrize("width", [1, 2, 4])
def test_reduced_mapping_matches_coupled_analytic_row_gradients(width: int) -> None:
    """Retain sum reduction, row order and nonseparable terms through real vmap.

    Parameters
    ----------
    width
        Number of interacting coordinates per row.

    """
    values = np.linspace(-0.6, 0.8, 3 * width).reshape(3, width)

    def row_objective(row: NDArray[np.float64]) -> float:
        return float(np.sum(row**3) + 0.25 * np.sum(row) ** 2)

    def reduced(flat: NDArray[np.float64]) -> float:
        mapped = np.asarray(vmap(row_objective)(flat.reshape(values.shape)), dtype=np.float64)
        return float(np.sum(mapped))

    lhs, rhs = reduced_mapped_gradient_pair(reduced, row_objective, values)
    expected = 3.0 * values**2 + 0.5 * np.sum(values, axis=1, keepdims=True)
    np.testing.assert_allclose(lhs, expected, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(rhs, expected, rtol=2e-6, atol=2e-8)
    assert lhs.shape == rhs.shape == values.shape
    assert lhs.dtype == rhs.dtype == np.dtype(np.float64)


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("weights", [(2.0, 0.5), (-0.75, -1.25), (0.0, 1.0)])
def test_weighted_difference_matches_independent_coupled_gradient(
    width: int, weights: tuple[float, float]
) -> None:
    """Keep actual subtraction and real gradients for signed and zero weights.

    Parameters
    ----------
    width
        Number of interacting input coordinates.
    weights
        First multiplier and separately subtracted multiplier.

    """
    values = np.linspace(-0.4, 0.7, width)
    a, b = weights

    def first(x: NDArray[np.float64]) -> float:
        return float(np.sin(np.sum(x)) + np.dot(x, x))

    def second(x: NDArray[np.float64]) -> float:
        return float(np.sum(x) ** 3 / 3.0)

    def combined(x: NDArray[np.float64]) -> float:
        return a * first(x) - b * second(x)

    expected = a * (np.cos(np.sum(values)) + 2.0 * values) - b * np.sum(values) ** 2
    lhs, rhs = gradient_linear_difference_pair(combined, first, second, values, a, b)
    np.testing.assert_allclose(lhs, expected, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(rhs, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_chain_product_matches_rectangular_coupled_analytic_jacobian(width: int) -> None:
    """Exercise nonlinear inner maps, dense coupling and rectangular outer outputs.

    Parameters
    ----------
    width
        Inner input and output dimension; the outer map adds one output.

    """
    values = np.linspace(-0.3, 0.5, width)
    matrix = np.arange(1.0, width * width + 1.0).reshape(width, width) / width**2

    def inner(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return matrix @ x + 0.1 * x**2

    def outer(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate([y**2, np.array([np.sum(y) ** 3])])

    def composed(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return outer(inner(x))

    y = matrix @ values + 0.1 * values**2
    outer_reference = np.vstack([np.diag(2.0 * y), np.full(width, 3.0 * np.sum(y) ** 2)])
    expected = outer_reference @ (matrix + np.diag(0.2 * values))
    lhs, rhs = chain_rule_jacobian_pair(composed, outer, inner, values)
    np.testing.assert_allclose(lhs, expected, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(rhs, expected, rtol=2e-6, atol=2e-8)
    assert lhs.shape == rhs.shape == (width + 1, width)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_adjoint_contractions_match_rectangular_analytic_products(width: int) -> None:
    """Use independent input/output dimensions and real derivative products.

    Parameters
    ----------
    width
        Input dimension, with one extra output coordinate.

    """
    values = np.linspace(-0.3, 0.5, width)
    tangent = np.linspace(0.1, 0.7, width)
    cotangent = np.linspace(-0.4, 0.8, width + 1)

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate([x**2, np.array([np.sum(x) ** 3])])

    reference = np.vstack([np.diag(2.0 * values), np.full(width, 3.0 * np.sum(values) ** 2)])
    expected = cotangent @ reference @ tangent
    lhs, rhs = adjoint_inner_product_pair(objective, values, tangent, cotangent)
    np.testing.assert_allclose(lhs, [expected], rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(rhs, [expected], rtol=2e-6, atol=2e-8)


def test_chain_reads_outer_provider_before_inner_callback_changes_it() -> None:
    """Retain Python callable-before-arguments order while executing real Jacobians."""
    events: list[str] = []
    current: list[JacobianProvider] = []

    def first(objective: VectorObjective, values: ArrayLike) -> NDArray[np.float64]:
        events.append("first")
        return jacobian(objective, values)

    def second(objective: VectorObjective, values: ArrayLike) -> NDArray[np.float64]:
        events.append("second")
        return jacobian(objective, values)

    current.append(first)

    def inner(x: NDArray[np.float64]) -> NDArray[np.float64]:
        current[0] = second
        return x**2

    def outer(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x**3

    def composed(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x**6

    primitives = LocalTransformPrimitives(
        gradient=lambda: grad,
        jacobian=lambda: current[0],
        jvp=lambda: jvp,
        vjp=lambda: vjp,
        numpy=lambda: np,
        scalar=lambda: float,
    )
    values = np.array([0.4, -0.6])
    lhs, rhs = chain_rule_jacobian_pair(composed, outer, inner, values, primitives=primitives)
    assert events == ["first", "first", "second"]
    np.testing.assert_allclose(lhs, np.diag(6.0 * values**5), rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(rhs, lhs, rtol=2e-6, atol=2e-8)


def test_adjoint_readers_preserve_product_and_scalar_evaluation_order() -> None:
    """Read every live numeric supplier at its original expression boundary."""
    events: list[str] = []

    def read_jvp() -> ProductProvider:
        events.append("jvp")
        return jvp

    def read_vjp() -> ProductProvider:
        events.append("vjp")
        return vjp

    def read_numpy() -> object:
        events.append("numpy")
        return np

    def read_scalar() -> type[float]:
        events.append("float")
        return float

    def objective(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x**2

    primitives = LocalTransformPrimitives(
        gradient=lambda: grad,
        jacobian=lambda: jacobian,
        jvp=read_jvp,
        vjp=read_vjp,
        numpy=read_numpy,
        scalar=read_scalar,
    )
    lhs, rhs = adjoint_inner_product_pair(
        objective,
        np.array([0.2, -0.3]),
        np.array([0.5, 0.7]),
        np.array([-0.1, 0.6]),
        primitives=primitives,
    )
    assert events == [
        "jvp",
        "vjp",
        "numpy",
        "float",
        "numpy",
        "numpy",
        "numpy",
        "float",
        "numpy",
        "numpy",
    ]
    np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-10)


def test_public_audit_keeps_original_callbacks_and_reloads_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the original public report with live provider rebinding and real AD.

    Parameters
    ----------
    monkeypatch
        Scoped replacement of the original module's numerical supplier.

    """
    events: list[tuple[str, str, str]] = []

    def late(
        objective: ScalarObjective,
        values: ArrayLike,
        *,
        method: str,
        parameters: Sequence[Parameter] | None = None,
    ) -> NDArray[np.float64]:
        events.append(("late", objective.__module__, objective.__qualname__))
        return grad(objective, values, method=method, parameters=parameters)

    def initial(
        objective: ScalarObjective,
        values: ArrayLike,
        *,
        method: str,
        parameters: Sequence[Parameter] | None = None,
    ) -> NDArray[np.float64]:
        monkeypatch.setitem(vars(algebra), "grad", late)
        events.append(("initial", objective.__module__, objective.__qualname__))
        return grad(objective, values, method=method, parameters=parameters)

    monkeypatch.setitem(vars(algebra), "grad", initial)
    report = algebra.run_transform_algebra_audit()
    assert report.passed
    assert events[0] == ("initial", algebra.__name__, "_grad_vmap_case.<locals>.reduced")
    assert all(provider == "late" for provider, _, _ in events[1:])
    assert all(module == algebra.__name__ for _, module, _ in events)
    names = {name for _, _, name in events}
    assert "_grad_vmap_case.<locals>.row_objective" in names
    assert "_linearity_case.<locals>.combined" in names


def test_composition_propagates_real_objective_failure() -> None:
    """A caller's failing objective cannot be converted into a successful identity."""

    class ObjectiveFailure(RuntimeError):
        """An objective's deliberate domain error."""

    def objective(values: NDArray[np.float64]) -> float:
        raise ObjectiveFailure("outside the objective domain")

    with pytest.raises(ObjectiveFailure, match="outside the objective domain"):
        gradient_linear_difference_pair(objective, objective, objective, np.array([0.2]), 1.0, 1.0)


def test_parameter_shift_law_names_the_actual_consistency_producer() -> None:
    """Exercise both real estimators behind the retained evidence-gated law."""

    def objective(values: NDArray[np.float64]) -> float:
        return float(np.sin(values[0]) + np.cos(values[1]))

    result = check_parameter_shift_consistency(objective, np.array([0.2, -0.4]))
    assert result.passed
    np.testing.assert_allclose(result.candidate.gradient, [np.cos(0.2), -np.sin(-0.4)])
    np.testing.assert_allclose(result.reference.gradient, result.candidate.gradient, atol=1e-8)
    law = get_metamorphic_law("law:fd.agreement_band.parameter_shift")
    assert check_parameter_shift_consistency.__module__ in law.evidence_modules


def test_complex_boundary_names_a_real_reverse_mode_refusal() -> None:
    """Qualify the named lower producer with an actual complex-valued objective."""
    with pytest.raises(ValueError, match="reverse-mode objective must return a scalar"):
        reverse_mode_gradient(lambda values: np.complex128(1.0 + 2.0j), [0.2])
    law = get_metamorphic_law("law:anti_silent.complex_without_wirtinger")
    assert reverse_mode_gradient.__module__ in law.evidence_modules
