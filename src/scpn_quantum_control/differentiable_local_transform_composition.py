# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Local derivative composition identities
"""Pure AD computations for both sides of local transform identities.

These functions calculate derivative arrays. Objectives remain supplied by the
caller; this layer constructs no report, QNode plan, case or claim of support.
Finite differences retain their diagnostic interpretation.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from .differentiable_canonical_api import grad
from .differentiable_finite_difference import (
    ScalarObjective,
    VectorObjective,
    jacobian,
    jvp,
    vjp,
)
from .differentiable_local_transform_primitives import (
    CompositionNumpy,
    FloatArray,
    LocalTransformPrimitives,
)

_LOCAL_PRIMITIVES = LocalTransformPrimitives(
    gradient=lambda: grad,
    jacobian=lambda: jacobian,
    jvp=lambda: jvp,
    vjp=lambda: vjp,
    numpy=lambda: np,
    scalar=lambda: float,
)


def reduced_mapped_gradient_pair(
    reduced: ScalarObjective,
    row_objective: ScalarObjective,
    values: FloatArray,
    *,
    primitives: LocalTransformPrimitives = _LOCAL_PRIMITIVES,
) -> tuple[FloatArray, FloatArray]:
    """Differentiate a reduced mapping and its individual row objectives.

    Parameters
    ----------
    reduced
        Scalar reduction accepting the flattened row matrix.
    row_objective
        Scalar objective applied independently to each row.
    values
        Real row matrix at which both gradients are evaluated.
    primitives
        Numerical providers, defaulting to the local AD and NumPy operations.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Reduced gradient reshaped to the matrix and stacked row gradients.
        Equality requires the caller's reduction to be the sum of those rows.

    """
    lhs = primitives.gradient()(reduced, values.ravel(), method="finite_difference").reshape(
        values.shape
    )
    rhs = (
        cast(CompositionNumpy, primitives.numpy())
        .vstack(
            [
                primitives.gradient()(row_objective, row, method="finite_difference")
                for row in values
            ]
        )
        .astype(cast(CompositionNumpy, primitives.numpy()).float64)
    )
    return lhs, rhs


def gradient_linear_difference_pair(
    combined: ScalarObjective,
    first: ScalarObjective,
    second: ScalarObjective,
    values: FloatArray,
    first_weight: float,
    subtracted_weight: float,
    *,
    primitives: LocalTransformPrimitives = _LOCAL_PRIMITIVES,
) -> tuple[FloatArray, FloatArray]:
    """Compute a combined gradient and the weighted difference of gradients.

    Parameters
    ----------
    combined
        Scalar objective intended to equal the weighted difference.
    first
        Scalar objective multiplied by ``first_weight``.
    second
        Scalar objective multiplied by ``subtracted_weight`` and subtracted.
    values
        Real parameter vector shared by all three gradient evaluations.
    first_weight
        Multiplier of the first gradient.
    subtracted_weight
        Multiplier of the gradient subtracted from the first term.
    primitives
        Numerical provider readers, with independent reads after each objective.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Combined gradient and separately evaluated weighted difference.

    """
    lhs = primitives.gradient()(combined, values, method="finite_difference")
    rhs = first_weight * primitives.gradient()(first, values, method="finite_difference") - (
        subtracted_weight * primitives.gradient()(second, values, method="finite_difference")
    )
    return lhs, rhs


def chain_rule_jacobian_pair(
    composed: VectorObjective,
    outer: VectorObjective,
    inner: VectorObjective,
    values: FloatArray,
    *,
    primitives: LocalTransformPrimitives = _LOCAL_PRIMITIVES,
) -> tuple[FloatArray, FloatArray]:
    """Compute a composed Jacobian and the ordered chain-rule matrix product.

    Parameters
    ----------
    composed
        Vector objective intended to be ``outer(inner(values))``.
    outer
        Outer vector objective, differentiated at the actual inner value.
    inner
        Inner vector objective, differentiated at ``values``.
    values
        Real parameter vector for both derivative paths.
    primitives
        Numerical provider readers. The outer Jacobian callable is read before
        executing ``inner(values)``, matching ordinary Python call evaluation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Composed Jacobian and outer Jacobian multiplied by inner Jacobian.

    """
    lhs = primitives.jacobian()(composed, values)
    rhs = primitives.jacobian()(outer, inner(values)) @ primitives.jacobian()(inner, values)
    return lhs, rhs


def adjoint_inner_product_pair(
    vector_objective: VectorObjective,
    values: FloatArray,
    tangent: FloatArray,
    cotangent: FloatArray,
    *,
    primitives: LocalTransformPrimitives = _LOCAL_PRIMITIVES,
) -> tuple[FloatArray, FloatArray]:
    """Contract real JVP and VJP results on their respective dual vectors.

    Parameters
    ----------
    vector_objective
        Vector-valued objective supplying both derivative products.
    values
        Real input vector at which both products are evaluated.
    tangent
        Input-space tangent vector for the JVP and final VJP contraction.
    cotangent
        Output-space cotangent for the VJP and final JVP contraction.
    primitives
        Live numerical readers. JVP executes before VJP; each subsequent
        array, scalar, dot product and dtype lookup keeps Python's evaluation order.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Length-one arrays containing ``cotangent · JVP`` and ``VJP · tangent``.

    """
    jvp_value = primitives.jvp()(vector_objective, values, tangent)
    vjp_value = primitives.vjp()(vector_objective, values, cotangent)
    lhs = cast(CompositionNumpy, primitives.numpy()).array(
        [
            primitives.scalar()(
                cast(CompositionNumpy, primitives.numpy()).dot(cotangent, jvp_value)
            )
        ],
        dtype=cast(CompositionNumpy, primitives.numpy()).float64,
    )
    rhs = cast(CompositionNumpy, primitives.numpy()).array(
        [primitives.scalar()(cast(CompositionNumpy, primitives.numpy()).dot(vjp_value, tangent))],
        dtype=cast(CompositionNumpy, primitives.numpy()).float64,
    )
    return lhs, rhs


__all__ = [
    "adjoint_inner_product_pair",
    "chain_rule_jacobian_pair",
    "gradient_linear_difference_pair",
    "reduced_mapped_gradient_pair",
]
