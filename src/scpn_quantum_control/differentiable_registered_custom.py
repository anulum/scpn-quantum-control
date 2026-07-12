# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable registered custom module
# scpn-quantum-control -- registry-backed custom derivative wrappers
"""Registry-backed custom derivative wrappers for native transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from numpy.typing import ArrayLike, NDArray

from .differentiable_parameter_contracts import Parameter
from .differentiable_result_contracts import JacobianResult, VJPResult
from .program_ad_registry import (
    CustomDerivativeRegistry,
    PrimitiveIdentity,
    custom_derivative_rule_for,
)


def registered_custom_jvp(
    identity: PrimitiveIdentity | str,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> NDArray[Any]:
    """Return a JVP by resolving the primitive's registered custom rule."""
    from . import differentiable as differentiable_facade

    return differentiable_facade.custom_jvp(
        custom_derivative_rule_for(identity, registry=registry),
        values,
        tangent,
        parameters=parameters,
    )


def registered_custom_vjp(
    identity: PrimitiveIdentity | str,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> VJPResult:
    """Return a VJP by resolving the primitive's registered custom rule."""
    from . import differentiable as differentiable_facade

    return differentiable_facade.custom_vjp(
        custom_derivative_rule_for(identity, registry=registry),
        values,
        cotangent,
        parameters=parameters,
    )


def registered_custom_jacobian(
    identity: PrimitiveIdentity | str,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
    registry: CustomDerivativeRegistry | None = None,
) -> JacobianResult:
    """Return a dense Jacobian by resolving the primitive's registered custom rule."""
    from . import differentiable as differentiable_facade

    return differentiable_facade.value_and_custom_jacobian(
        custom_derivative_rule_for(identity, registry=registry),
        values,
        parameters=parameters,
    )


__all__ = [
    "registered_custom_jacobian",
    "registered_custom_jvp",
    "registered_custom_vjp",
]
