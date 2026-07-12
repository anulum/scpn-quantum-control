# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable custom derivatives module
# scpn-quantum-control -- exact custom derivative transforms
"""Exact custom JVP, VJP, and Jacobian transform wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_batch_helpers import _as_batch_parameter_array, _as_batch_vector_array
from .differentiable_parameter_contracts import (
    Parameter,
    _as_parameter_array,
    _as_real_numeric_array,
)
from .differentiable_result_contracts import JacobianResult, JVPResult, VJPResult
from .differentiable_transform_helpers import _as_vector_output, _normalise_parameters
from .program_ad_registry import CustomDerivativeRule, CustomVJPRule


def _normalise_custom_derivative_parameters(
    values: NDArray[np.float64],
    rule: CustomDerivativeRule,
    parameters: Sequence[Parameter] | None,
) -> tuple[Parameter, ...]:
    """Return explicit parameter metadata for a custom derivative primitive."""
    if parameters is not None:
        return _normalise_parameters(values, parameters)
    if rule.parameter_names:
        if len(rule.parameter_names) != values.size:
            raise ValueError(
                "custom derivative parameter_names length must match parameter length"
            )
        trainable = (
            rule.trainable
            if rule.trainable
            else tuple(True for _ in range(len(rule.parameter_names)))
        )
        if len(trainable) != values.size:
            raise ValueError("custom derivative trainable mask length must match parameter length")
        return tuple(
            Parameter(name, trainable=flag)
            for name, flag in zip(rule.parameter_names, trainable, strict=True)
        )
    return _normalise_parameters(values, None)


def custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return an exact custom Jacobian-vector product for a registered primitive."""
    return value_and_custom_jvp(rule, values, tangent, parameters=parameters).jvp


def batch_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return stacked exact custom JVPs for a batch of tangent vectors."""
    return cast(
        NDArray[np.float64],
        np.vstack(
            [
                result.jvp
                for result in batch_value_and_custom_jvp(
                    rule,
                    values,
                    tangents,
                    parameters=parameters,
                )
            ]
        ),
    )


def batch_value_and_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> tuple[JVPResult, ...]:
    """Return one exact custom JVP result per tangent row."""
    parameter_values = _as_parameter_array(values)
    tangent_batch = _as_batch_parameter_array(
        "custom JVP tangents", tangents, parameter_values.size
    )
    return tuple(
        value_and_custom_jvp(
            rule,
            parameter_values,
            tangent,
            parameters=parameters,
        )
        for tangent in tangent_batch
    )


def value_and_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> JVPResult:
    """Evaluate a custom primitive and its exact JVP rule."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom JVP requires a CustomDerivativeRule")
    if rule.jvp_rule is None:
        raise ValueError("custom derivative rule does not define a JVP rule")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
    tangent_values = _as_parameter_array(tangent)
    if tangent_values.shape != parameter_values.shape:
        raise ValueError("custom JVP tangent length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_tangent = tangent_values.copy()
    masked_tangent[~trainable] = 0.0
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    jvp = _as_vector_output(rule.jvp_rule(parameter_values.copy(), masked_tangent.copy()))
    if jvp.shape != value.shape:
        raise ValueError("custom JVP output shape must match primitive value shape")
    return JVPResult(
        value=value,
        jvp=jvp,
        tangent=masked_tangent,
        method=f"custom_jvp:{rule.name}",
        step=0.0,
        evaluations=1,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> VJPResult:
    """Return an exact custom vector-Jacobian product for a registered primitive."""
    return value_and_custom_vjp(rule, values, cotangent, parameters=parameters)


def batch_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return stacked exact custom VJPs for a batch of cotangent vectors."""
    return cast(
        NDArray[np.float64],
        np.vstack(
            [
                result.vjp
                for result in batch_value_and_custom_vjp(
                    rule,
                    values,
                    cotangents,
                    parameters=parameters,
                )
            ]
        ),
    )


def batch_value_and_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> tuple[VJPResult, ...]:
    """Return one exact custom VJP result per cotangent row."""
    parameter_values = _as_parameter_array(values)
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    cotangent_batch = _as_batch_vector_array("custom VJP cotangents", cotangents, value.size)
    return tuple(
        value_and_custom_vjp(
            rule,
            parameter_values,
            cotangent,
            parameters=parameters,
        )
        for cotangent in cotangent_batch
    )


def value_and_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> VJPResult:
    """Evaluate a custom primitive and its exact VJP rule."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom VJP requires a CustomDerivativeRule")
    if rule.vjp_rule is None:
        raise ValueError("custom derivative rule does not define a VJP rule")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    cotangent_values = _as_vector_output(cotangent)
    if cotangent_values.shape != value.shape:
        raise ValueError("custom VJP cotangent shape must match primitive value shape")
    vjp = _as_parameter_array(rule.vjp_rule(parameter_values.copy(), cotangent_values.copy()))
    if vjp.shape != parameter_values.shape:
        raise ValueError("custom VJP output length must match parameter length")
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    masked_vjp = vjp.copy()
    masked_vjp[~trainable] = 0.0
    return VJPResult(
        value=value,
        cotangent=cotangent_values,
        vjp=masked_vjp,
        method=f"custom_vjp:{rule.name}",
        step=0.0,
        evaluations=1,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )


def custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return the exact dense Jacobian implied by a custom derivative rule."""
    return value_and_custom_jacobian(rule, values, parameters=parameters).jacobian


def batch_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return stacked exact custom Jacobians for a batch of parameter rows."""
    return cast(
        NDArray[np.float64],
        np.stack(
            [
                result.jacobian
                for result in batch_value_and_custom_jacobian(
                    rule,
                    values,
                    parameters=parameters,
                )
            ],
            axis=0,
        ),
    )


def batch_value_and_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> tuple[JacobianResult, ...]:
    """Return one exact custom Jacobian result per parameter row."""
    batch = _as_real_numeric_array("custom Jacobian values", values)
    if batch.ndim != 2:
        raise ValueError("custom Jacobian values must be a two-dimensional batch")
    if not np.all(np.isfinite(batch)):
        raise ValueError("custom Jacobian values must contain only finite values")
    return tuple(
        value_and_custom_jacobian(
            rule,
            row,
            parameters=parameters,
        )
        for row in batch
    )


def value_and_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> JacobianResult:
    """Evaluate a custom primitive and materialise its exact dense Jacobian."""
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom Jacobian requires a CustomDerivativeRule")
    if rule.jvp_rule is None and rule.vjp_rule is None:
        raise ValueError("custom derivative rule requires a JVP or VJP rule")
    parameter_values = _as_parameter_array(values)
    parameter_meta = _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
    value = _as_vector_output(rule.value_fn(parameter_values.copy()))
    trainable = np.array([parameter.trainable for parameter in parameter_meta], dtype=bool)
    jacobian_arr = np.zeros((value.size, parameter_values.size), dtype=np.float64)
    evaluations = 1
    if rule.jvp_rule is not None:
        for column, is_trainable in enumerate(trainable):
            if not is_trainable:
                continue
            basis = np.zeros(parameter_values.size, dtype=np.float64)
            basis[column] = 1.0
            jvp = _as_vector_output(rule.jvp_rule(parameter_values.copy(), basis))
            if jvp.shape != value.shape:
                raise ValueError("custom JVP output shape must match primitive value shape")
            jacobian_arr[:, column] = jvp
    else:
        vjp_rule = cast(CustomVJPRule, rule.vjp_rule)
        for row in range(value.size):
            cotangent = np.zeros(value.size, dtype=np.float64)
            cotangent[row] = 1.0
            vjp = _as_parameter_array(vjp_rule(parameter_values.copy(), cotangent))
            if vjp.shape != parameter_values.shape:
                raise ValueError("custom VJP output length must match parameter length")
            vjp[~trainable] = 0.0
            jacobian_arr[row, :] = vjp
    return JacobianResult(
        value=value,
        jacobian=jacobian_arr,
        method=f"custom_jacobian:{rule.name}",
        step=0.0,
        evaluations=evaluations,
        parameter_names=tuple(parameter.name for parameter in parameter_meta),
        trainable=tuple(parameter.trainable for parameter in parameter_meta),
    )
