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
    if rule.parameter_names and len(rule.parameter_names) != values.size:
        raise ValueError("custom derivative parameter_names length must match parameter length")
    if rule.trainable and len(rule.trainable) != values.size:
        raise ValueError("custom derivative trainable mask length must match parameter length")
    defaults = _normalise_parameters(values, None)
    names = rule.parameter_names or tuple(parameter.name for parameter in defaults)
    trainable = rule.trainable or tuple(True for _ in defaults)
    return _normalise_parameters(
        values,
        tuple(
            Parameter(name, trainable=flag) for name, flag in zip(names, trainable, strict=True)
        ),
    )


def custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangent: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return exact custom Jacobian-vector product values.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    tangent : array_like
        Tangent vector aligned with the parameter vector. Frozen parameter
        components contribute zero.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    numpy.ndarray
        Exact derivative; frozen parameter directions contribute zero.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
    return value_and_custom_jvp(rule, values, tangent, parameters=parameters).jvp


def batch_custom_jvp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    tangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return batched exact custom Jacobian-vector product values.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    tangents : array_like
        Matrix of tangent rows aligned with the parameter vector. Frozen
        parameter components contribute zero.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    numpy.ndarray
        Stacked derivatives in input-row order; frozen parameter directions
        contribute zero.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Evaluate batched exact custom Jacobian-vector product results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    tangents : array_like
        Matrix of tangent rows aligned with the parameter vector. Frozen
        parameter components contribute zero.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    tuple[JVPResult, ...]
        One result per input row, in input order, retaining parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Evaluate exact custom Jacobian-vector product results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    tangent : array_like
        Tangent vector aligned with the parameter vector. Frozen parameter
        components contribute zero.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    JVPResult
        Exact value and derivative with parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Evaluate exact custom vector-Jacobian product results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    cotangent : array_like
        Cotangent vector aligned with the primitive output.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    VJPResult
        Exact value and derivative with parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
    return value_and_custom_vjp(rule, values, cotangent, parameters=parameters)


def batch_custom_vjp(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    cotangents: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return batched exact custom vector-Jacobian product values.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    cotangents : array_like
        Matrix of cotangent rows aligned with the primitive output.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    numpy.ndarray
        Stacked derivatives in input-row order; frozen parameter directions
        contribute zero.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Evaluate batched exact custom vector-Jacobian product results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    cotangents : array_like
        Matrix of cotangent rows aligned with the primitive output.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    tuple[VJPResult, ...]
        One result per input row, in input order, retaining parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("custom VJP requires a CustomDerivativeRule")
    if rule.vjp_rule is None:
        raise ValueError("custom derivative rule does not define a VJP rule")
    parameter_values = _as_parameter_array(values)
    _normalise_custom_derivative_parameters(parameter_values, rule, parameters)
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
    """Evaluate exact custom vector-Jacobian product results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    cotangent : array_like
        Cotangent vector aligned with the primitive output.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    VJPResult
        Exact value and derivative with parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Return exact custom dense Jacobian values.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    numpy.ndarray
        Exact derivative; frozen parameter directions contribute zero.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
    return value_and_custom_jacobian(rule, values, parameters=parameters).jacobian


def batch_custom_jacobian(
    rule: CustomDerivativeRule,
    values: ArrayLike,
    *,
    parameters: Sequence[Parameter] | None = None,
) -> NDArray[np.float64]:
    """Return batched exact custom dense Jacobian values.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite real parameter matrix; each row is a separate evaluation point.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    numpy.ndarray
        Stacked derivatives in input-row order; frozen parameter directions
        contribute zero.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Evaluate batched exact custom dense Jacobian results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite real parameter matrix; each row is a separate evaluation point.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    tuple[JacobianResult, ...]
        One result per input row, in input order, retaining parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
    """Evaluate exact custom dense Jacobian results.

    Parameters
    ----------
    rule : CustomDerivativeRule
        Declared real-vector primitive with the required exact derivative rule.
    values : array_like
        Finite one-dimensional real parameter vector.
    parameters : sequence of Parameter, optional
        Explicit caller metadata overrides rule-local names and trainability.
        Otherwise the rule mask applies even with generated parameter names.

    Returns
    -------
    JacobianResult
        Exact value and derivative with parameter names and mask.

    Raises
    ------
    ValueError
        If the required rule is absent, values or derivatives are non-finite
        or mis-shaped, or parameter metadata has invalid length or identity.

    """
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
