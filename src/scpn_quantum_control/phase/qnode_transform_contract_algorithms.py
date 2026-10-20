# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Scalar QNode admission algorithms
"""Single original scalar QNode admission and result-construction bodies."""

from __future__ import annotations

from typing import TypeVar

from numpy.typing import ArrayLike

from scpn_quantum_control.phase.qnode_transform_contract_bindings import (
    FloatArray,
    QNodeContractBindings,
)
from scpn_quantum_control.phase.transform_nesting import GradientTransformNestingPlan

_Result = TypeVar("_Result")


def _supported_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    params: FloatArray,
    *,
    value: float | None = None,
    gradient: FloatArray | None = None,
    hessian: FloatArray | None = None,
    hessian_vector_product: FloatArray | None = None,
    jvp: float | None = None,
    vjp: FloatArray | None = None,
    jacobian: FloatArray | None = None,
    tangent: FloatArray | None = None,
    cotangent: FloatArray | None = None,
    parameter_shift_evaluations: int | None = None,
    claim_boundary: str = "supported scalar local phase-QNode transforms only; not hardware execution, not arbitrary program AD, and not full vector-output framework transform algebra",
    _bindings: QNodeContractBindings[_Result],
) -> _Result:
    """Construct the original supported scalar transform record and retain independent arrays."""
    return _bindings.result_class()(
        transform=transform,
        plan=plan,
        params=params.copy(),
        supported=True,
        value=value,
        gradient=None
        if gradient is None
        else gradient.astype(_bindings.numpy().float64, copy=True),
        hessian=None if hessian is None else hessian.astype(_bindings.numpy().float64, copy=True),
        hessian_vector_product=None
        if hessian_vector_product is None
        else hessian_vector_product.astype(_bindings.numpy().float64, copy=True),
        jvp=jvp,
        vjp=None if vjp is None else vjp.astype(_bindings.numpy().float64, copy=True),
        jacobian=None
        if jacobian is None
        else jacobian.astype(_bindings.numpy().float64, copy=True),
        tangent=None if tangent is None else tangent.astype(_bindings.numpy().float64, copy=True),
        cotangent=None
        if cotangent is None
        else cotangent.astype(_bindings.numpy().float64, copy=True),
        parameter_shift_evaluations=plan.support_plan.backend_plan.evaluations
        if parameter_shift_evaluations is None
        else parameter_shift_evaluations,
        failure_reason="",
        claim_boundary=claim_boundary,
    )


def _blocked_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    params: FloatArray,
    *,
    reason: str | None = None,
    _bindings: QNodeContractBindings[_Result],
) -> _Result:
    """Construct the original refused scalar transform record and its original reason."""
    return _bindings.result_class()(
        transform=transform,
        plan=plan,
        params=params.copy(),
        supported=False,
        value=None,
        gradient=None,
        hessian=None,
        hessian_vector_product=None,
        jvp=None,
        vjp=None,
        jacobian=None,
        tangent=None,
        cotangent=None,
        parameter_shift_evaluations=0,
        failure_reason=reason if reason is not None else "; ".join(plan.blocked_reasons),
    )


def _as_parameter_vector(
    name: str,
    values: ArrayLike | None,
    *,
    width: int | None = None,
    _bindings: QNodeContractBindings[_Result],
) -> FloatArray:
    """Admit an independent real finite parameter or directional vector."""
    if values is None:
        raise _bindings.builtins.error()(f"{name} must be provided")
    raw = _bindings.numpy().asarray(values)
    if raw.dtype.kind in {"c", "O", "S", "U"}:
        raise _bindings.builtins.error()(
            f"{name} must be a real-valued finite array; complex/Wirtinger derivatives are outside the supported Phase-QNode transform contract"
        )
    vector = _bindings.numpy().asarray(values, dtype=_bindings.builtins.scalar())
    if vector.ndim != 1:
        raise _bindings.builtins.error()(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise _bindings.builtins.error()(f"{name} must have shape ({width},), got {vector.shape}")
    if not _bindings.numpy().all(_bindings.numpy().isfinite(vector)):
        raise _bindings.builtins.error()(f"{name} must contain only finite values")
    return _bindings.array_cast()(
        _bindings.array_type(), vector.astype(_bindings.numpy().float64, copy=True)
    )


def _hessian_evaluations(n_params: int, shift_terms: int) -> int:
    """Count the original diagonal and mixed single-term Hessian probes."""
    diagonal = 2 * n_params * shift_terms
    mixed_pairs = n_params * (n_params - 1) // 2
    mixed = 4 * mixed_pairs * shift_terms * shift_terms
    return diagonal + mixed


def _as_cotangent(
    values: ArrayLike | float | None, *, _bindings: QNodeContractBindings[_Result]
) -> FloatArray:
    """Admit the original real finite scalar or one-element cotangent."""
    if values is None:
        raise _bindings.builtins.error()("cotangent must be provided")
    raw = _bindings.numpy().asarray(values)
    if raw.dtype.kind in {"c", "O", "S", "U"}:
        raise _bindings.builtins.error()(
            "cotangent must be a real-valued finite array; complex/Wirtinger derivatives are outside the supported Phase-QNode transform contract"
        )
    vector = _bindings.numpy().asarray(values, dtype=_bindings.builtins.scalar())
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1 or vector.shape != (1,):
        raise _bindings.builtins.error()("cotangent must be a scalar or one-element vector")
    if not _bindings.numpy().all(_bindings.numpy().isfinite(vector)):
        raise _bindings.builtins.error()("cotangent must contain only finite values")
    return _bindings.array_cast()(
        _bindings.array_type(), vector.astype(_bindings.numpy().float64, copy=True)
    )
