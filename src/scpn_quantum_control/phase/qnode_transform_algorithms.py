# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Scalar QNode execution algorithms
"""Single numerical bodies for original scalar QNode transforms and readiness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from numpy.typing import ArrayLike

from scpn_quantum_control.differentiable import Parameter, ParameterShiftRule
from scpn_quantum_control.phase.qnode_transform_bindings import QNodeTransformBindings
from scpn_quantum_control.phase.qnode_transform_contract_bindings import (
    ScalarObjective,
)

_Result = TypeVar("_Result")
_Complex = TypeVar("_Complex")
_Suite = TypeVar("_Suite")


def execute_phase_qnode_transform(
    transform: str | tuple[str, ...],
    objective: ScalarObjective,
    params: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    cotangent: ArrayLike | float | None = None,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    adapter: str = "native",
    shots: int | None = None,
    shift_terms: int = 1,
    allow_hardware: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    _bindings: QNodeTransformBindings[_Result, _Complex, _Suite],
) -> _Result:
    """Execute a supported scalar phase-QNode transform or return fail-closed evidence."""
    values = _bindings.finite_vector()("params", params)
    plan = _bindings.planner()(
        transform,
        gate=gate,
        observable=observable,
        backend=backend,
        adapter=adapter,
        n_params=values.size,
        shift_terms=shift_terms if rule is None else _bindings.builtins.length()(rule.terms),
        shots=shots,
        allow_hardware=allow_hardware,
    )
    label = ".".join(plan.transforms)
    if plan.fail_closed:
        return _bindings.blocked()(label, plan, values)
    if plan.transforms == ("grad",):
        gradient = _bindings.gradient()(objective, values, parameters=parameters, rule=rule)
        return _bindings.supported()(label, plan, values, gradient=gradient)
    if plan.transforms == ("value_and_grad",):
        result = _bindings.value_and_gradient()(
            objective, values, parameters=parameters, rule=rule
        )
        return _bindings.supported()(
            label,
            plan,
            values,
            value=_bindings.builtins.scalar()(result.value),
            gradient=_bindings.array_cast()(
                _bindings.array_type(),
                result.gradient.astype(_bindings.numpy().float64, copy=True),
            ),
        )
    if plan.transforms in {("hessian",), ("grad", "grad")}:
        hessian = _bindings.hessian()(objective, values, parameters=parameters, rule=rule)
        return _bindings.supported()(
            label,
            plan,
            values,
            hessian=hessian,
            parameter_shift_evaluations=_bindings.hessian_count()(
                values.size, plan.support_plan.backend_plan.shift_terms
            ),
        )
    if plan.transforms == ("jvp",):
        tangent_vector = _bindings.finite_vector()("tangent", tangent, width=values.size)
        value = _bindings.builtins.scalar()(objective(values.copy()))
        gradient = _bindings.gradient()(objective, values, parameters=parameters, rule=rule)
        return _bindings.supported()(
            label,
            plan,
            values,
            value=value,
            gradient=gradient,
            tangent=tangent_vector,
            jvp=_bindings.builtins.scalar()(_bindings.numpy().dot(gradient, tangent_vector)),
        )
    if plan.transforms == ("vjp",):
        cotangent_vector = _bindings.cotangent()(cotangent)
        value = _bindings.builtins.scalar()(objective(values.copy()))
        gradient = _bindings.gradient()(objective, values, parameters=parameters, rule=rule)
        return _bindings.supported()(
            label,
            plan,
            values,
            value=value,
            gradient=gradient,
            cotangent=cotangent_vector,
            vjp=cotangent_vector[0] * gradient,
        )
    if plan.transforms in {("jacfwd",), ("jacrev",)}:
        gradient = _bindings.gradient()(objective, values, parameters=parameters, rule=rule)
        jacobian = gradient.reshape(1, -1).astype(_bindings.numpy().float64, copy=True)
        return _bindings.supported()(label, plan, values, gradient=gradient, jacobian=jacobian)
    if plan.transforms == ("vmap", "grad"):
        return _bindings.result_class()(
            transform=label,
            plan=plan,
            params=values.copy(),
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
            failure_reason="scalar QNode transform executor does not accept batched parameters; use execute_phase_qnode_vmap_grad for native manual vmap(grad)",
        )
    return _bindings.blocked()(
        label,
        plan,
        values,
        reason=f"scalar QNode transform executor has no execution route for {label}; use the dedicated phase_qnode_tape or vector-transform API",
    )


def phase_qnode_complex_derivative_contract(
    requested_derivative: str = "wirtinger",
    *,
    _bindings: QNodeTransformBindings[_Result, _Complex, _Suite],
) -> _Complex:
    """Return the real-only contract for complex Phase-QNode derivative requests."""
    requested = requested_derivative.strip().lower()
    if requested not in {"complex", "wirtinger"}:
        raise _bindings.builtins.error()("requested_derivative must be 'complex' or 'wirtinger'")
    return _bindings.complex_class()(
        supported=False,
        parameter_domain="real",
        requested_derivative=requested,
        failure_reason="Complex and Wirtinger Phase-QNode derivatives are not implemented; the deterministic local transform surfaces accept real-valued parameter, tangent, cotangent, vector, and output arrays only",
        alternatives=(
            "split complex parameters into explicit real and imaginary real-valued controls",
            "use real-valued parameter-shift gradients on supported observables",
            "keep complex-state amplitudes internal to the statevector simulator",
        ),
        claim_boundary="real-valued parameter vectors and real scalar objectives only; complex-valued objectives, holomorphic derivatives, Wirtinger partials, and complex tangent/cotangent algebra are fail-closed",
    )


def execute_phase_qnode_hessian_vector_product(
    objective: ScalarObjective,
    params: ArrayLike,
    vector: ArrayLike,
    *,
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    adapter: str = "native",
    shots: int | None = None,
    shift_terms: int = 1,
    allow_hardware: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
    _bindings: QNodeTransformBindings[_Result, _Complex, _Suite],
) -> _Result:
    """Execute a deterministic local Hessian-vector product for a scalar QNode."""
    values = _bindings.finite_vector()("params", params)
    vector_values = _bindings.finite_vector()("vector", vector, width=values.size)
    plan = _bindings.planner()(
        "hessian",
        gate=gate,
        observable=observable,
        backend=backend,
        adapter=adapter,
        n_params=values.size,
        shift_terms=shift_terms if rule is None else _bindings.builtins.length()(rule.terms),
        shots=shots,
        allow_hardware=allow_hardware,
    )
    if plan.fail_closed:
        return _bindings.blocked()("hessian_vector_product", plan, values)
    hessian = _bindings.hessian()(objective, values, parameters=parameters, rule=rule)
    return _bindings.supported()(
        "hessian_vector_product",
        plan,
        values,
        hessian=hessian,
        hessian_vector_product=hessian @ vector_values,
        tangent=vector_values,
        parameter_shift_evaluations=_bindings.hessian_count()(
            values.size, plan.support_plan.backend_plan.shift_terms
        ),
        claim_boundary="deterministic local Hessian-vector product via parameter-shift Hessian materialisation; no finite-shot, hardware, sparse implicit HVP, or arbitrary-program second-order AD claim",
    )


def run_phase_qnode_transform_readiness_suite(
    objective: ScalarObjective, *, _bindings: QNodeTransformBindings[_Result, _Complex, _Suite]
) -> _Suite:
    """Run scalar QNode transform readiness evidence."""
    params = _bindings.numpy().array([0.2, -0.4], dtype=_bindings.numpy().float64)
    records = [
        _bindings.execute()("grad", objective, params),
        _bindings.execute()("value_and_grad", objective, params),
        _bindings.execute()("hessian", objective, params),
        _bindings.hessian_vector()(
            objective,
            params,
            _bindings.numpy().array([0.5, -1.25], dtype=_bindings.numpy().float64),
        ),
        _bindings.execute()(
            "jvp", objective, params, tangent=_bindings.numpy().array([0.5, -2.0])
        ),
        _bindings.execute()("vjp", objective, params, cotangent=_bindings.numpy().array([3.0])),
        _bindings.execute()("jacfwd", objective, params),
        _bindings.execute()("jacrev", objective, params),
        _bindings.execute()("grad", objective, params, backend="hardware", shots=1024),
        _bindings.execute()(
            "hessian", objective, params, backend="finite_shot_simulator", shots=256
        ),
        _bindings.execute()(("vmap", "grad"), objective, params),
    ]
    return _bindings.suite_class()(records=_bindings.sequence()(records))
