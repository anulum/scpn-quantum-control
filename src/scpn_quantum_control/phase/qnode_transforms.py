# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Transforms
"""Executable scalar phase-QNode transform evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_quantum_control.phase import qnode_transform_algorithms as _execution_algorithms
from scpn_quantum_control.phase import qnode_transform_contract_algorithms as _contract_algorithms
from scpn_quantum_control.phase.qnode_transform_bindings import (
    QNodeTransformBindings as _ExecutionBindings,
)
from scpn_quantum_control.phase.qnode_transform_contract_bindings import (
    QNodeContractBindings as _ContractBindings,
)
from scpn_quantum_control.phase.qnode_transform_contract_bindings import (
    QNodeTransformBuiltins as _BuiltinReaders,
)

from ..differentiable import Parameter, ParameterShiftRule
from .param_shift import (
    parameter_shift_gradient,
    parameter_shift_hessian,
    value_and_parameter_shift_grad,
)
from .transform_nesting import GradientTransformNestingPlan, plan_gradient_transform_nesting

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]

EVIDENCE_CLASS = "phase_qnode_transform_execution"
CLAIM_BOUNDARY = (
    "supported scalar local phase-QNode transforms only; not hardware execution, "
    "not arbitrary program AD, and not full vector-output framework transform algebra"
)


@dataclass(frozen=True)
class PhaseQNodeComplexDerivativeContract:
    """Fail-closed complex/Wirtinger derivative contract for Phase-QNode transforms."""

    supported: bool
    parameter_domain: str
    requested_derivative: str
    failure_reason: str
    alternatives: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready complex derivative contract evidence."""
        return {
            "supported": self.supported,
            "parameter_domain": self.parameter_domain,
            "requested_derivative": self.requested_derivative,
            "failure_reason": self.failure_reason,
            "alternatives": list(self.alternatives),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseQNodeTransformResult:
    """Execution or fail-closed evidence for one scalar phase-QNode transform."""

    transform: str
    plan: GradientTransformNestingPlan
    params: FloatArray
    supported: bool
    value: float | None
    gradient: FloatArray | None
    hessian: FloatArray | None
    hessian_vector_product: FloatArray | None
    jvp: float | None
    vjp: FloatArray | None
    jacobian: FloatArray | None
    tangent: FloatArray | None
    cotangent: FloatArray | None
    parameter_shift_evaluations: int
    failure_reason: str
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def fail_closed(self) -> bool:
        """True when execution was intentionally refused."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready transform evidence."""
        return {
            "transform": self.transform,
            "plan": self.plan.to_dict(),
            "params": self.params.tolist(),
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "hessian": None if self.hessian is None else self.hessian.tolist(),
            "hessian_vector_product": (
                None
                if self.hessian_vector_product is None
                else self.hessian_vector_product.tolist()
            ),
            "jvp": self.jvp,
            "vjp": None if self.vjp is None else self.vjp.tolist(),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "tangent": None if self.tangent is None else self.tangent.tolist(),
            "cotangent": None if self.cotangent is None else self.cotangent.tolist(),
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "failure_reason": self.failure_reason,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class PhaseQNodeTransformReadinessSuiteResult:
    """Readiness evidence for supported and blocked scalar QNode transforms."""

    records: tuple[PhaseQNodeTransformResult, ...]
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def record_count(self) -> int:
        """Total records."""
        return len(self.records)

    @property
    def supported_count(self) -> int:
        """Supported transform records."""
        return sum(1 for record in self.records if record.supported)

    @property
    def fail_closed_count(self) -> int:
        """Fail-closed transform records."""
        return sum(1 for record in self.records if record.fail_closed)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Total parameter-shift evaluations across supported records."""
        return sum(record.parameter_shift_evaluations for record in self.records)

    @property
    def passed(self) -> bool:
        """True when default supported routes execute and blocked routes refuse."""
        supported = {record.transform for record in self.records if record.supported}
        blocked = [record for record in self.records if record.fail_closed]
        return (
            {
                "grad",
                "value_and_grad",
                "hessian",
                "hessian_vector_product",
                "jvp",
                "vjp",
                "jacfwd",
                "jacrev",
            }.issubset(supported)
            and len(blocked) >= 3
            and not self.hardware_execution
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "supported_count": self.supported_count,
            "fail_closed_count": self.fail_closed_count,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "records": [record.to_dict() for record in self.records],
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


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
) -> PhaseQNodeTransformResult:
    """Execute a supported scalar phase-QNode transform or return fail-closed evidence."""
    return _execution_algorithms.execute_phase_qnode_transform(
        transform,
        objective,
        params,
        tangent=tangent,
        cotangent=cotangent,
        gate=gate,
        observable=observable,
        backend=backend,
        adapter=adapter,
        shots=shots,
        shift_terms=shift_terms,
        allow_hardware=allow_hardware,
        parameters=parameters,
        rule=rule,
        _bindings=_qnode_bindings(),
    )


def phase_qnode_complex_derivative_contract(
    requested_derivative: str = "wirtinger",
) -> PhaseQNodeComplexDerivativeContract:
    """Return the real-only contract for complex Phase-QNode derivative requests."""
    return _execution_algorithms.phase_qnode_complex_derivative_contract(
        requested_derivative, _bindings=_qnode_bindings()
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
) -> PhaseQNodeTransformResult:
    """Execute a deterministic local Hessian-vector product for a scalar QNode."""
    return _execution_algorithms.execute_phase_qnode_hessian_vector_product(
        objective,
        params,
        vector,
        gate=gate,
        observable=observable,
        backend=backend,
        adapter=adapter,
        shots=shots,
        shift_terms=shift_terms,
        allow_hardware=allow_hardware,
        parameters=parameters,
        rule=rule,
        _bindings=_qnode_bindings(),
    )


def run_phase_qnode_transform_readiness_suite() -> PhaseQNodeTransformReadinessSuiteResult:
    """Run scalar QNode transform readiness evidence."""

    def objective(params: FloatArray) -> float:
        """Evaluate the original two-control trigonometric readiness observable."""
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    return _execution_algorithms.run_phase_qnode_transform_readiness_suite(
        objective, _bindings=_qnode_bindings()
    )


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
    claim_boundary: str = CLAIM_BOUNDARY,
) -> PhaseQNodeTransformResult:
    """Construct the original supported scalar transform record and retain independent arrays."""
    return _contract_algorithms._supported_result(
        transform,
        plan,
        params,
        value=value,
        gradient=gradient,
        hessian=hessian,
        hessian_vector_product=hessian_vector_product,
        jvp=jvp,
        vjp=vjp,
        jacobian=jacobian,
        tangent=tangent,
        cotangent=cotangent,
        parameter_shift_evaluations=parameter_shift_evaluations,
        claim_boundary=claim_boundary,
        _bindings=_contract_bindings(),
    )


def _blocked_result(
    transform: str,
    plan: GradientTransformNestingPlan,
    params: FloatArray,
    *,
    reason: str | None = None,
) -> PhaseQNodeTransformResult:
    """Construct the original refused scalar transform record and its original reason."""
    return _contract_algorithms._blocked_result(
        transform, plan, params, reason=reason, _bindings=_contract_bindings()
    )


def _as_parameter_vector(
    name: str, values: ArrayLike | None, *, width: int | None = None
) -> FloatArray:
    """Admit an independent real finite parameter or directional vector."""
    return _contract_algorithms._as_parameter_vector(
        name, values, width=width, _bindings=_contract_bindings()
    )


def _hessian_evaluations(n_params: int, shift_terms: int) -> int:
    """Count the original diagonal and mixed single-term Hessian probes."""
    return _contract_algorithms._hessian_evaluations(n_params, shift_terms)


def _as_cotangent(values: ArrayLike | float | None) -> FloatArray:
    """Admit the original real finite scalar or one-element cotangent."""
    return _contract_algorithms._as_cotangent(values, _bindings=_contract_bindings())


__all__ = [
    "PhaseQNodeComplexDerivativeContract",
    "PhaseQNodeTransformReadinessSuiteResult",
    "PhaseQNodeTransformResult",
    "execute_phase_qnode_hessian_vector_product",
    "execute_phase_qnode_transform",
    "phase_qnode_complex_derivative_contract",
    "run_phase_qnode_transform_readiness_suite",
]

_dependency_readers: dict[str, Callable[[], object]] = {
    "PhaseQNodeTransformResult": lambda: PhaseQNodeTransformResult,
    "cast": lambda: cast,
    "FloatArray": lambda: FloatArray,
    "PhaseQNodeComplexDerivativeContract": lambda: PhaseQNodeComplexDerivativeContract,
    "PhaseQNodeTransformReadinessSuiteResult": lambda: PhaseQNodeTransformReadinessSuiteResult,
    "_as_parameter_vector": lambda: _as_parameter_vector,
    "_supported_result": lambda: _supported_result,
    "_blocked_result": lambda: _blocked_result,
    "_hessian_evaluations": lambda: _hessian_evaluations,
    "_as_cotangent": lambda: _as_cotangent,
    "parameter_shift_gradient": lambda: parameter_shift_gradient,
    "parameter_shift_hessian": lambda: parameter_shift_hessian,
    "value_and_parameter_shift_grad": lambda: value_and_parameter_shift_grad,
    "plan_gradient_transform_nesting": lambda: plan_gradient_transform_nesting,
    "execute_phase_qnode_transform": lambda: execute_phase_qnode_transform,
    "execute_phase_qnode_hessian_vector_product": lambda: (
        execute_phase_qnode_hessian_vector_product
    ),
}


def _numpy_reader() -> object:
    """Read the actual defining source numerical namespace."""
    return np


_builtin_readers = _BuiltinReaders(
    scalar=lambda: float,
    length=lambda: len,
    sequence=lambda: tuple,
    error=lambda: ValueError,
)


def _contract_bindings() -> _ContractBindings[PhaseQNodeTransformResult]:
    """Retain genuine local record ownership and original source lookup."""
    return _ContractBindings(_dependency_readers, _numpy_reader, _builtin_readers)


def _qnode_bindings() -> _ExecutionBindings[
    PhaseQNodeTransformResult,
    PhaseQNodeComplexDerivativeContract,
    PhaseQNodeTransformReadinessSuiteResult,
]:
    """Read the actual execution suppliers from this defining source."""
    return _ExecutionBindings(_dependency_readers, _numpy_reader, _builtin_readers)
