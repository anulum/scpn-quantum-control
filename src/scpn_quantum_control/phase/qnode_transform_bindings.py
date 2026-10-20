# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Scalar QNode execution dependencies
"""Typed live operation suppliers for the original scalar QNode executor."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar, cast

from numpy.typing import ArrayLike

from scpn_quantum_control.differentiable import GradientResult, Parameter, ParameterShiftRule
from scpn_quantum_control.phase.qnode_transform_contract_bindings import (
    FloatArray,
    QNodeContractBindings,
    ScalarObjective,
)
from scpn_quantum_control.phase.transform_nesting import GradientTransformNestingPlan

_Result = TypeVar("_Result")
_Complex = TypeVar("_Complex")
_Suite = TypeVar("_Suite")
_ResultCo = TypeVar("_ResultCo", covariant=True)
_ComplexCo = TypeVar("_ComplexCo", covariant=True)
_SuiteCo = TypeVar("_SuiteCo", covariant=True)
_ResultContra = TypeVar("_ResultContra", contravariant=True)


class _AsVector(Protocol):
    """Admit an actual parameter or directional vector."""

    def __call__(
        self, name: str, values: ArrayLike | None, *, width: int | None = None
    ) -> FloatArray:
        """Retain the original missing-value, shape and finiteness refusals."""
        ...


class _Derivative(Protocol):
    """Read the original parameter-shift derivative and legacy shift keyword."""

    def __call__(
        self,
        objective: ScalarObjective,
        values: ArrayLike,
        shift: float = ...,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
    ) -> FloatArray:
        """Evaluate the actual current derivative supplier."""
        ...


class _ValueAndGradient(Protocol):
    """Read the existing automatic-derivative value and structured result."""

    def __call__(
        self,
        objective: ScalarObjective,
        values: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
    ) -> GradientResult:
        """Evaluate the original value and gradient supplier."""
        ...


class _Planner(Protocol):
    """Read the actual supported-nesting admission planner."""

    def __call__(
        self,
        transforms: str | tuple[str, ...],
        *,
        gate: str,
        observable: str,
        backend: str,
        adapter: str,
        n_params: int,
        shift_terms: int,
        shots: int | None,
        allow_hardware: bool,
    ) -> GradientTransformNestingPlan:
        """Retain the original backend, gate and transform admission."""
        ...


class _ComplexFactory(Protocol[_ComplexCo]):
    """Construct the real-only complex-derivative refusal record."""

    def __call__(
        self,
        *,
        supported: bool,
        parameter_domain: str,
        requested_derivative: str,
        failure_reason: str,
        alternatives: tuple[str, ...],
        claim_boundary: str,
    ) -> _ComplexCo:
        """Construct the real caller-owned record with its original fields."""
        ...


class _SuiteFactory(Protocol[_ResultContra, _SuiteCo]):
    """Construct the original scalar readiness aggregation record."""

    def __call__(
        self,
        *,
        records: tuple[_ResultContra, ...],
        evidence_class: str = ...,
        claim_boundary: str = ...,
        hardware_execution: bool = ...,
    ) -> _SuiteCo:
        """Construct the real caller-owned record with its original fields."""
        ...


class _SupportedResult(Protocol[_ResultCo]):
    """Call the supported-result helper with its original array-copy arguments."""

    def __call__(
        self,
        transform: str,
        plan: GradientTransformNestingPlan,
        params: FloatArray,
        *,
        value: float | None = ...,
        gradient: FloatArray | None = ...,
        hessian: FloatArray | None = ...,
        hessian_vector_product: FloatArray | None = ...,
        jvp: float | None = ...,
        vjp: FloatArray | None = ...,
        jacobian: FloatArray | None = ...,
        tangent: FloatArray | None = ...,
        cotangent: FloatArray | None = ...,
        parameter_shift_evaluations: int | None = ...,
        claim_boundary: str = ...,
    ) -> _ResultCo:
        """Call the actual source supplier with its original arguments."""
        ...


class _BlockedResult(Protocol[_ResultCo]):
    """Call the refusal helper with its original plan and optional reason."""

    def __call__(
        self,
        transform: str,
        plan: GradientTransformNestingPlan,
        params: FloatArray,
        *,
        reason: str | None = ...,
    ) -> _ResultCo:
        """Call the actual source supplier with its original arguments."""
        ...


class _ScalarExecutor(Protocol[_ResultCo]):
    """Execute a scalar transform with its original directional arguments."""

    def __call__(
        self,
        transform: str | tuple[str, ...],
        objective: ScalarObjective,
        params: ArrayLike,
        *,
        tangent: ArrayLike | None = ...,
        cotangent: ArrayLike | float | None = ...,
        gate: str = ...,
        observable: str = ...,
        backend: str = ...,
        adapter: str = ...,
        shots: int | None = ...,
        shift_terms: int = ...,
        allow_hardware: bool = ...,
        parameters: Sequence[Parameter] | None = ...,
        rule: ParameterShiftRule | None = ...,
    ) -> _ResultCo:
        """Call the actual source supplier with its original arguments."""
        ...


class _HessianVectorExecutor(Protocol[_ResultCo]):
    """Read the Hessian-vector composition used by scalar readiness."""

    def __call__(
        self,
        objective: ScalarObjective,
        params: ArrayLike,
        vector: ArrayLike,
        *,
        gate: str = ...,
        observable: str = ...,
        backend: str = ...,
        adapter: str = ...,
        shots: int | None = ...,
        shift_terms: int = ...,
        allow_hardware: bool = ...,
        parameters: Sequence[Parameter] | None = ...,
        rule: ParameterShiftRule | None = ...,
    ) -> _ResultCo:
        """Call the actual source supplier with its original arguments."""
        ...


class QNodeTransformBindings(QNodeContractBindings[_Result], Generic[_Result, _Complex, _Suite]):
    """Read actual scalar execution suppliers from the defining source."""

    def complex_class(self) -> _ComplexFactory[_Complex]:
        """Read the current real-only complex-derivative refusal constructor."""
        return cast(_ComplexFactory[_Complex], self._read("PhaseQNodeComplexDerivativeContract"))

    def suite_class(self) -> _SuiteFactory[_Result, _Suite]:
        """Read the current scalar readiness aggregation constructor."""
        return cast(
            _SuiteFactory[_Result, _Suite], self._read("PhaseQNodeTransformReadinessSuiteResult")
        )

    def finite_vector(self) -> _AsVector:
        """Read the current shape and finite-value admission helper."""
        return cast(_AsVector, self._read("_as_parameter_vector"))

    def supported(self) -> _SupportedResult[_Result]:
        """Read the supported-result helper with its original copy semantics."""
        return cast(_SupportedResult[_Result], self._read("_supported_result"))

    def blocked(self) -> _BlockedResult[_Result]:
        """Read the current refusal-result helper."""
        return cast(_BlockedResult[_Result], self._read("_blocked_result"))

    def hessian_count(self) -> Callable[[int, int], int]:
        """Read the original diagonal and mixed-probe counting helper."""
        return cast(Callable[[int, int], int], self._read("_hessian_evaluations"))

    def cotangent(self) -> Callable[[ArrayLike | float | None], FloatArray]:
        """Read the current scalar cotangent admission helper."""
        return cast(Callable[[ArrayLike | float | None], FloatArray], self._read("_as_cotangent"))

    def gradient(self) -> _Derivative:
        """Read the current gradient function with its legacy shift argument."""
        return cast(_Derivative, self._read("parameter_shift_gradient"))

    def hessian(self) -> _Derivative:
        """Read the current Hessian function with its legacy shift argument."""
        return cast(_Derivative, self._read("parameter_shift_hessian"))

    def value_and_gradient(self) -> _ValueAndGradient:
        """Read the current structured value-and-gradient operation."""
        return cast(_ValueAndGradient, self._read("value_and_parameter_shift_grad"))

    def planner(self) -> _Planner:
        """Read the current backend and transform admission planner."""
        return cast(_Planner, self._read("plan_gradient_transform_nesting"))

    def execute(self) -> _ScalarExecutor[_Result]:
        """Read the scalar executor used by the readiness suite."""
        return cast(_ScalarExecutor[_Result], self._read("execute_phase_qnode_transform"))

    def hessian_vector(self) -> _HessianVectorExecutor[_Result]:
        """Read the Hessian-vector executor used by the readiness suite."""
        return cast(
            _HessianVectorExecutor[_Result],
            self._read("execute_phase_qnode_hessian_vector_product"),
        )
