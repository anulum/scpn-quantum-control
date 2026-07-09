# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — canonical differentiable user namespace.
"""Canonical first-path namespace for differentiable quantum-control workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .differentiable_canonical_api import grad, value_and_grad
from .differentiable_finite_difference import hessian, jacfwd, jacobian, jacrev, jvp, vjp
from .differentiable_parameter_contracts import Parameter
from .differentiable_result_contracts import GradientResult
from .differentiable_vmap import vmap
from .phase.gradient_support_matrix import GradientSupportPlan, plan_gradient_support
from .phase.gradient_tape import QuantumGradientTape, TapeGradientRecord, gradient_tape

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective: TypeAlias = Callable[[FloatArray], float | int | np.floating[Any]]
DiffTransformName: TypeAlias = Literal[
    "grad",
    "value_and_grad",
    "jacfwd",
    "jacrev",
    "jacobian",
    "hessian",
    "jvp",
    "vjp",
    "vmap",
    "gradient_tape",
]

CANONICAL_DIFF_NAMESPACE: Final[str] = "scpn_quantum_control.diff"
COMPATIBILITY_DIFF_NAMESPACE: Final[str] = "scpn.diff"
DIFF_CLAIM_BOUNDARY: Final[str] = (
    "canonical differentiable namespace over supported local SCPN routes; "
    "JIT, provider callbacks, hardware gradients, and performance claims fail "
    "closed unless a dedicated evidence surface says otherwise"
)
DIFFERENTIABLE_CIRCUIT_SCHEMA: Final[str] = "scpn.diff.differentiable_circuit.v1"
CANONICAL_GRADIENT_METHODS: Final[tuple[str, ...]] = (
    "parameter_shift",
    "finite_difference",
    "complex_step",
    "forward_mode",
    "reverse_mode",
    "whole_program",
)


@dataclass(frozen=True)
class ShotPolicy:
    """Shot and hardware policy attached to a differentiable circuit."""

    shots: int | None = None
    seed: int | None = None
    allow_hardware: bool = False
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate shot-policy bounds and hardware safety coupling."""
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shot policy shots must be positive when provided")
        if self.seed is not None and self.seed < 0:
            raise ValueError("shot policy seed must be non-negative when provided")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("shot policy confidence_level must be between 0 and 1")
        if self.allow_hardware and self.shots is None:
            raise ValueError("hardware-enabled shot policy must declare shots")

    @property
    def finite_shot(self) -> bool:
        """Return true when the circuit is configured for finite-shot evidence."""
        return self.shots is not None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready shot policy metadata."""
        return {
            "shots": self.shots,
            "seed": self.seed,
            "allow_hardware": self.allow_hardware,
            "confidence_level": self.confidence_level,
            "finite_shot": self.finite_shot,
        }


@dataclass(frozen=True)
class EstimatorProvenance:
    """Provenance for the estimator route behind a differentiable circuit."""

    estimator: str
    route: str
    package_version: str
    artifact_ids: tuple[str, ...] = ()
    claim_boundary: str = DIFF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate estimator provenance fields."""
        if not self.estimator:
            raise ValueError("estimator provenance estimator must be non-empty")
        if not self.route:
            raise ValueError("estimator provenance route must be non-empty")
        if not self.package_version:
            raise ValueError("estimator provenance package_version must be non-empty")
        if any(not artifact_id for artifact_id in self.artifact_ids):
            raise ValueError("estimator provenance artifact_ids must be non-empty")
        if not self.claim_boundary:
            raise ValueError("estimator provenance claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready estimator provenance metadata."""
        return {
            "estimator": self.estimator,
            "route": self.route,
            "package_version": self.package_version,
            "artifact_ids": list(self.artifact_ids),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class BackendCapabilityMetadata:
    """User-facing capability metadata for one differentiable circuit route."""

    gate: str
    observable: str
    backend: str
    transform: str
    adapter: str
    supported: bool
    recommended_method: str
    evaluation_mode: str
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    alternatives: tuple[str, ...]
    requires_finite_shot_variance: bool
    requires_hardware_policy: bool
    claim_boundary: str

    @classmethod
    def from_plan(cls, plan: GradientSupportPlan) -> BackendCapabilityMetadata:
        """Build public capability metadata from a gradient support plan."""
        return cls(
            gate=plan.gate,
            observable=plan.observable,
            backend=plan.backend,
            transform=plan.transform,
            adapter=plan.adapter,
            supported=plan.supported,
            recommended_method=plan.recommended_method,
            evaluation_mode=plan.evaluation_mode,
            blocked_reasons=plan.blocked_reasons,
            warnings=plan.warnings,
            alternatives=plan.alternatives,
            requires_finite_shot_variance=plan.requires_finite_shot_variance,
            requires_hardware_policy=plan.requires_hardware_policy,
            claim_boundary=plan.claim_boundary,
        )

    @property
    def fail_closed(self) -> bool:
        """Return true when this route is intentionally unsupported."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready backend capability metadata."""
        return {
            "gate": self.gate,
            "observable": self.observable,
            "backend": self.backend,
            "transform": self.transform,
            "adapter": self.adapter,
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "recommended_method": self.recommended_method,
            "evaluation_mode": self.evaluation_mode,
            "blocked_reasons": list(self.blocked_reasons),
            "warnings": list(self.warnings),
            "alternatives": list(self.alternatives),
            "requires_finite_shot_variance": self.requires_finite_shot_variance,
            "requires_hardware_policy": self.requires_hardware_policy,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableCircuitDiagnostics:
    """Fail-closed diagnostics for a differentiable circuit route."""

    name: str
    supported: bool
    capability: BackendCapabilityMetadata
    shot_policy: ShotPolicy
    estimator_provenance: EstimatorProvenance
    claim_boundary: str

    @property
    def fail_closed(self) -> bool:
        """Return true when the circuit route is intentionally unsupported."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready diagnostics for API and documentation examples."""
        return {
            "name": self.name,
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "capability": self.capability.to_dict(),
            "shot_policy": self.shot_policy.to_dict(),
            "estimator_provenance": self.estimator_provenance.to_dict(),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class JITExplanation:
    """Fail-closed result returned by :func:`jit_or_explain`."""

    target: str
    compiled: bool
    blocked_reasons: tuple[str, ...]
    suggested_alternatives: tuple[str, ...]
    claim_boundary: str = DIFF_CLAIM_BOUNDARY

    @property
    def fail_closed(self) -> bool:
        """Return true when no compiled callable is exposed."""
        return not self.compiled

    def require_compiled(self) -> None:
        """Raise when a caller tries to treat an explanation as compiled code."""
        if self.fail_closed:
            joined = "; ".join(self.blocked_reasons)
            raise RuntimeError(f"JIT route is unsupported: {joined}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JIT route diagnostics."""
        return {
            "target": self.target,
            "compiled": self.compiled,
            "fail_closed": self.fail_closed,
            "blocked_reasons": list(self.blocked_reasons),
            "suggested_alternatives": list(self.suggested_alternatives),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableCircuit:
    """Callable, serializable scalar differentiable circuit facade.

    Parameters
    ----------
    name:
        Stable circuit identifier used in diagnostics and serialized metadata.
    objective:
        Local scalar objective. The current facade executes only local Python
        objectives and delegates gradients to existing SCPN transform routes.
    parameter_names:
        Optional names for the one-dimensional parameter vector.
    gate:
        Gate class used for support-matrix routing.
    observable:
        Observable class used for support-matrix routing.
    backend:
        Backend route used for support-matrix routing.
    transform:
        Transform route used for support-matrix routing.
    adapter:
        Framework adapter used for support-matrix routing.
    gradient_method:
        Canonical gradient method used by :meth:`value_and_grad` when callers
        do not override the method explicitly.
    shot_policy:
        Finite-shot and hardware policy.
    estimator_provenance:
        Provenance for the estimator route.
    claim_boundary:
        Explicit claim boundary for public diagnostics.
    """

    name: str
    objective: ScalarObjective
    parameter_names: tuple[str, ...] = ()
    gate: str = "ry"
    observable: str = "pauli_expectation"
    backend: str = "statevector"
    transform: str = "grad"
    adapter: str = "native"
    gradient_method: str = "parameter_shift"
    shot_policy: ShotPolicy = ShotPolicy()
    estimator_provenance: EstimatorProvenance | None = None
    claim_boundary: str = DIFF_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate circuit metadata and attach default estimator provenance."""
        if not self.name.strip():
            raise ValueError("differentiable circuit name must be non-empty")
        if not callable(self.objective):
            raise ValueError("differentiable circuit objective must be callable")
        if any(not name for name in self.parameter_names):
            raise ValueError("differentiable circuit parameter_names must be non-empty")
        if self.gradient_method not in CANONICAL_GRADIENT_METHODS:
            allowed = ", ".join(CANONICAL_GRADIENT_METHODS)
            raise ValueError(f"differentiable circuit gradient_method must be one of: {allowed}")
        if not self.claim_boundary:
            raise ValueError("differentiable circuit claim_boundary must be non-empty")
        if self.estimator_provenance is None:
            object.__setattr__(
                self,
                "estimator_provenance",
                EstimatorProvenance(
                    estimator="local_scalar_objective",
                    route=(
                        f"{self.adapter}:{self.backend}:{self.transform}:{self.gradient_method}"
                    ),
                    package_version=_package_version(),
                    artifact_ids=(),
                    claim_boundary=self.claim_boundary,
                ),
            )

    @property
    def support_plan(self) -> GradientSupportPlan:
        """Return the current fail-closed support plan for this circuit."""
        return plan_gradient_support(
            gate=self.gate,
            observable=self.observable,
            backend=self.backend,
            transform=self.transform,
            adapter=self.adapter,
            n_params=max(1, len(self.parameter_names)),
            shots=self.shot_policy.shots,
            allow_hardware=self.shot_policy.allow_hardware,
        )

    @property
    def capability(self) -> BackendCapabilityMetadata:
        """Return public capability metadata for the current route."""
        return BackendCapabilityMetadata.from_plan(self.support_plan)

    @property
    def diagnostics(self) -> DifferentiableCircuitDiagnostics:
        """Return fail-closed diagnostics for the current route."""
        return DifferentiableCircuitDiagnostics(
            name=self.name,
            supported=self.support_plan.supported,
            capability=self.capability,
            shot_policy=self.shot_policy,
            estimator_provenance=self._provenance,
            claim_boundary=self.claim_boundary,
        )

    @property
    def fail_closed(self) -> bool:
        """Return true when the current route is intentionally unsupported."""
        return self.diagnostics.fail_closed

    @property
    def _provenance(self) -> EstimatorProvenance:
        provenance = self.estimator_provenance
        if provenance is None:
            raise RuntimeError("differentiable circuit provenance was not initialised")
        return provenance

    def __call__(self, values: ArrayLike) -> float:
        """Evaluate the local scalar objective after support validation."""
        self._require_supported()
        parameter_values = _parameter_vector(values)
        return _objective_value(self.objective, parameter_values)

    def value_and_grad(
        self,
        values: ArrayLike,
        *,
        method: str | None = None,
        parameters: Sequence[Parameter] | None = None,
        step: float | None = None,
    ) -> GradientResult:
        """Evaluate objective value and gradient through the canonical transform."""
        self._require_supported()
        result = value_and_grad(
            self.objective,
            values,
            parameters=self._parameters(values, parameters),
            method=self.gradient_method if method is None else method,
            step=step,
        )
        return cast(GradientResult, result)

    def grad(
        self,
        values: ArrayLike,
        *,
        method: str | None = None,
        parameters: Sequence[Parameter] | None = None,
        step: float | None = None,
    ) -> FloatArray:
        """Evaluate a gradient through the canonical transform namespace."""
        return self.value_and_grad(
            values,
            method=method,
            parameters=parameters,
            step=step,
        ).gradient

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready metadata without serializing executable code."""
        payload: dict[str, object] = {
            "schema": DIFFERENTIABLE_CIRCUIT_SCHEMA,
            "name": self.name,
            "objective": _callable_label(self.objective),
            "parameter_names": list(self.parameter_names),
            "gate": self.gate,
            "observable": self.observable,
            "backend": self.backend,
            "transform": self.transform,
            "adapter": self.adapter,
            "gradient_method": self.gradient_method,
            "diagnostics": self.diagnostics.to_dict(),
            "claim_boundary": self.claim_boundary,
        }
        payload["serialization_provenance"] = _serialization_provenance(payload)
        return payload

    def to_json(self) -> str:
        """Return deterministic JSON metadata for audit artifacts."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def _require_supported(self) -> None:
        plan = self.support_plan
        if plan.fail_closed:
            joined = "; ".join(plan.blocked_reasons)
            raise ValueError(f"differentiable circuit route is unsupported: {joined}")

    def _parameters(
        self,
        values: ArrayLike,
        parameters: Sequence[Parameter] | None,
    ) -> Sequence[Parameter] | None:
        if parameters is not None or not self.parameter_names:
            return parameters
        values_array = _parameter_vector(values)
        if values_array.size != len(self.parameter_names):
            raise ValueError("parameter_names length must match values length")
        return tuple(Parameter(name=name) for name in self.parameter_names)


QuantumFunction: TypeAlias = DifferentiableCircuit


def differentiable_circuit(
    objective: ScalarObjective,
    *,
    name: str = "differentiable_circuit",
    parameter_names: Sequence[str] = (),
    gate: str = "ry",
    observable: str = "pauli_expectation",
    backend: str = "statevector",
    transform: str = "grad",
    adapter: str = "native",
    gradient_method: str = "parameter_shift",
    shot_policy: ShotPolicy | None = None,
    estimator_provenance: EstimatorProvenance | None = None,
) -> DifferentiableCircuit:
    """Return a configured differentiable circuit facade for a scalar objective."""
    return DifferentiableCircuit(
        name=name,
        objective=objective,
        parameter_names=tuple(parameter_names),
        gate=gate,
        observable=observable,
        backend=backend,
        transform=transform,
        adapter=adapter,
        gradient_method=gradient_method,
        shot_policy=ShotPolicy() if shot_policy is None else shot_policy,
        estimator_provenance=estimator_provenance,
    )


def jit_or_explain(
    function: Callable[..., object] | DifferentiableCircuit,
    *,
    backend: str = "statevector",
    adapter: str = "native",
) -> JITExplanation:
    """Return a fail-closed JIT explanation for the canonical namespace.

    The project currently exposes executable local gradients and compiler-AD
    evidence surfaces separately. This helper gives first-path users a stable
    JIT entry point that refuses unsupported compilation routes with actionable
    alternatives instead of silently falling back to eager execution.
    """
    target = (
        function.name if isinstance(function, DifferentiableCircuit) else _callable_label(function)
    )
    return JITExplanation(
        target=target,
        compiled=False,
        blocked_reasons=(
            f"no promoted JIT route for adapter={adapter!r} backend={backend!r}",
            "use eager transforms or compiler report surfaces until executable lowering is promoted",
        ),
        suggested_alternatives=(
            "grad",
            "value_and_grad",
            "differentiable_api('compile_report')",
            "differentiable_api('frontend_report')",
        ),
    )


def supported_transforms() -> tuple[DiffTransformName, ...]:
    """Return the stable transform names exposed by the canonical namespace."""
    return (
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "jacobian",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "gradient_tape",
    )


def namespace_metadata() -> dict[str, object]:
    """Return JSON-ready metadata for the canonical differentiable namespace."""
    return {
        "namespace": CANONICAL_DIFF_NAMESPACE,
        "compatibility_namespace": COMPATIBILITY_DIFF_NAMESPACE,
        "transforms": list(supported_transforms()),
        "circuit_surface": "DifferentiableCircuit",
        "quantum_function_alias": "QuantumFunction",
        "claim_boundary": DIFF_CLAIM_BOUNDARY,
    }


def _parameter_vector(values: ArrayLike) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "differentiable circuit values must be a one-dimensional numeric array"
        ) from exc
    if array.ndim != 1:
        raise ValueError("differentiable circuit values must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("differentiable circuit values must be finite")
    return array.astype(np.float64, copy=True)


def _objective_value(objective: ScalarObjective, values: FloatArray) -> float:
    value = objective(values.copy())
    scalar = np.asarray(value, dtype=np.float64)
    if scalar.shape != ():
        raise ValueError("differentiable circuit objective must return a scalar")
    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError("differentiable circuit objective returned a non-finite value")
    return result


def _callable_label(function: Callable[..., object]) -> str:
    module = getattr(function, "__module__", "")
    qualname = getattr(
        function, "__qualname__", getattr(function, "__name__", type(function).__name__)
    )
    return f"{module}.{qualname}" if module else str(qualname)


def _package_version() -> str:
    try:
        return version("scpn-quantum-control")
    except PackageNotFoundError:
        return "editable-local"


def _serialization_provenance(metadata: dict[str, object]) -> dict[str, object]:
    digest_payload = dict(metadata)
    encoded = json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": DIFFERENTIABLE_CIRCUIT_SCHEMA,
        "metadata_digest": hashlib.sha256(encoded).hexdigest(),
        "serializes_executable_code": False,
        "objective_label": metadata["objective"],
        "gradient_method": metadata["gradient_method"],
        "claim_boundary": metadata["claim_boundary"],
    }


from .diff_contract_audit import (  # noqa: E402
    DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY,
    DifferentiableCircuitContractAuditResult,
    DifferentiableCircuitContractCheck,
    DifferentiableCircuitContractStatus,
    run_differentiable_circuit_contract_audit,
)

__all__ = [
    "BackendCapabilityMetadata",
    "CANONICAL_DIFF_NAMESPACE",
    "CANONICAL_GRADIENT_METHODS",
    "COMPATIBILITY_DIFF_NAMESPACE",
    "DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY",
    "DIFFERENTIABLE_CIRCUIT_SCHEMA",
    "DIFF_CLAIM_BOUNDARY",
    "DifferentiableCircuit",
    "DifferentiableCircuitContractAuditResult",
    "DifferentiableCircuitContractCheck",
    "DifferentiableCircuitContractStatus",
    "DifferentiableCircuitDiagnostics",
    "DiffTransformName",
    "EstimatorProvenance",
    "JITExplanation",
    "QuantumFunction",
    "QuantumGradientTape",
    "ShotPolicy",
    "TapeGradientRecord",
    "differentiable_circuit",
    "grad",
    "gradient_tape",
    "hessian",
    "jacfwd",
    "jacobian",
    "jacrev",
    "jit_or_explain",
    "jvp",
    "namespace_metadata",
    "run_differentiable_circuit_contract_audit",
    "supported_transforms",
    "value_and_grad",
    "vjp",
    "vmap",
]
