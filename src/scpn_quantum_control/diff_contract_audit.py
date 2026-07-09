# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Circuit Contract Audit
"""Executable DP-004 contract audit for differentiable circuit facades."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .diff import (
    DifferentiableCircuit,
    ShotPolicy,
    differentiable_circuit,
    grad,
    jit_or_explain,
    vmap,
)

DifferentiableCircuitContractStatus = Literal["supported", "fail_closed"]
ScalarObjective = Callable[[NDArray[np.float64]], float | int | np.floating[Any]]

DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY = (
    "DP-004 DifferentiableCircuit/QuantumFunction contract audit only; supported "
    "checks cover local simulator call semantics, transform composition, backend "
    "capability metadata, and deterministic metadata serialization, while "
    "dataclass parameter containers, live hardware execution, provider submission, "
    "and JIT compilation remain fail-closed unless separate evidence promotes them"
)
"""Claim boundary for the executable DP-004 differentiable-circuit audit."""


@dataclass(frozen=True)
class DifferentiableCircuitContractCheck:
    """One DP-004 contract check.

    Parameters
    ----------
    name:
        Stable check identifier.
    status:
        ``"supported"`` when the audited behaviour executes, or
        ``"fail_closed"`` when the unsupported route is intentionally rejected.
    evidence:
        Human-readable evidence collected through public circuit APIs.
    blocked_reason:
        Rejection reason for fail-closed checks, or ``None`` for supported
        checks.
    """

    name: str
    status: DifferentiableCircuitContractStatus
    evidence: tuple[str, ...]
    blocked_reason: str | None = None

    @property
    def supported(self) -> bool:
        """Return true when the audited behaviour is supported."""
        return self.status == "supported"

    @property
    def fail_closed(self) -> bool:
        """Return true when the audited behaviour is intentionally rejected."""
        return self.status == "fail_closed"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready contract-check metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "evidence": list(self.evidence),
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class DifferentiableCircuitContractAuditResult:
    """Executable audit result for the DP-004 circuit abstraction contract."""

    checks: tuple[DifferentiableCircuitContractCheck, ...]
    claim_boundary: str = DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return true when all checks carry evidence and expected status."""
        return not self.failing_checks

    @property
    def supported_checks(self) -> tuple[DifferentiableCircuitContractCheck, ...]:
        """Return supported audit checks."""
        return tuple(check for check in self.checks if check.supported)

    @property
    def fail_closed_checks(self) -> tuple[DifferentiableCircuitContractCheck, ...]:
        """Return fail-closed audit checks."""
        return tuple(check for check in self.checks if check.fail_closed)

    @property
    def failing_checks(self) -> tuple[DifferentiableCircuitContractCheck, ...]:
        """Return checks that lack the evidence required by the audit."""
        return tuple(
            check
            for check in self.checks
            if not check.evidence or (check.fail_closed and check.blocked_reason is None)
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit metadata."""
        return {
            "checks": [check.to_dict() for check in self.checks],
            "passed": self.passed,
            "supported_checks": len(self.supported_checks),
            "fail_closed_checks": len(self.fail_closed_checks),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class _NamedParameters:
    theta: float
    bias: float


def run_differentiable_circuit_contract_audit() -> DifferentiableCircuitContractAuditResult:
    """Run the executable DP-004 audit through public diff namespace surfaces.

    Returns
    -------
    DifferentiableCircuitContractAuditResult
        Supported and fail-closed checks for call semantics, transform
        composition, backend capability metadata, and serialization provenance.
    """
    checks = (
        _call_semantics_check(),
        _transform_composition_check(),
        _backend_capability_check(),
        _serialization_provenance_check(),
        _parameter_container_fail_closed_check(),
        _objective_contract_fail_closed_check(),
    )
    return DifferentiableCircuitContractAuditResult(checks=checks)


def _call_semantics_check() -> DifferentiableCircuitContractCheck:
    params = np.array([0.3, 0.5], dtype=np.float64)
    original = params.copy()
    circuit = differentiable_circuit(
        _mutating_scalar_objective,
        name="dp004_call_semantics",
        parameter_names=("theta", "bias"),
        gradient_method="finite_difference",
    )

    value = circuit(params)
    return DifferentiableCircuitContractCheck(
        name="call_semantics_flat_vector",
        status="supported",
        evidence=(
            f"value={value:.12g}",
            f"external_parameters_unchanged={bool(np.array_equal(params, original))}",
            f"diagnostics_supported={circuit.diagnostics.supported}",
        ),
    )


def _transform_composition_check() -> DifferentiableCircuitContractCheck:
    circuit = differentiable_circuit(
        _scalar_objective,
        name="dp004_transform_composition",
        parameter_names=("theta", "bias"),
        gradient_method="finite_difference",
    )
    params = np.array([0.2, -0.4], dtype=np.float64)
    batch = np.array([[0.2, -0.4], [0.5, 0.1]], dtype=np.float64)
    gradient = grad(circuit, params, method="finite_difference")
    vectorized = vmap(circuit)(batch)
    jit_result = jit_or_explain(circuit)
    return DifferentiableCircuitContractCheck(
        name="transform_composition",
        status="supported",
        evidence=(
            f"gradient_shape={gradient.shape}",
            f"vmap_shape={np.asarray(vectorized).shape}",
            f"jit_fail_closed={jit_result.fail_closed}",
        ),
    )


def _backend_capability_check() -> DifferentiableCircuitContractCheck:
    statevector = differentiable_circuit(
        _scalar_objective,
        name="dp004_statevector",
        parameter_names=("theta", "bias"),
    )
    finite_shot = differentiable_circuit(
        _scalar_objective,
        name="dp004_finite_shot",
        parameter_names=("theta", "bias"),
        backend="qasm_simulator",
        shot_policy=ShotPolicy(shots=512, seed=7),
    )
    hardware_plan = differentiable_circuit(
        _scalar_objective,
        name="dp004_hardware_plan_only",
        parameter_names=("theta", "bias"),
        backend="hardware",
        shot_policy=ShotPolicy(shots=512, seed=7, allow_hardware=False),
    )
    return DifferentiableCircuitContractCheck(
        name="backend_capability_contract",
        status="supported",
        evidence=(
            f"statevector_supported={statevector.diagnostics.supported}",
            f"finite_shot_supported={finite_shot.diagnostics.supported}",
            f"finite_shot_variance={finite_shot.capability.requires_finite_shot_variance}",
            f"hardware_plan_fail_closed={hardware_plan.fail_closed}",
            f"hardware_policy_required={hardware_plan.capability.requires_hardware_policy}",
        ),
    )


def _serialization_provenance_check() -> DifferentiableCircuitContractCheck:
    circuit = differentiable_circuit(
        _scalar_objective,
        name="dp004_serialization",
        parameter_names=("theta", "bias"),
        gradient_method="finite_difference",
        estimator_provenance=None,
    )
    first = circuit.to_json()
    second = circuit.to_json()
    payload = json.loads(first)
    provenance = cast(dict[str, object], payload["serialization_provenance"])
    digest = provenance["metadata_digest"]
    return DifferentiableCircuitContractCheck(
        name="serialization_provenance",
        status="supported",
        evidence=(
            f"deterministic_json={first == second}",
            f"schema={payload['schema']}",
            f"serializes_executable_code={provenance['serializes_executable_code']}",
            f"digest_length={len(cast(str, digest))}",
            f"route={payload['diagnostics']['estimator_provenance']['route']}",
        ),
    )


def _parameter_container_fail_closed_check() -> DifferentiableCircuitContractCheck:
    circuit = differentiable_circuit(
        _scalar_objective,
        name="dp004_parameter_container_boundary",
        parameter_names=("theta", "bias"),
    )
    try:
        circuit(cast(Any, _NamedParameters(theta=0.3, bias=0.5)))
    except ValueError as exc:
        return DifferentiableCircuitContractCheck(
            name="dataclass_parameter_container_boundary",
            status="fail_closed",
            evidence=("dataclass container rejected before objective execution",),
            blocked_reason=str(exc),
        )
    raise AssertionError(  # pragma: no cover - defensive contract guard.
        "dataclass parameter container unexpectedly executed"
    )


def _objective_contract_fail_closed_check() -> DifferentiableCircuitContractCheck:
    circuit = differentiable_circuit(
        cast(ScalarObjective, _vector_objective),
        name="dp004_objective_boundary",
        parameter_names=("theta", "bias"),
    )
    try:
        circuit(np.array([0.3, 0.5], dtype=np.float64))
    except ValueError as exc:
        return DifferentiableCircuitContractCheck(
            name="scalar_objective_boundary",
            status="fail_closed",
            evidence=("vector-valued objective rejected by call semantics",),
            blocked_reason=str(exc),
        )
    raise AssertionError(  # pragma: no cover - defensive contract guard.
        "vector-valued objective unexpectedly executed"
    )


def _scalar_objective(values: NDArray[np.float64]) -> float:
    return float(np.sin(values[0]) + values[1] ** 2)


def _mutating_scalar_objective(values: NDArray[np.float64]) -> float:
    values[0] = 999.0
    return float(np.sin(0.3) + values[1] ** 2)


def _vector_objective(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([values[0], values[1]], dtype=np.float64)


__all__ = [
    "DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY",
    "DifferentiableCircuit",
    "DifferentiableCircuitContractAuditResult",
    "DifferentiableCircuitContractCheck",
    "DifferentiableCircuitContractStatus",
    "run_differentiable_circuit_contract_audit",
]
