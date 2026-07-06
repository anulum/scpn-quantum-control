# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Readiness Audit
"""Unified readiness ledger for differentiable-programming evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .gradient_support_matrix import run_gradient_support_matrix_audit
from .hardware_gradient_policy import run_hardware_gradient_policy_readiness_suite
from .provider_gradient_audit import run_provider_gradient_readiness_audit
from .provider_hardware_gradient_audit import run_provider_hardware_gradient_preparation_audit
from .qnode_provider_transforms import run_provider_qnode_transform_readiness_suite
from .qnode_tape import run_phase_qnode_tape_readiness_suite
from .qnode_transforms import run_phase_qnode_transform_readiness_suite
from .qnode_vector_transforms import run_phase_qnode_vector_transform_readiness_suite
from .transform_nesting import run_gradient_transform_nesting_audit

ReadinessRunner = Callable[[], object]


@dataclass(frozen=True)
class DifferentiableReadinessSurface:
    """One callable readiness surface in the differentiable audit ledger."""

    surface: str
    runner: ReadinessRunner
    description: str

    def __post_init__(self) -> None:
        """Validate readiness surface metadata after construction."""
        if not self.surface.strip():
            raise ValueError("surface must be non-empty")
        if not self.description.strip():
            raise ValueError("description must be non-empty")
        object.__setattr__(self, "surface", self.surface.strip())
        object.__setattr__(self, "description", self.description.strip())

    def to_dict(self) -> dict[str, str]:
        """Return JSON-ready surface metadata."""
        return {
            "surface": self.surface,
            "runner": getattr(self.runner, "__name__", type(self.runner).__name__),
            "description": self.description,
        }


@dataclass(frozen=True)
class DifferentiableReadinessAuditRecord:
    """Aggregated readiness record for one differentiable-programming surface."""

    surface: str
    description: str
    passed: bool
    supported_count: int
    blocked_count: int
    hardware_execution_count: int
    hardware_gradient_available_count: int
    claim_boundary: str
    blocked_boundaries: tuple[str, ...]
    payload: Mapping[str, Any]

    @property
    def failed(self) -> bool:
        """Whether the surface failed its readiness audit."""
        return not self.passed

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready aggregated record metadata."""
        return {
            "surface": self.surface,
            "description": self.description,
            "passed": self.passed,
            "failed": self.failed,
            "supported_count": self.supported_count,
            "blocked_count": self.blocked_count,
            "hardware_execution_count": self.hardware_execution_count,
            "hardware_gradient_available_count": self.hardware_gradient_available_count,
            "claim_boundary": self.claim_boundary,
            "blocked_boundaries": list(self.blocked_boundaries),
            "payload": _json_ready(self.payload),
        }


@dataclass(frozen=True)
class DifferentiableReadinessAuditResult:
    """Unified differentiable-programming readiness audit result."""

    records: tuple[DifferentiableReadinessAuditRecord, ...]
    claim_boundary: str

    @property
    def record_count(self) -> int:
        """Number of readiness records."""
        return len(self.records)

    @property
    def passed_count(self) -> int:
        """Number of readiness records whose own audit passed."""
        return sum(record.passed for record in self.records)

    @property
    def failed_count(self) -> int:
        """Number of readiness records whose own audit failed."""
        return sum(record.failed for record in self.records)

    @property
    def supported_count(self) -> int:
        """Total supported subroutes reported by focused readiness surfaces."""
        return sum(record.supported_count for record in self.records)

    @property
    def blocked_count(self) -> int:
        """Total blocked subroutes reported by focused readiness surfaces."""
        return sum(record.blocked_count for record in self.records)

    @property
    def hardware_execution_count(self) -> int:
        """Total hardware executions reported by readiness surfaces."""
        return sum(record.hardware_execution_count for record in self.records)

    @property
    def hardware_gradient_available_count(self) -> int:
        """Total hardware-gradient results reported by readiness surfaces."""
        return sum(record.hardware_gradient_available_count for record in self.records)

    @property
    def blocked_boundaries(self) -> tuple[str, ...]:
        """Unique fail-closed boundary reasons surfaced by focused audits."""
        seen: set[str] = set()
        boundaries: list[str] = []
        for record in self.records:
            for boundary in record.blocked_boundaries:
                if boundary and boundary not in seen:
                    seen.add(boundary)
                    boundaries.append(boundary)
        return tuple(boundaries)

    @property
    def passed(self) -> bool:
        """Whether every focused surface passed without live hardware gradients."""
        return (
            self.record_count == len(default_differentiable_readiness_surfaces())
            and self.failed_count == 0
            and self.hardware_execution_count == 0
            and self.hardware_gradient_available_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready unified readiness metadata."""
        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "supported_count": self.supported_count,
            "blocked_count": self.blocked_count,
            "hardware_execution_count": self.hardware_execution_count,
            "hardware_gradient_available_count": self.hardware_gradient_available_count,
            "blocked_boundaries": list(self.blocked_boundaries),
            "claim_boundary": self.claim_boundary,
            "records": [record.to_dict() for record in self.records],
        }


def default_differentiable_readiness_surfaces() -> tuple[DifferentiableReadinessSurface, ...]:
    """Return the default focused readiness surfaces for differentiable programming."""
    return (
        DifferentiableReadinessSurface(
            surface="gradient_support_matrix",
            runner=run_gradient_support_matrix_audit,
            description="registered gates, observables, backends, transforms, and adapters",
        ),
        DifferentiableReadinessSurface(
            surface="transform_nesting",
            runner=run_gradient_transform_nesting_audit,
            description="nested transform support and fail-closed transform algebra boundaries",
        ),
        DifferentiableReadinessSurface(
            surface="phase_qnode_tape",
            runner=run_phase_qnode_tape_readiness_suite,
            description="QNode-style tape recording and replay evidence",
        ),
        DifferentiableReadinessSurface(
            surface="phase_qnode_transforms",
            runner=run_phase_qnode_transform_readiness_suite,
            description="scalar local QNode transform execution evidence",
        ),
        DifferentiableReadinessSurface(
            surface="phase_qnode_vector_transforms",
            runner=run_phase_qnode_vector_transform_readiness_suite,
            description="native vector-output Jacobians and manual local vmap gradients",
        ),
        DifferentiableReadinessSurface(
            surface="provider_gradient_readiness",
            runner=run_provider_gradient_readiness_audit,
            description="provider callback parameter-shift readiness and blocked sample routes",
        ),
        DifferentiableReadinessSurface(
            surface="provider_qnode_transforms",
            runner=run_provider_qnode_transform_readiness_suite,
            description="provider callback QNode transforms and finite-shot uncertainty propagation",
        ),
        DifferentiableReadinessSurface(
            surface="hardware_gradient_policy",
            runner=run_hardware_gradient_policy_readiness_suite,
            description="hardware-gradient policy approval and fail-closed evidence gates",
        ),
        DifferentiableReadinessSurface(
            surface="provider_hardware_gradient_preparation",
            runner=run_provider_hardware_gradient_preparation_audit,
            description="provider hardware-gradient preparation audit without QPU execution",
        ),
    )


def run_differentiable_readiness_audit(
    surfaces: Sequence[DifferentiableReadinessSurface] | None = None,
) -> DifferentiableReadinessAuditResult:
    """Run the unified differentiable-programming readiness ledger."""
    active_surfaces = (
        tuple(surfaces) if surfaces is not None else default_differentiable_readiness_surfaces()
    )
    records = tuple(_run_surface(surface) for surface in active_surfaces)
    return DifferentiableReadinessAuditResult(
        records=records,
        claim_boundary=(
            "unified differentiable-programming readiness ledger; aggregates local, "
            "provider, transform, and hardware-preparation evidence without live QPU "
            "execution or hardware-gradient-result promotion"
        ),
    )


def _run_surface(surface: DifferentiableReadinessSurface) -> DifferentiableReadinessAuditRecord:
    raw_result = surface.runner()
    payload = _result_payload(raw_result)
    return DifferentiableReadinessAuditRecord(
        surface=surface.surface,
        description=surface.description,
        passed=_extract_passed(raw_result, payload),
        supported_count=_extract_supported_count(raw_result, payload),
        blocked_count=_extract_blocked_count(raw_result, payload),
        hardware_execution_count=_extract_hardware_execution_count(payload),
        hardware_gradient_available_count=_extract_hardware_gradient_available_count(payload),
        claim_boundary=_extract_claim_boundary(raw_result, payload),
        blocked_boundaries=_extract_blocked_boundaries(payload),
        payload=payload,
    )


def _result_payload(value: object) -> Mapping[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {"repr": repr(value)}


def _extract_passed(raw_result: object, payload: Mapping[str, Any]) -> bool:
    raw_passed = getattr(raw_result, "passed", payload.get("passed", False))
    return bool(raw_passed)


def _extract_supported_count(raw_result: object, payload: Mapping[str, Any]) -> int:
    for key in ("supported_count", "approved_count"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in ("supported_records", "approved_records"):
        value = getattr(raw_result, key, None)
        if isinstance(value, Sequence):
            return len(value)
    records = payload.get("records")
    if isinstance(records, list):
        return sum(
            1
            for record in records
            if _record_bool(record, "supported") or _record_bool(record, "approved")
        )
    return 0


def _extract_blocked_count(raw_result: object, payload: Mapping[str, Any]) -> int:
    for key in ("blocked_count", "fail_closed_count"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    blocked_records = getattr(raw_result, "blocked_records", None)
    if isinstance(blocked_records, Sequence):
        return len(blocked_records)
    records = payload.get("records")
    if isinstance(records, list):
        return sum(
            1
            for record in records
            if _record_bool(record, "blocked") or _record_bool(record, "fail_closed")
        )
    return 0


def _extract_hardware_execution_count(payload: Mapping[str, Any]) -> int:
    direct = payload.get("hardware_execution_count")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    records = payload.get("records")
    if not isinstance(records, list):
        return 0
    return sum(1 for record in records if _nested_bool(record, ("hardware_execution",)))


def _extract_hardware_gradient_available_count(payload: Mapping[str, Any]) -> int:
    direct = payload.get("hardware_gradient_available_count")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    direct = payload.get("gradient_available_count")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    records = payload.get("records")
    if not isinstance(records, list):
        return 0
    return sum(1 for record in records if _nested_bool(record, ("gradient_available",)))


def _extract_claim_boundary(raw_result: object, payload: Mapping[str, Any]) -> str:
    value = getattr(raw_result, "claim_boundary", payload.get("claim_boundary", ""))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "focused differentiable readiness surface"


def _extract_blocked_boundaries(payload: Mapping[str, Any]) -> tuple[str, ...]:
    records = payload.get("records")
    if not isinstance(records, list):
        return ()
    boundaries: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if not (
            _record_bool(record, "blocked")
            or _record_bool(record, "fail_closed")
            or not _record_bool(record, "supported")
        ):
            continue
        reason = _string_from_record(record, "failure_reason")
        if not reason:
            reason = _string_from_record(record, "claim_boundary")
        if reason:
            boundaries.append(reason)
    return tuple(boundaries)


def _record_bool(record: object, key: str) -> bool:
    if not isinstance(record, Mapping):
        return False
    value = record.get(key)
    return bool(value) if isinstance(value, bool) else False


def _nested_bool(record: object, path: tuple[str, ...]) -> bool:
    if not isinstance(record, Mapping):
        return False
    current: object = record
    for key in path:
        if not isinstance(current, Mapping):
            return False
        current = current.get(key)
    if isinstance(current, bool):
        return current
    result = record.get("result")
    if isinstance(result, Mapping):
        value = result.get(path[-1])
        return bool(value) if isinstance(value, bool) else False
    return False


def _string_from_record(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if isinstance(value, str):
        return value
    result = record.get("result")
    if isinstance(result, Mapping):
        nested = result.get(key)
        if isinstance(nested, str):
            return nested
    return ""


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    return repr(value)


__all__ = [
    "DifferentiableReadinessAuditRecord",
    "DifferentiableReadinessAuditResult",
    "DifferentiableReadinessSurface",
    "default_differentiable_readiness_surfaces",
    "run_differentiable_readiness_audit",
]
