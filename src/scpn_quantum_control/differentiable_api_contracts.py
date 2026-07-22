# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API contracts module
# scpn-quantum-control -- unified differentiable API contracts
"""Immutable envelopes and type contracts for the unified differentiable API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
UnifiedDifferentiableOperation = Literal[
    "value",
    "gradient",
    "jacobian",
    "hessian",
    "support_report",
    "diagnostic_report",
    "compile_report",
    "frontend_report",
    "benchmark_report",
    "dashboard_status",
    "baseline_scorecard",
    "competitive_baseline_refresh",
    "rust_python_inventory",
    "architecture_rustification_map",
    "dependency_environment_map",
    "isolated_benchmark_plan",
    "transform_algebra_report",
    "qfi_fss_report",
]
_UNIFIED_DIFFERENTIABLE_OPERATIONS = frozenset(
    {
        "value",
        "gradient",
        "jacobian",
        "hessian",
        "support_report",
        "diagnostic_report",
        "compile_report",
        "frontend_report",
        "benchmark_report",
        "dashboard_status",
        "baseline_scorecard",
        "competitive_baseline_refresh",
        "rust_python_inventory",
        "architecture_rustification_map",
        "dependency_environment_map",
        "isolated_benchmark_plan",
        "transform_algebra_report",
        "qfi_fss_report",
    }
)
DifferentiableDashboardCapabilityState = Literal[
    "planned",
    "metadata_only",
    "diagnostic",
    "conformance_backed",
    "executable",
    "blocked",
    "unsupported",
]
_DASHBOARD_CAPABILITY_STATES = frozenset(
    {
        "planned",
        "metadata_only",
        "diagnostic",
        "conformance_backed",
        "executable",
        "blocked",
        "unsupported",
    }
)

CLAIM_BOUNDARY = (
    "unified differentiable API facade over already-supported local routes; "
    "finite-difference paths remain diagnostic, support and compile reports "
    "are fail-closed, and no hardware execution or performance claim is implied"
)


@dataclass(frozen=True)
class UnifiedDifferentiableAPIResult:
    """Stable JSON evidence envelope returned by the unified facade."""

    operation: UnifiedDifferentiableOperation
    supported: bool
    method: str
    value: float | None
    gradient: FloatArray | None
    jacobian: FloatArray | None
    hessian: FloatArray | None
    payload: Mapping[str, Any]
    claim_boundary: str = CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Reject malformed claim envelopes before they reach public consumers."""
        if (
            not isinstance(self.operation, str)
            or self.operation not in _UNIFIED_DIFFERENTIABLE_OPERATIONS
        ):
            raise ValueError("unified differentiable operation is unknown")
        if type(self.supported) is not bool:
            raise ValueError("unified differentiable supported must be boolean")
        _require_nonblank(self.method, "unified differentiable method")
        if self.value is not None and (
            not isinstance(self.value, float) or not np.isfinite(self.value)
        ):
            raise ValueError("unified differentiable value must be a finite float or None")
        for field_name, array in (
            ("gradient", self.gradient),
            ("jacobian", self.jacobian),
            ("hessian", self.hessian),
        ):
            if array is not None and (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float64
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    f"unified differentiable {field_name} must be a finite float64 array or None"
                )
        if not isinstance(self.payload, Mapping):
            raise ValueError("unified differentiable payload must be a mapping")
        _require_nonblank(self.claim_boundary, "unified differentiable claim_boundary")

    @property
    def fail_closed(self) -> bool:
        """Return true when the requested operation is intentionally unsupported."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready unified differentiable evidence."""
        return {
            "operation": self.operation,
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "method": self.method,
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "hessian": None if self.hessian is None else self.hessian.tolist(),
            "payload": dict(self.payload),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiabilityDiagnosticReport:
    """JSON-ready explanation for a differentiability support decision."""

    request: Mapping[str, object]
    supported: bool
    blocked_reasons: tuple[str, ...]
    suggested_alternatives: tuple[str, ...]
    dependency_matrix: tuple[Mapping[str, object], ...]
    device_matrix: tuple[Mapping[str, object], ...]
    backend_matrix: tuple[Mapping[str, object], ...]
    support_payload: Mapping[str, object]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate diagnostic evidence and supported/blocked coherence."""
        if not isinstance(self.request, Mapping):
            raise ValueError("differentiability diagnostic request must be a mapping")
        if type(self.supported) is not bool:
            raise ValueError("differentiability diagnostic supported must be boolean")
        _require_string_tuple(
            self.blocked_reasons,
            "differentiability diagnostic blocked_reasons",
        )
        _require_string_tuple(
            self.suggested_alternatives,
            "differentiability diagnostic suggested_alternatives",
        )
        if self.supported and self.blocked_reasons:
            raise ValueError("supported diagnostics cannot carry blocked reasons")
        if not self.supported and not self.blocked_reasons:
            raise ValueError("unsupported diagnostics require blocked reasons")
        for field_name, rows in (
            ("dependency_matrix", self.dependency_matrix),
            ("device_matrix", self.device_matrix),
            ("backend_matrix", self.backend_matrix),
        ):
            if not isinstance(rows, tuple) or any(not isinstance(row, Mapping) for row in rows):
                raise ValueError(
                    f"differentiability diagnostic {field_name} must contain mappings"
                )
        if not isinstance(self.support_payload, Mapping):
            raise ValueError("differentiability diagnostic support_payload must be a mapping")
        _require_nonblank(
            self.claim_boundary,
            "differentiability diagnostic claim_boundary",
        )

    @property
    def fail_closed(self) -> bool:
        """Return true when the requested route is intentionally blocked."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready differentiability diagnostics."""
        return {
            "request": dict(self.request),
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "blocked_reasons": list(self.blocked_reasons),
            "suggested_alternatives": list(self.suggested_alternatives),
            "dependency_matrix": [dict(row) for row in self.dependency_matrix],
            "device_matrix": [dict(row) for row in self.device_matrix],
            "backend_matrix": [dict(row) for row in self.backend_matrix],
            "support_payload": dict(self.support_payload),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableDashboardCapabilityRow:
    """One claim-bounded row for differentiable dashboard consumers."""

    surface: str
    state: DifferentiableDashboardCapabilityState
    backing_api: str
    evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate dashboard row identifiers and claim-boundary metadata."""
        if not isinstance(self.surface, str) or not self.surface.strip():
            raise ValueError("dashboard status surface must be non-empty")
        if not isinstance(self.state, str) or self.state not in _DASHBOARD_CAPABILITY_STATES:
            raise ValueError("dashboard status state is unknown")
        if not isinstance(self.backing_api, str) or not self.backing_api.strip():
            raise ValueError("dashboard status backing_api must be non-empty")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.evidence
        ):
            raise ValueError("dashboard status evidence entries must be non-empty")
        if not self.evidence:
            raise ValueError("dashboard status evidence must be non-empty")
        if not isinstance(self.blocked_reasons, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.blocked_reasons
        ):
            raise ValueError("dashboard status blocked reasons must be non-empty")
        if self.fail_closed and not self.blocked_reasons:
            raise ValueError("fail-closed dashboard states require blocked reasons")
        if not self.fail_closed and self.blocked_reasons:
            raise ValueError("executable dashboard states cannot carry blocked reasons")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("dashboard status claim_boundary must be non-empty")

    @property
    def fail_closed(self) -> bool:
        """Return true when the dashboard row is intentionally non-executable."""
        return self.state in {"planned", "metadata_only", "diagnostic", "blocked", "unsupported"}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dashboard row."""
        return {
            "surface": self.surface,
            "state": self.state,
            "backing_api": self.backing_api,
            "evidence": list(self.evidence),
            "blocked_reasons": list(self.blocked_reasons),
            "fail_closed": self.fail_closed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class DifferentiableDashboardStatus:
    """Machine-readable differentiable status for GUI/audit-dashboard layers."""

    rows: tuple[DifferentiableDashboardCapabilityRow, ...]
    status_api_ready: bool
    generated_from: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate dashboard status row collection and metadata."""
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("dashboard status rows must be non-empty")
        if any(not isinstance(row, DifferentiableDashboardCapabilityRow) for row in self.rows):
            raise ValueError("dashboard status rows must contain dashboard row entries")
        surfaces = tuple(row.surface for row in self.rows)
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("dashboard status rows must have unique surfaces")
        if type(self.status_api_ready) is not bool:
            raise ValueError("dashboard status status_api_ready must be boolean")
        if (
            not isinstance(self.generated_from, tuple)
            or not self.generated_from
            or any(not isinstance(item, str) or not item.strip() for item in self.generated_from)
        ):
            raise ValueError("dashboard status generated_from entries must be non-empty")
        if len(set(self.generated_from)) != len(self.generated_from):
            raise ValueError("dashboard status generated_from entries must be unique")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("dashboard status claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dashboard status payload."""
        return {
            "status_api_ready": self.status_api_ready,
            "generated_from": list(self.generated_from),
            "rows": [row.to_dict() for row in self.rows],
            "claim_boundary": self.claim_boundary,
        }


def _require_nonblank(value: object, field_name: str) -> None:
    """Require an exact non-blank string for a public contract field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_string_tuple(value: object, field_name: str) -> None:
    """Require an exact tuple containing only non-blank strings."""
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must contain non-empty strings")


__all__ = [
    "CLAIM_BOUNDARY",
    "DifferentiabilityDiagnosticReport",
    "DifferentiableDashboardCapabilityRow",
    "DifferentiableDashboardCapabilityState",
    "DifferentiableDashboardStatus",
    "FloatArray",
    "UnifiedDifferentiableAPIResult",
    "UnifiedDifferentiableOperation",
]
