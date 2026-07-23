# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API contracts tests
# scpn-quantum-control -- unified differentiable API contract tests
"""Validation and facade-identity tests for unified differentiable API contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable_api import (
    DifferentiabilityDiagnosticReport as FacadeDifferentiabilityDiagnosticReport,
)
from scpn_quantum_control.differentiable_api import (
    DifferentiableDashboardCapabilityRow as FacadeDifferentiableDashboardCapabilityRow,
)
from scpn_quantum_control.differentiable_api import (
    DifferentiableDashboardCapabilityState as FacadeDifferentiableDashboardCapabilityState,
)
from scpn_quantum_control.differentiable_api import (
    DifferentiableDashboardStatus as FacadeDifferentiableDashboardStatus,
)
from scpn_quantum_control.differentiable_api import (
    UnifiedDifferentiableAPIResult as FacadeUnifiedDifferentiableAPIResult,
)
from scpn_quantum_control.differentiable_api import (
    UnifiedDifferentiableOperation as FacadeUnifiedDifferentiableOperation,
)
from scpn_quantum_control.differentiable_api_contracts import (
    DifferentiabilityDiagnosticReport,
    DifferentiableDashboardCapabilityRow,
    DifferentiableDashboardCapabilityState,
    DifferentiableDashboardStatus,
    FloatArray,
    UnifiedDifferentiableAPIResult,
    UnifiedDifferentiableOperation,
)


def _api_result() -> UnifiedDifferentiableAPIResult:
    """Build one valid public result envelope for contract tests."""
    return UnifiedDifferentiableAPIResult(
        operation="gradient",
        supported=True,
        method="parameter_shift",
        value=1.0,
        gradient=np.array([0.5], dtype=np.float64),
        jacobian=None,
        hessian=None,
        payload={"evaluations": 2},
    )


def _diagnostic_report() -> DifferentiabilityDiagnosticReport:
    """Build one valid blocked diagnostic report for contract tests."""
    return DifferentiabilityDiagnosticReport(
        request={"backend": "hardware"},
        supported=False,
        blocked_reasons=("hardware disabled",),
        suggested_alternatives=("statevector",),
        dependency_matrix=({"name": "qiskit"},),
        device_matrix=({"name": "cpu"},),
        backend_matrix=({"name": "hardware"},),
        support_payload={"supported": False},
        claim_boundary="no hardware execution",
    )


def test_differentiable_dashboard_status_validates_rows() -> None:
    """Dashboard status contracts must reject ambiguous row evidence."""
    row = DifferentiableDashboardCapabilityRow(
        surface="demo",
        state="unsupported",
        backing_api="demo_api",
        evidence=("contract",),
        blocked_reasons=("not implemented",),
        claim_boundary="bounded",
    )

    assert row.fail_closed is True
    assert row.to_dict()["state"] == "unsupported"
    with pytest.raises(ValueError, match="surface"):
        DifferentiableDashboardCapabilityRow(
            surface="",
            state="planned",
            backing_api="demo_api",
            evidence=("contract",),
            blocked_reasons=(),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="rows"):
        DifferentiableDashboardStatus(
            rows=(),
            status_api_ready=True,
            generated_from=("demo",),
            claim_boundary="bounded",
        )


def test_differentiable_dashboard_row_rejects_ambiguous_claim_states() -> None:
    """Runtime dashboard rows enforce state, evidence, and blocker coherence."""
    row = DifferentiableDashboardCapabilityRow(
        surface="demo",
        state="unsupported",
        backing_api="demo_api",
        evidence=("contract",),
        blocked_reasons=("not implemented",),
        claim_boundary="bounded",
    )

    with pytest.raises(ValueError, match="state is unknown"):
        replace(row, state=cast(DifferentiableDashboardCapabilityState, "available"))
    with pytest.raises(ValueError, match="evidence must be non-empty"):
        replace(row, evidence=())
    with pytest.raises(ValueError, match="require blocked reasons"):
        replace(row, blocked_reasons=())
    with pytest.raises(ValueError, match="cannot carry blocked reasons"):
        replace(row, state="executable")
    with pytest.raises(ValueError, match="surface must be non-empty"):
        replace(row, surface=cast(str, 0))
    with pytest.raises(ValueError, match="backing_api must be non-empty"):
        replace(row, backing_api=" ")
    with pytest.raises(ValueError, match="evidence entries must be non-empty"):
        replace(row, evidence=("",))
    with pytest.raises(ValueError, match="evidence entries must be non-empty"):
        replace(row, evidence=cast(tuple[str, ...], ["contract"]))
    with pytest.raises(ValueError, match="blocked reasons must be non-empty"):
        replace(row, blocked_reasons=(cast(str, 0),))
    with pytest.raises(ValueError, match="blocked reasons must be non-empty"):
        replace(row, blocked_reasons=cast(tuple[str, ...], ["blocked"]))
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        replace(row, claim_boundary=" ")


def test_differentiable_dashboard_status_rejects_ambiguous_sources_and_rows() -> None:
    """Public status payloads cannot contain duplicate surfaces or provenance."""
    row = DifferentiableDashboardCapabilityRow(
        surface="demo",
        state="unsupported",
        backing_api="demo_api",
        evidence=("contract",),
        blocked_reasons=("not implemented",),
        claim_boundary="bounded",
    )

    with pytest.raises(ValueError, match="unique surfaces"):
        DifferentiableDashboardStatus(
            rows=(row, row),
            status_api_ready=True,
            generated_from=("demo",),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="generated_from entries must be unique"):
        DifferentiableDashboardStatus(
            rows=(row,),
            status_api_ready=True,
            generated_from=("demo", "demo"),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="rows must be non-empty"):
        DifferentiableDashboardStatus(
            rows=cast(tuple[DifferentiableDashboardCapabilityRow, ...], [row]),
            status_api_ready=True,
            generated_from=("demo",),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="rows must contain"):
        DifferentiableDashboardStatus(
            rows=(cast(DifferentiableDashboardCapabilityRow, "bad"),),
            status_api_ready=True,
            generated_from=("demo",),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="status_api_ready must be boolean"):
        DifferentiableDashboardStatus(
            rows=(row,),
            status_api_ready=cast(bool, 1),
            generated_from=("demo",),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="generated_from entries must be non-empty"):
        DifferentiableDashboardStatus(
            rows=(row,),
            status_api_ready=True,
            generated_from=(),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="generated_from entries must be non-empty"):
        DifferentiableDashboardStatus(
            rows=(row,),
            status_api_ready=True,
            generated_from=(cast(str, 0),),
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        DifferentiableDashboardStatus(
            rows=(row,),
            status_api_ready=True,
            generated_from=("demo",),
            claim_boundary=" ",
        )

    assert DifferentiableDashboardStatus(
        rows=(row,),
        status_api_ready=True,
        generated_from=("demo",),
        claim_boundary="bounded",
    ).to_dict()["rows"] == [row.to_dict()]


def test_unified_result_validates_claim_envelope_and_serializes_arrays() -> None:
    """Public result envelopes reject runtime type and numerical ambiguity."""
    result = _api_result()
    payload = result.to_dict()

    assert payload["gradient"] == [0.5]
    assert payload["fail_closed"] is False
    with pytest.raises(ValueError, match="operation is unknown"):
        replace(result, operation=cast(UnifiedDifferentiableOperation, "unknown"))
    with pytest.raises(ValueError, match="supported must be boolean"):
        replace(result, supported=cast(bool, 1))
    with pytest.raises(ValueError, match="method must be non-empty"):
        replace(result, method=" ")
    with pytest.raises(ValueError, match="value must be a finite float"):
        replace(result, value=float("nan"))
    with pytest.raises(ValueError, match="gradient must be a finite float64 array"):
        replace(
            result,
            gradient=cast(FloatArray, np.array([0.5], dtype=np.float32)),
        )
    with pytest.raises(ValueError, match="payload must be a mapping"):
        replace(result, payload=cast(Mapping[str, object], []))
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        replace(result, claim_boundary="")


def test_diagnostic_report_validates_blocked_evidence_and_serializes() -> None:
    """Diagnostic reports preserve coherent blocked evidence at runtime."""
    report = _diagnostic_report()
    payload = report.to_dict()

    assert payload["fail_closed"] is True
    assert payload["blocked_reasons"] == ["hardware disabled"]
    with pytest.raises(ValueError, match="request must be a mapping"):
        replace(report, request=cast(Mapping[str, object], []))
    with pytest.raises(ValueError, match="supported must be boolean"):
        replace(report, supported=cast(bool, 0))
    with pytest.raises(ValueError, match="blocked_reasons must contain"):
        replace(report, blocked_reasons=cast(tuple[str, ...], ["blocked"]))
    with pytest.raises(ValueError, match="suggested_alternatives must contain"):
        replace(report, suggested_alternatives=("",))
    with pytest.raises(ValueError, match="supported diagnostics cannot carry"):
        replace(report, supported=True)
    with pytest.raises(ValueError, match="unsupported diagnostics require"):
        replace(report, blocked_reasons=())
    with pytest.raises(ValueError, match="device_matrix must contain mappings"):
        replace(report, device_matrix=cast(tuple[Mapping[str, object], ...], ("bad",)))
    with pytest.raises(ValueError, match="support_payload must be a mapping"):
        replace(report, support_payload=cast(Mapping[str, object], []))
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        replace(report, claim_boundary=" ")


def test_differentiable_api_contracts_are_exact_facade_aliases() -> None:
    """The facade should re-export the exact dependency-free contract objects."""
    assert FacadeUnifiedDifferentiableAPIResult is UnifiedDifferentiableAPIResult
    assert FacadeDifferentiabilityDiagnosticReport is DifferentiabilityDiagnosticReport
    assert FacadeDifferentiableDashboardCapabilityRow is DifferentiableDashboardCapabilityRow
    assert FacadeDifferentiableDashboardStatus is DifferentiableDashboardStatus
    assert FacadeUnifiedDifferentiableOperation is UnifiedDifferentiableOperation
    assert FacadeDifferentiableDashboardCapabilityState is DifferentiableDashboardCapabilityState
