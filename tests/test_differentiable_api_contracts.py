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
    UnifiedDifferentiableAPIResult,
    UnifiedDifferentiableOperation,
)


def test_differentiable_dashboard_status_validates_rows() -> None:
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


def test_differentiable_api_contracts_are_exact_facade_aliases() -> None:
    """The facade should re-export the exact dependency-free contract objects."""

    assert FacadeUnifiedDifferentiableAPIResult is UnifiedDifferentiableAPIResult
    assert FacadeDifferentiabilityDiagnosticReport is DifferentiabilityDiagnosticReport
    assert FacadeDifferentiableDashboardCapabilityRow is DifferentiableDashboardCapabilityRow
    assert FacadeDifferentiableDashboardStatus is DifferentiableDashboardStatus
    assert FacadeUnifiedDifferentiableOperation is UnifiedDifferentiableOperation
    assert FacadeDifferentiableDashboardCapabilityState is DifferentiableDashboardCapabilityState
