# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera topology bridge tests
"""Real constraint-ledger tests for hierarchy coupling projection reports."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.chimera_control.schema import (
    HierarchyLevel,
    MultiscaleHierarchy,
    two_population_hierarchy,
)
from scpn_quantum_control.chimera_control.topology import (
    HierarchyCouplingSummary,
    TopologyProjectionReport,
    project_chimera_coupling,
)
from scpn_quantum_control.topology_control.constraints import (
    CouplingGraphBounds,
    TopologyConstraintLedger,
)


def test_projection_uses_existing_ledger_and_reports_multiscale_means() -> None:
    """Delegate projection to the ledger and report each hierarchy scale."""
    hierarchy = two_population_hierarchy(2)
    candidate = np.array(
        [
            [9.0, 1.2, -0.3, 0.4],
            [0.8, 7.0, 0.2, 0.6],
            [0.1, 0.2, 5.0, 1.4],
            [0.4, 0.6, 0.9, 3.0],
        ]
    )
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(0.0, 1.0),
        sign_policy="nonnegative",
        total_weight=(0.0, 6.0),
    )
    report = project_chimera_coupling(candidate, hierarchy, ledger)

    np.testing.assert_allclose(report.projected, report.projected.T)
    np.testing.assert_allclose(np.diag(report.projected), 0.0)
    assert np.min(report.projected) >= 0.0
    assert np.max(report.projected) <= 1.0
    assert report.violations_before.total > report.violations_after.total
    assert report.violations_after.total == pytest.approx(0.0)
    assert tuple(row.level_name for row in report.summaries_after) == (
        "population",
        "ensemble",
    )
    assert report.summaries_after[0].mean_between is not None
    assert report.summaries_after[1].mean_between is None
    assert not report.candidate.flags.writeable
    assert not report.projected.flags.writeable
    assert (
        report.content_digest
        == project_chimera_coupling(candidate, hierarchy, ledger).content_digest
    )


def test_projection_rejects_wrong_shape_and_non_finite_candidate() -> None:
    """Reject coupling candidates with the wrong shape or non-finite entries."""
    hierarchy = two_population_hierarchy(2)
    ledger = TopologyConstraintLedger()
    with pytest.raises(ValueError, match="shape"):
        project_chimera_coupling(np.zeros((3, 3)), hierarchy, ledger)
    candidate = np.zeros((4, 4))
    candidate[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        project_chimera_coupling(candidate, hierarchy, ledger)


def test_coupling_summary_contract_validates_fields() -> None:
    """Validate named within- and between-community coupling summaries."""
    assert HierarchyCouplingSummary("singletons", None, 0.2).mean_within is None
    with pytest.raises(ValueError, match="level_name"):
        HierarchyCouplingSummary(" ", 0.1, 0.2)
    with pytest.raises(ValueError, match="mean_within"):
        HierarchyCouplingSummary("fine", np.nan, 0.2)
    with pytest.raises(ValueError, match="mean_between"):
        HierarchyCouplingSummary("fine", 0.1, np.inf)


def test_projection_report_contract_rejects_inconsistent_custody() -> None:
    """Reject projection reports with inconsistent arrays, levels, or digests."""
    hierarchy = two_population_hierarchy(2)
    valid = project_chimera_coupling(np.ones((4, 4)), hierarchy, TopologyConstraintLedger())
    values: dict[str, object] = {
        "candidate": valid.candidate,
        "projected": valid.projected,
        "violations_before": valid.violations_before,
        "violations_after": valid.violations_after,
        "summaries_before": valid.summaries_before,
        "summaries_after": valid.summaries_after,
        "content_digest": valid.content_digest,
    }
    with pytest.raises(ValueError, match="equal non-empty"):
        TopologyProjectionReport(**(values | {"projected": np.zeros((2, 2))}))
    bad = np.array(valid.candidate, copy=True)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        TopologyProjectionReport(**(values | {"candidate": bad}))
    with pytest.raises(ValueError, match="identical levels"):
        TopologyProjectionReport(**(values | {"summaries_after": tuple()}))
    with pytest.raises(ValueError, match="non-empty"):
        TopologyProjectionReport(
            **(values | {"summaries_before": tuple(), "summaries_after": tuple()})
        )
    with pytest.raises(ValueError, match="content_digest"):
        TopologyProjectionReport(**(values | {"content_digest": "bad"}))
    with pytest.raises(ValueError, match="claim_boundary"):
        TopologyProjectionReport(**(values | {"claim_boundary": " "}))


def test_projection_supports_a_valid_singleton_fine_level() -> None:
    """Represent singleton fine scales without inventing within-node edges."""
    hierarchy = MultiscaleHierarchy(
        3,
        (
            HierarchyLevel("node", ((0,), (1,), (2,))),
            HierarchyLevel("ensemble", ((0, 1, 2),)),
        ),
    )
    report = project_chimera_coupling(np.ones((3, 3)), hierarchy, TopologyConstraintLedger())
    assert report.summaries_after[0].mean_within is None
    assert report.summaries_after[0].mean_between == pytest.approx(1.0)
