# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-control topology differential tests
"""Fixed-active-set JVP/VJP and fail-closed branch tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.dla_topology_control.projection import (
    TopologyProjectionDifferential,
    topology_projection_jvp,
    topology_projection_support,
    topology_projection_vjp,
)
from scpn_quantum_control.dla_topology_control.schema import (
    DifferentiabilityKind,
    UnsupportedDifferentiableConstraintError,
)
from scpn_quantum_control.topology_control.constraints import (
    CouplingGraphBounds,
    TopologyConstraintLedger,
)


def _supported_ledger() -> TopologyConstraintLedger:
    return TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="signed",
        hardware_edges={(0, 1), (1, 2), (2, 3), (0, 3)},
        frozen_edges={(0, 1): 0.25},
    )


def _primal() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.4, -0.6, 0.7],
            [0.2, 0.0, 0.5, -0.4],
            [-0.3, 0.8, 0.0, 0.6],
            [0.9, -0.7, 0.2, 0.0],
        ]
    )


def test_supported_projection_jvp_matches_central_difference_and_custody() -> None:
    """Differentiate signed bounds, mask, and frozen edges on one fixed branch."""
    rng = np.random.default_rng(541)
    ledger = _supported_ledger()
    primal = _primal()
    tangent = rng.normal(size=(4, 4))
    differential = topology_projection_jvp(ledger, primal, tangent)
    epsilon = 1.0e-6
    central = (
        ledger.project(primal + epsilon * tangent) - ledger.project(primal - epsilon * tangent)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(differential.projected_tangent, central, atol=1.0e-9)
    np.testing.assert_array_equal(differential.projected, ledger.project(primal))
    assert differential.support.derivative_supported
    assert len(differential.content_digest) == 64
    assert all(
        not value.flags.writeable
        for value in (
            differential.matrix,
            differential.tangent,
            differential.projected,
            differential.projected_tangent,
        )
    )


def test_supported_projection_vjp_satisfies_adjoint_identity() -> None:
    """Match the local JVP and VJP under the Frobenius inner product."""
    rng = np.random.default_rng(542)
    ledger = _supported_ledger()
    primal = _primal()
    tangent = rng.normal(size=(4, 4))
    cotangent = rng.normal(size=(4, 4))
    jvp = topology_projection_jvp(ledger, primal, tangent).projected_tangent
    vjp = topology_projection_vjp(ledger, primal, cotangent)
    assert float(np.vdot(jvp, cotangent).real) == pytest.approx(
        float(np.vdot(tangent, vjp).real), abs=1.0e-12
    )
    assert not vjp.flags.writeable


def test_nonnegative_policy_supports_fixed_positive_and_negative_branches() -> None:
    """Support nonnegative maximum away from zero and use zero derivative below it."""
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-1.0, 2.0), sign_policy="nonnegative"
    )
    primal = np.array([[0.0, -0.5], [-0.7, 0.0]])
    tangent = np.ones((2, 2))
    report = topology_projection_support(ledger, primal)
    assert report.derivative_supported
    differential = topology_projection_jvp(ledger, primal, tangent)
    np.testing.assert_array_equal(differential.projected_tangent, np.zeros((2, 2)))


def test_nonnegative_zero_kink_fails_closed() -> None:
    """Refuse a derivative at the exact nonnegative and lower-bound kink."""
    ledger = TopologyConstraintLedger()
    report = topology_projection_support(ledger, np.zeros((3, 3)))
    assert not report.derivative_supported
    assert "sign_policy" in report.blocking_capabilities
    assert "uniform_bounds" in report.blocking_capabilities
    with pytest.raises(UnsupportedDifferentiableConstraintError, match="sign_policy"):
        topology_projection_jvp(ledger, np.zeros((3, 3)), np.ones((3, 3)))


def test_fixed_sign_branch_matches_central_difference() -> None:
    """Differentiate a non-zero fixed-sign absolute-value branch exactly."""
    reference = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]])
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="fixed_sign",
        fixed_sign_reference=reference,
    )
    primal = np.array([[0.0, -0.4, 0.6], [-0.2, 0.0, -0.5], [0.8, -0.7, 0.0]])
    tangent = np.arange(9, dtype=np.float64).reshape(3, 3) / 10.0
    differential = topology_projection_jvp(ledger, primal, tangent)
    epsilon = 1.0e-6
    central = (
        ledger.project(primal + epsilon * tangent) - ledger.project(primal - epsilon * tangent)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(differential.projected_tangent, central, atol=1.0e-9)


def test_fixed_sign_missing_mismatched_and_zero_kink_contracts() -> None:
    """Reject missing/mismatched references and refuse absolute-value kinks."""
    missing = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0), sign_policy="fixed_sign"
    )
    with pytest.raises(ValueError, match="requires fixed_sign_reference"):
        topology_projection_support(missing, np.ones((2, 2)))
    mismatch = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="fixed_sign",
        fixed_sign_reference=np.ones((3, 3)),
    )
    with pytest.raises(ValueError, match="shape"):
        topology_projection_support(mismatch, np.ones((2, 2)))
    kink = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="fixed_sign",
        fixed_sign_reference=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    report = topology_projection_support(kink, np.zeros((2, 2)))
    assert report.blocking_capabilities == ("sign_policy",)


def test_clip_active_branch_is_supported_away_from_bounds() -> None:
    """Use a zero tangent on a fixed active clipping branch away from its kink."""
    ledger = TopologyConstraintLedger(bounds=CouplingGraphBounds(-0.5, 0.5), sign_policy="signed")
    primal = np.array([[0.0, 0.9], [1.1, 0.0]])
    differential = topology_projection_jvp(ledger, primal, np.ones((2, 2)))
    np.testing.assert_array_equal(differential.projected_tangent, np.zeros((2, 2)))


def test_clip_boundary_fails_closed() -> None:
    """Refuse derivatives when an off-diagonal value lies at a clip boundary."""
    ledger = TopologyConstraintLedger(bounds=CouplingGraphBounds(-0.5, 0.5), sign_policy="signed")
    primal = np.array([[0.0, 0.5], [0.5, 0.0]])
    report = topology_projection_support(ledger, primal)
    assert report.blocking_capabilities == ("uniform_bounds",)


def test_total_weight_is_supported_only_strictly_inside_interval() -> None:
    """Support inactive budgets and reject active or boundary rescaling branches."""
    inactive = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="signed",
        total_weight=(1.0, 3.0),
    )
    primal = np.array([[0.0, 0.5], [0.5, 0.0]]) * 2.0
    assert topology_projection_support(inactive, primal).derivative_supported
    active = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="signed",
        total_weight=(3.0, 4.0),
    )
    assert "total_weight" in topology_projection_support(active, primal).blocking_capabilities
    boundary = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="signed",
        total_weight=(2.0, 4.0),
    )
    assert "total_weight" in topology_projection_support(boundary, primal).blocking_capabilities


def test_connectivity_threshold_fails_closed() -> None:
    """Refuse to relabel connectivity violation checks as projected derivatives."""
    ledger = TopologyConstraintLedger(
        bounds=CouplingGraphBounds(-2.0, 2.0),
        sign_policy="signed",
        algebraic_connectivity_min=0.2,
    )
    report = topology_projection_support(ledger, _primal())
    row = next(row for row in report.rows if row.capability == "algebraic_connectivity_threshold")
    assert row.status == "unsupported"
    assert row.differentiability is DifferentiabilityKind.NON_SMOOTH


@pytest.mark.parametrize("margin", [0.0, -1.0, np.nan])
def test_support_rejects_invalid_margin(margin: float) -> None:
    """Require a finite positive active-set margin."""
    with pytest.raises(ValueError, match="margin"):
        topology_projection_support(_supported_ledger(), _primal(), margin=margin)


def test_support_rejects_invalid_ledger_and_matrix() -> None:
    """Reject wrong ledger types and malformed primal arrays."""
    with pytest.raises(ValueError, match="ledger"):
        topology_projection_support(cast(TopologyConstraintLedger, object()), _primal())
    for matrix, message in (
        (np.ones(3), "square"),
        (np.ones((2, 3)), "square"),
        (np.ones((1, 1)), "at least two"),
        (np.array([[0.0, np.nan], [0.0, 0.0]]), "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            topology_projection_support(_supported_ledger(), matrix)


def test_jvp_vjp_reject_shape_mismatch() -> None:
    """Require tangent and cotangent shapes to match the primal matrix."""
    with pytest.raises(ValueError, match="tangent must match"):
        topology_projection_jvp(_supported_ledger(), _primal(), np.ones((3, 3)))
    with pytest.raises(ValueError, match="cotangent must match"):
        topology_projection_vjp(_supported_ledger(), _primal(), np.ones((3, 3)))


def test_differential_contract_rejects_invalid_arrays_digest_and_boundary() -> None:
    """Reject inconsistent topology differential custody fields."""
    valid = topology_projection_jvp(_supported_ledger(), _primal(), np.ones((4, 4)))
    values: dict[str, object] = {
        "matrix": valid.matrix,
        "tangent": valid.tangent,
        "projected": valid.projected,
        "projected_tangent": valid.projected_tangent,
        "support": valid.support,
        "content_digest": valid.content_digest,
        "claim_boundary": valid.claim_boundary,
    }
    with pytest.raises(ValueError, match="equal shape"):
        TopologyProjectionDifferential(**(values | {"tangent": np.ones((3, 3))}))
    with pytest.raises(ValueError, match="content_digest"):
        TopologyProjectionDifferential(**(values | {"content_digest": "bad"}))
    with pytest.raises(ValueError, match="claim_boundary"):
        TopologyProjectionDifferential(**(values | {"claim_boundary": " "}))
