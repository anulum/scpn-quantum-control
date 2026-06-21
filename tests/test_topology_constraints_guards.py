# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Guard and branch tests for topology constraints
"""Validation and projection-branch tests for the topology constraint ledger.

Covers the canonical-edge and bounds guards, the hardware-mask index guard, the
ledger policy validators, the matrix shape/finiteness checks, the fixed-sign
projection branch, the total-weight short-circuits and the frozen-edge violation
accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.topology_control.constraints import (
    CouplingGraphBounds,
    HardwareEmbeddingConstraint,
    TopologyConstraintLedger,
    canonical_edge,
)


def test_canonical_edge_rejects_self_edge() -> None:
    """A self edge is not a valid coupling edge."""
    with pytest.raises(ValueError, match="self edges are not valid"):
        canonical_edge(2, 2)


def test_bounds_require_upper_above_lower() -> None:
    """The upper bound must strictly exceed the lower bound."""
    with pytest.raises(ValueError, match="upper bound must exceed lower bound"):
        CouplingGraphBounds(lower=1.0, upper=0.5)


def test_hardware_mask_rejects_out_of_range_edge() -> None:
    """An edge index beyond the graph size is rejected by the mask builder."""
    constraint = HardwareEmbeddingConstraint.from_edges({(0, 5)})
    with pytest.raises(ValueError, match="hardware edge index exceeds graph size"):
        constraint.mask(3)


def test_ledger_rejects_invalid_sign_policy() -> None:
    """An unknown sign policy is rejected."""
    with pytest.raises(ValueError, match="invalid sign_policy"):
        TopologyConstraintLedger(sign_policy="bogus")  # type: ignore[arg-type]


def test_ledger_rejects_disordered_total_weight() -> None:
    """A negative or disordered total-weight interval is rejected."""
    with pytest.raises(ValueError, match="total_weight must be an ordered non-negative interval"):
        TopologyConstraintLedger(total_weight=(-1.0, 2.0))


def test_ledger_rejects_negative_connectivity_minimum() -> None:
    """A negative algebraic-connectivity minimum is rejected."""
    with pytest.raises(ValueError, match="algebraic_connectivity_min must be non-negative"):
        TopologyConstraintLedger(algebraic_connectivity_min=-1.0)


def test_project_rejects_non_square_matrix() -> None:
    """A non-square candidate matrix is rejected."""
    ledger = TopologyConstraintLedger()
    with pytest.raises(ValueError, match="coupling matrix must be square"):
        ledger.project(np.zeros((2, 3), dtype=np.float64))


def test_project_rejects_non_finite_matrix() -> None:
    """A non-finite candidate matrix is rejected."""
    ledger = TopologyConstraintLedger()
    matrix = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="coupling matrix must contain only finite values"):
        ledger.project(matrix)


def test_fixed_sign_policy_requires_reference() -> None:
    """The fixed-sign policy requires a sign reference matrix."""
    ledger = TopologyConstraintLedger(sign_policy="fixed_sign")
    with pytest.raises(ValueError, match="fixed_sign policy requires fixed_sign_reference"):
        ledger.project(np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64))


def test_fixed_sign_policy_imposes_reference_signs() -> None:
    """The fixed-sign policy copies the reference signs onto the magnitudes."""
    reference = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
    ledger = TopologyConstraintLedger(
        sign_policy="fixed_sign",
        bounds=CouplingGraphBounds(lower=-1.0, upper=1.0),
        fixed_sign_reference=reference,
    )
    projected = ledger.project(np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64))
    assert projected[0, 1] < 0.0
    assert projected[1, 0] < 0.0


def test_project_total_weight_returns_matrix_when_unbounded() -> None:
    """The total-weight projection is a no-op when no budget is configured."""
    ledger = TopologyConstraintLedger()
    matrix = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
    result = ledger._project_total_weight(matrix)
    np.testing.assert_array_equal(result, matrix)


def test_project_total_weight_short_circuits_without_adjustable_mass() -> None:
    """A budget overshoot carried entirely by frozen edges leaves the matrix intact."""
    ledger = TopologyConstraintLedger(
        total_weight=(0.0, 0.1),
        frozen_edges={(0, 1): 5.0},
    )
    matrix = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)
    result = ledger._project_total_weight(matrix)
    np.testing.assert_array_equal(result, matrix)


def test_violations_accumulate_frozen_edge_error() -> None:
    """Frozen-edge deviations contribute to the reported violation magnitudes."""
    ledger = TopologyConstraintLedger(frozen_edges={(0, 1): 0.5})
    matrix = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64)
    violation = ledger.violations(matrix)
    assert violation.frozen_edges > 0.0
