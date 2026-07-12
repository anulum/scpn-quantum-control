# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — symmetry sector compiler tests
# scpn-quantum-control -- symmetry-sector mitigation compiler tests
"""Tests for the sector-aware mitigation planning contract."""

from __future__ import annotations

from scpn_quantum_control.mitigation import (
    SymmetrySectorProblem,
    plan_symmetry_sector_mitigation,
)


def _problem(**overrides) -> SymmetrySectorProblem:
    kwargs = {
        "n_qubits": 4,
        "coupling_matrix": (
            (0.0, 0.45, 0.0, 0.45),
            (0.45, 0.0, 0.45, 0.0),
            (0.0, 0.45, 0.0, 0.45),
            (0.45, 0.0, 0.45, 0.0),
        ),
        "omega": (0.8, 0.9333333333, 1.0666666667, 1.2),
        "initial_state": "0011",
        "measurement_basis": "counts",
        "has_raw_counts": True,
        "has_noise_scaled_symmetry_observables": True,
    }
    kwargs.update(overrides)
    return SymmetrySectorProblem(**kwargs)


def test_planner_returns_full_plan_when_evidence_is_present() -> None:
    """Valid XY/counts inputs enable parity, expansion, and GUESS primitives."""

    plan = plan_symmetry_sector_mitigation(_problem())

    assert plan.status == "eligible"
    assert plan.expected_parity == 0
    assert plan.primitives == (
        "parity_postselection",
        "symmetry_expansion",
        "guess_symmetry_decay",
    )
    assert not plan.blockers


def test_planner_blocks_without_raw_counts() -> None:
    """Mitigation planning is not allowed without raw measurement counts."""

    plan = plan_symmetry_sector_mitigation(_problem(has_raw_counts=False))

    assert plan.status == "blocked"
    assert plan.primitives == ()
    assert "raw measurement counts are required before mitigation planning" in plan.blockers


def test_planner_blocks_guess_without_noise_scaled_observables() -> None:
    """GUESS is not silently enabled without its calibration evidence."""

    plan = plan_symmetry_sector_mitigation(_problem(has_noise_scaled_symmetry_observables=False))

    assert plan.status == "blocked"
    assert "GUESS requires noise-scaled symmetry observables" in plan.blockers


def test_planner_blocks_nonsymmetric_coupling() -> None:
    """Sector planning requires a symmetric XY-style coupling matrix."""

    plan = plan_symmetry_sector_mitigation(
        _problem(
            coupling_matrix=(
                (0.0, 0.45, 0.0, 0.0),
                (0.0, 0.0, 0.45, 0.0),
                (0.0, 0.45, 0.0, 0.45),
                (0.45, 0.0, 0.45, 0.0),
            )
        )
    )

    assert plan.status == "blocked"
    assert "coupling_matrix must be symmetric for XY parity-sector planning" in plan.blockers
