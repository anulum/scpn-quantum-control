# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- symmetry-sector replay tests
"""Tests for the raw-count symmetry-sector replay adapter."""

from __future__ import annotations

import pytest

from scpn_quantum_control.mitigation import (
    SymmetrySectorProblem,
    replay_symmetry_sector_counts,
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


def test_replay_applies_raw_count_primitives_and_defers_guess() -> None:
    """Replay preserves shot accounting and reports non-count GUESS as deferred."""

    result = replay_symmetry_sector_counts(
        _problem(),
        {"0011": 40, "0000": 10, "0001": 5, "1110": 7},
    )

    assert result.status == "applied"
    assert result.raw_shots == 62
    assert result.postselected_counts == {"0011": 40, "0000": 10}
    assert result.rejected_counts == {"0001": 5, "1110": 7}
    assert result.postselected_shots == 50
    assert result.expanded_shots == 62
    assert result.expanded_counts["0000"] == 15
    assert result.expanded_counts["1111"] == 7
    assert result.applied_primitives == ("parity_postselection", "symmetry_expansion")
    assert result.deferred_primitives == ("guess_symmetry_decay",)
    assert result.blockers == (
        "GUESS replay requires calibrated noise-scaled symmetry observable rows",
    )


def test_replay_rejects_blocked_planner_output() -> None:
    """Replay does not run when planner evidence is incomplete."""

    with pytest.raises(ValueError, match="raw measurement counts"):
        replay_symmetry_sector_counts(_problem(has_raw_counts=False), {"0011": 10})


def test_replay_rejects_invalid_count_keys_and_values() -> None:
    """Replay validates count shape before invoking mitigation primitives."""

    with pytest.raises(ValueError, match="4-bit computational-basis"):
        replay_symmetry_sector_counts(_problem(), {"011": 10})

    with pytest.raises(ValueError, match="non-negative integer"):
        replay_symmetry_sector_counts(_problem(), {"0011": -1})

    with pytest.raises(ValueError, match="at least one shot"):
        replay_symmetry_sector_counts(_problem(), {"0011": 0})


def test_replay_merges_equivalent_spaced_bitstrings() -> None:
    """Counts are normalised before postselection and expansion."""

    result = replay_symmetry_sector_counts(_problem(), {"00 11": 3, "0011": 4})

    assert result.raw_counts == {"0011": 7}
    assert result.postselected_counts == {"0011": 7}
