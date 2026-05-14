# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 opening foundation fixtures
"""Tests for Paper 0 opening foundation and global-boundary axiom fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.opening_foundation_validation import (
    OpeningFoundationConfig,
    beta0_boundary_assertion,
    boundary_set_c0,
    terminal_set_f0,
    validate_opening_foundation_fixture,
)


def test_boundary_and_terminal_sets_preserve_source_axiom_members() -> None:
    assert boundary_set_c0() == ("G_local", "F_env", "G_cosmic", "O_state")
    assert terminal_set_f0() == ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def test_beta0_rejects_free_untracked_boundary_conditions() -> None:
    assert beta0_boundary_assertion(
        boundary_members=("G_local", "F_env", "G_cosmic", "O_state"),
        active_terminals=("T1", "T4"),
    )

    with pytest.raises(ValueError, match="unknown boundary members"):
        beta0_boundary_assertion(
            boundary_members=("G_local", "free_boundary"), active_terminals=("T1",)
        )
    with pytest.raises(ValueError, match="unknown terminal ids"):
        beta0_boundary_assertion(boundary_members=("G_local",), active_terminals=("T8",))
    with pytest.raises(ValueError, match="at least one active terminal"):
        beta0_boundary_assertion(boundary_members=("G_local",), active_terminals=())


def test_opening_foundation_fixture_preserves_claim_boundary() -> None:
    result = validate_opening_foundation_fixture()

    assert result.hardware_status == "source_foundation_no_experiment"
    assert result.source_ledger_span == ("P0R00001", "P0R00017")
    assert result.boundary_set_size == 4
    assert result.terminal_count == 7
    assert result.spec_count == 5
    assert result.null_controls["free_boundary_rejection_label"] == 1.0
    assert result.null_controls["unknown_terminal_rejection_label"] == 1.0
    assert result.null_controls["empty_terminal_subset_rejection_label"] == 1.0
    assert "not empirical validation evidence" in result.claim_boundary

    with pytest.raises(ValueError, match="expected_terminal_count must be at least 1"):
        OpeningFoundationConfig(expected_terminal_count=0)
