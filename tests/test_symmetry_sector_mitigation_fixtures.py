# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — symmetry sector mitigation fixtures tests
# scpn-quantum-control -- symmetry-sector mitigation fixture tests
"""Tests for deterministic symmetry-sector mitigation planner fixtures."""

from __future__ import annotations

import json

from scpn_quantum_control.mitigation.symmetry_sector_fixtures import (
    fixture_markdown,
    fixture_payload,
    normalised_json,
)


def test_fixture_payload_locks_expected_planner_outcomes() -> None:
    """Fixture payload covers eligible and fail-closed planner branches."""

    payload = fixture_payload()
    fixtures = {row["fixture_id"]: row for row in payload["fixtures"]}

    assert payload["schema"] == "symmetry_sector_mitigation_fixture_v1"
    assert payload["hardware_submission"] is False
    assert set(fixtures) == {
        "eligible_counts_guess",
        "blocked_missing_counts",
        "blocked_missing_guess_observables",
        "blocked_nonsymmetric_coupling",
    }
    assert fixtures["eligible_counts_guess"]["plan"]["status"] == "eligible"
    assert fixtures["eligible_counts_guess"]["plan"]["primitives"] == [
        "parity_postselection",
        "symmetry_expansion",
        "guess_symmetry_decay",
    ]
    assert fixtures["blocked_missing_counts"]["plan"]["status"] == "blocked"
    assert fixtures["blocked_missing_counts"]["plan"]["primitives"] == []
    assert fixtures["blocked_missing_guess_observables"]["plan"]["status"] == "blocked"
    assert fixtures["blocked_nonsymmetric_coupling"]["plan"]["status"] == "blocked"


def test_fixture_payload_locks_expected_replay_outcomes() -> None:
    """Fixture payload covers applied and fail-closed replay branches."""

    payload = fixture_payload()
    fixtures = {row["fixture_id"]: row for row in payload["replay_fixtures"]}

    assert set(fixtures) == {
        "replay_counts_postselection_expansion",
        "replay_blocked_missing_counts",
    }
    applied = fixtures["replay_counts_postselection_expansion"]["result"]
    assert applied["raw_shots"] == 62
    assert applied["postselected_counts"] == {"0000": 10, "0011": 40}
    assert applied["expanded_shots"] == 62
    assert applied["applied_primitives"] == [
        "parity_postselection",
        "symmetry_expansion",
    ]
    assert applied["deferred_primitives"] == ["guess_symmetry_decay"]
    assert fixtures["replay_blocked_missing_counts"]["status"] == "blocked"
    assert "raw measurement counts" in fixtures["replay_blocked_missing_counts"]["blockers"][0]


def test_normalised_json_is_stable_and_round_trips() -> None:
    """Fixture JSON is deterministic and schema-preserving."""

    payload = fixture_payload()
    encoded = normalised_json(payload)

    assert encoded.endswith("\n")
    assert json.loads(encoded) == payload
    assert normalised_json(payload) == encoded


def test_fixture_markdown_exposes_gate_and_claim_boundary() -> None:
    """Public fixture docs include the reproduction command and no-QPU boundary."""

    text = fixture_markdown(fixture_payload())

    assert "# Symmetry-Sector Mitigation Planner Fixtures" in text
    assert "## Raw-count replay fixtures" in text
    assert "scpn-bench symmetry-sector-mitigation-gate" in text
    assert "Replay fixtures prove offline raw-count accounting" in text
