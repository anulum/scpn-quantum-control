# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 K_nm preregistered replay gate tests
"""Tests for Paper 0 K_nm replay comparator and gate wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.compare_paper0_knm_preregistered_replay import (
    compare_paper0_knm_preregistered_replay,
)
from scripts.export_paper0_knm_replay_contract import (
    build_contract_payload,
    validate_replay_against_contract,
)
from scripts.run_paper0_knm_preregistered_replay import (
    build_replay_payload,
    write_replay_artifacts,
)
from scripts.run_paper0_knm_preregistered_replay_gate import (
    COMPARATOR_SCRIPT,
    build_paper0_knm_preregistered_replay_gate_commands,
)


def _make_expected_payload(tmp_path: Path) -> tuple[Path, Path]:
    expected_json = tmp_path / "expected_paper0_knm_preregistered_replay.json"
    expected_markdown = tmp_path / "expected_paper0_knm_preregistered_replay.md"
    write_replay_artifacts(output_json=expected_json, output_doc=expected_markdown)
    return expected_json, expected_markdown


def test_compare_returns_valid_for_matching_expected_artifacts(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)

    result = compare_paper0_knm_preregistered_replay(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is True
    assert result["blockers"] == ()
    assert result["digests"]["expected_json_sha256"] == result["digests"]["actual_json_sha256"]
    assert (
        result["digests"]["expected_markdown_sha256"]
        == result["digests"]["actual_markdown_sha256"]
    )


def test_compare_reports_json_drift(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    payload = json.loads(expected_json.read_text(encoding="utf-8"))
    payload["status"] = "incorrectly_promoted"
    expected_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = compare_paper0_knm_preregistered_replay(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is False
    assert any("JSON artifacts differ" in blocker for blocker in result["blockers"])


def test_compare_rejects_unsafe_promotion_payload_even_when_actual_matches(
    tmp_path: Path,
) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    unsafe = json.loads(expected_json.read_text(encoding="utf-8"))
    unsafe["promotion_decision"]["decision"] = "promote"
    unsafe["promotion_decision"]["hardware_submission_authorised"] = True
    actual_json = tmp_path / "actual_unsafe.json"
    actual_markdown = tmp_path / "actual.md"
    actual_json.write_text(json.dumps(unsafe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    actual_markdown.write_text(expected_markdown.read_text(encoding="utf-8"), encoding="utf-8")
    expected_json.write_text(json.dumps(unsafe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = compare_paper0_knm_preregistered_replay(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
        actual_json_path=actual_json,
        actual_markdown_path=actual_markdown,
    )

    assert result["valid"] is False
    assert any(
        "promotion decision must be do_not_promote" in blocker for blocker in result["blockers"]
    )
    assert any(
        "must not authorise hardware submission" in blocker for blocker in result["blockers"]
    )


def test_compare_rejects_stale_input_manifest_digest_even_when_actual_matches(
    tmp_path: Path,
) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    stale = json.loads(expected_json.read_text(encoding="utf-8"))
    stale["reproducibility"]["input_manifest"]["primary_candidate"]["sha256"] = "0" * 64
    actual_json = tmp_path / "actual_stale_digest.json"
    actual_markdown = tmp_path / "actual.md"
    actual_json.write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    actual_markdown.write_text(expected_markdown.read_text(encoding="utf-8"), encoding="utf-8")
    expected_json.write_text(
        json.dumps(stale, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = compare_paper0_knm_preregistered_replay(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
        actual_json_path=actual_json,
        actual_markdown_path=actual_markdown,
    )

    assert result["valid"] is False
    assert any("input manifest digest mismatch" in blocker for blocker in result["blockers"])


def test_compare_reports_markdown_drift(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    expected_markdown.write_text(
        expected_markdown.read_text(encoding="utf-8") + "\n<!-- drift-marker -->\n",
        encoding="utf-8",
    )

    result = compare_paper0_knm_preregistered_replay(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is False
    assert any("Markdown artifacts differ" in blocker for blocker in result["blockers"])


def test_compare_rejects_asymmetric_actual_paths(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    with pytest.raises(ValueError, match="--actual-json and --actual-markdown"):
        compare_paper0_knm_preregistered_replay(
            expected_json_path=expected_json,
            expected_markdown_path=expected_markdown,
            actual_json_path=tmp_path / "actual.json",
        )


def test_gate_helper_builds_comparator_command_with_explicit_expected_paths(
    tmp_path: Path,
) -> None:
    expected_json = tmp_path / "expected.json"
    expected_markdown = tmp_path / "expected.md"

    commands = build_paper0_knm_preregistered_replay_gate_commands(
        comparator_script=COMPARATOR_SCRIPT,
        expected_json=expected_json,
        expected_markdown=expected_markdown,
    )

    assert commands == (
        (
            sys.executable,
            "scripts/compare_paper0_knm_preregistered_replay.py",
            "--expected-json",
            str(expected_json),
            "--expected-markdown",
            str(expected_markdown),
        ),
    )


def test_contract_export_preserves_fail_closed_replay_boundary() -> None:
    contract = build_contract_payload()

    assert contract["schema"] == "paper0_knm_preregistered_replay_contract_v1"
    assert contract["replay_schema"] == "paper0_knm_preregistered_replay_v1"
    assert "hardware execution" in contract["claim_boundary"]
    assert set(contract["locked_inputs"]) == {
        "primary_candidate",
        "negative_control",
        "negative_measured_couplings",
    }
    assert contract["digest_algorithm"] == "sha256"
    assert contract["required_gates"]["qpu_submission"] == ("blocked_no_qpu_preregistration_lane")
    assert contract["required_promotion_decision"] == {
        "decision": "do_not_promote",
        "hardware_submission_authorised": False,
        "claim_promotion_authorised": False,
        "minimum_required_evidence_items": 4,
        "minimum_falsifier_items": 4,
    }


def test_contract_validator_accepts_current_replay_payload() -> None:
    blockers = validate_replay_against_contract(build_replay_payload())

    assert blockers == ()


def test_contract_validator_rejects_weakened_replay_boundary() -> None:
    replay = build_replay_payload()
    replay["gates"]["qpu_submission"] = "pass"
    replay["promotion_decision"]["hardware_submission_authorised"] = True
    replay["reproducibility"]["input_manifest"]["primary_candidate"]["path"] = "data/changed.json"

    blockers = validate_replay_against_contract(replay)

    assert any("qpu_submission" in blocker for blocker in blockers)
    assert any("hardware_submission_authorised" in blocker for blocker in blockers)
    assert any("primary_candidate path does not match contract" in blocker for blocker in blockers)
