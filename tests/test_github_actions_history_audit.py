# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — GitHub Actions history audit tests
"""Tests for the GitHub Actions history audit helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_AUDIT_TOOL = ROOT / "tools" / "audit_github_actions_history.py"
_SPEC = importlib.util.spec_from_file_location("audit_github_actions_history", _AUDIT_TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

classify_workflow_runs = _MODULE.classify_workflow_runs
classified_runs_to_json = _MODULE.classified_runs_to_json
format_classified_runs = _MODULE.format_classified_runs
main = _MODULE.main
workflow_run_from_mapping = _MODULE.workflow_run_from_mapping
workflow_runs_from_json = _MODULE.workflow_runs_from_json


def _run(
    database_id: int,
    conclusion: str,
    created_at: str,
    workflow: str = "CI",
    branch: str = "main",
    status: str = "completed",
) -> dict[str, str | int]:
    return {
        "databaseId": database_id,
        "conclusion": conclusion,
        "status": status,
        "workflowName": workflow,
        "headSha": f"sha-{database_id}",
        "headBranch": branch,
        "createdAt": created_at,
        "event": "push",
    }


def test_actions_audit_marks_old_failures_resolved_by_later_success() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(1, "failure", "2026-05-06T00:00:00Z"),
                _run(2, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )

    classified = classify_workflow_runs(runs)

    assert classified[0].bucket == "resolved_failure"
    assert classified[0].safe_delete_candidate
    assert classified[1].bucket == "clean_success"


def test_actions_audit_keeps_unresolved_failures_when_no_later_success() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(1, "failure", "2026-05-06T00:00:00Z"),
                _run(2, "success", "2026-05-06T01:00:00Z", workflow="Docker"),
            ]
        )
    )

    classified = classify_workflow_runs(runs)

    assert classified[0].bucket == "unresolved_failure"
    assert not classified[0].safe_delete_candidate


def test_actions_audit_classifies_cancelled_runs_by_later_success_evidence() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(10, "cancelled", "2026-05-06T00:00:00Z"),
                _run(11, "cancelled", "2026-05-06T00:30:00Z", branch="feature"),
                _run(12, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )

    classified = classify_workflow_runs(runs)

    assert classified[0].bucket == "superseded_cancelled"
    assert classified[0].safe_delete_candidate
    assert classified[1].bucket == "unresolved_cancelled"
    assert not classified[1].safe_delete_candidate


def test_actions_audit_reports_in_progress_runs_as_not_deletable() -> None:
    runs = workflow_runs_from_json(
        json.dumps([_run(20, "", "2026-05-06T00:00:00Z", status="in_progress")])
    )

    classified = classify_workflow_runs(runs)

    assert classified[0].bucket == "in_progress"
    assert not classified[0].safe_delete_candidate


def test_actions_audit_cli_reads_fixture_and_returns_nonzero_for_unresolved(
    tmp_path: Path, capsys: object
) -> None:
    fixture = tmp_path / "runs.json"
    fixture.write_text(
        json.dumps([_run(30, "failure", "2026-05-06T00:00:00Z")]),
        encoding="utf-8",
    )

    assert main(["--input", str(fixture)]) == 1
    output = capsys.readouterr().out
    assert "unresolved_failure: 1" in output


def test_actions_audit_text_summary_lists_safe_delete_candidates() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(40, "failure", "2026-05-06T00:00:00Z"),
                _run(41, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )

    summary = format_classified_runs(classify_workflow_runs(runs))

    assert "resolved_failure: 1" in summary
    assert "40 CI main resolved_failure" in summary


def test_actions_audit_json_output_is_deterministic_and_machine_readable() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(60, "failure", "2026-05-06T00:00:00Z"),
                _run(61, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )

    encoded = classified_runs_to_json(classify_workflow_runs(runs))
    decoded = json.loads(encoded)

    assert decoded[0]["databaseId"] == 60
    assert decoded[0]["bucket"] == "resolved_failure"
    assert decoded[0]["safeDeleteCandidate"] is True
    assert decoded[0]["createdAt"] == "2026-05-06T00:00:00Z"


def test_actions_audit_cli_json_mode_returns_zero_when_no_unresolved_runs(
    tmp_path: Path, capsys: object
) -> None:
    fixture = tmp_path / "runs.json"
    fixture.write_text(
        json.dumps([_run(70, "success", "2026-05-06T00:00:00Z")]),
        encoding="utf-8",
    )

    assert main(["--input", str(fixture), "--json"]) == 0
    decoded = json.loads(capsys.readouterr().out)
    assert decoded[0]["bucket"] == "clean_success"


def test_actions_audit_rejects_non_array_json() -> None:
    with pytest.raises(ValueError, match="must be an array"):
        workflow_runs_from_json(json.dumps({"databaseId": 80}))


def test_actions_audit_normalises_naive_timestamps_to_utc() -> None:
    run = workflow_run_from_mapping(_run(90, "success", "2026-05-06T00:00:00"))

    encoded = classified_runs_to_json(classify_workflow_runs((run,)))

    assert json.loads(encoded)[0]["createdAt"] == "2026-05-06T00:00:00Z"
