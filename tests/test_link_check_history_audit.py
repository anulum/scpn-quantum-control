# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — link-check history audit tests
"""Tests for the Link Check history audit helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
_AUDIT_TOOL = TOOLS / "audit_link_check_history.py"
_SPEC = importlib.util.spec_from_file_location("audit_link_check_history", _AUDIT_TOOL)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

accepted_failures_from_json = _MODULE.accepted_failures_from_json
classify_link_check_runs = _MODULE.classify_link_check_runs
classified_link_runs_to_json = _MODULE.classified_link_runs_to_json
format_classified_link_runs = _MODULE.format_classified_link_runs
main = _MODULE.main
workflow_runs_from_json = _MODULE.workflow_runs_from_json


def _run(
    database_id: int,
    conclusion: str,
    created_at: str,
    workflow: str = "Link Check",
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


def test_link_check_audit_marks_resolved_failures_as_safe_candidates() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(1, "failure", "2026-05-06T00:00:00Z"),
                _run(2, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )

    classified = classify_link_check_runs(runs)

    assert classified[0].bucket == "resolved_link_failure"
    assert classified[0].safe_delete_candidate
    assert classified[1].bucket == "clean_success"


def test_link_check_audit_reports_live_failure_without_later_success() -> None:
    runs = workflow_runs_from_json(json.dumps([_run(10, "failure", "2026-05-06T00:00:00Z")]))

    classified = classify_link_check_runs(runs)

    assert classified[0].bucket == "live_link_failure"
    assert not classified[0].safe_delete_candidate


def test_link_check_audit_ignores_other_workflows() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(20, "failure", "2026-05-06T00:00:00Z", workflow="CI"),
                _run(21, "success", "2026-05-06T01:00:00Z", workflow="Docker"),
            ]
        )
    )

    assert classify_link_check_runs(runs) == ()


def test_link_check_audit_reports_in_progress_runs_as_not_deletable() -> None:
    runs = workflow_runs_from_json(
        json.dumps([_run(25, "", "2026-05-06T00:00:00Z", status="in_progress")])
    )

    classified = classify_link_check_runs(runs)

    assert classified[0].bucket == "in_progress"
    assert not classified[0].safe_delete_candidate


def test_link_check_audit_records_accepted_external_transient_failures() -> None:
    runs = workflow_runs_from_json(json.dumps([_run(30, "failure", "2026-05-06T00:00:00Z")]))
    accepted = accepted_failures_from_json(
        json.dumps(
            [
                {
                    "databaseId": 30,
                    "reason": "External journal returned HTTP 503 to HEAD requests.",
                }
            ]
        )
    )

    classified = classify_link_check_runs(runs, accepted_failures=accepted)

    assert classified[0].bucket == "accepted_external_transient"
    assert "HTTP 503" in classified[0].reason
    assert not classified[0].safe_delete_candidate


def test_link_check_audit_classifies_cancelled_runs_by_later_success_evidence() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(35, "cancelled", "2026-05-06T00:00:00Z"),
                _run(36, "cancelled", "2026-05-06T00:30:00Z", branch="docs"),
                _run(37, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )

    classified = classify_link_check_runs(runs)

    assert classified[0].bucket == "superseded_cancelled"
    assert classified[0].safe_delete_candidate
    assert classified[1].bucket == "unresolved_cancelled"
    assert not classified[1].safe_delete_candidate


def test_link_check_audit_classifies_other_completed_conclusions() -> None:
    runs = workflow_runs_from_json(json.dumps([_run(38, "skipped", "2026-05-06T00:00:00Z")]))

    classified = classify_link_check_runs(runs)

    assert classified[0].bucket == "other_completed"
    assert not classified[0].safe_delete_candidate


def test_link_check_audit_cli_returns_nonzero_for_live_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = tmp_path / "runs.json"
    fixture.write_text(
        json.dumps([_run(40, "failure", "2026-05-06T00:00:00Z")]),
        encoding="utf-8",
    )

    assert main(["--input", str(fixture)]) == 1
    assert "live_link_failure: 1" in capsys.readouterr().out


def test_link_check_audit_summary_lists_accepted_and_deletable_runs() -> None:
    runs = workflow_runs_from_json(
        json.dumps(
            [
                _run(50, "failure", "2026-05-06T00:00:00Z"),
                _run(51, "failure", "2026-05-06T00:10:00Z", branch="docs"),
                _run(52, "success", "2026-05-06T01:00:00Z"),
            ]
        )
    )
    accepted = accepted_failures_from_json(
        json.dumps([{"databaseId": 51, "reason": "Vendor rate-limit transient."}])
    )

    summary = format_classified_link_runs(
        classify_link_check_runs(runs, accepted_failures=accepted)
    )

    assert "accepted_external_transient: 1" in summary
    assert "resolved_link_failure: 1" in summary
    assert "51: Vendor rate-limit transient." in summary
    assert "50 main resolved_link_failure" in summary


def test_link_check_audit_json_output_preserves_live_failure_contract() -> None:
    runs = workflow_runs_from_json(json.dumps([_run(60, "failure", "2026-05-06T00:00:00Z")]))

    encoded = classified_link_runs_to_json(classify_link_check_runs(runs))
    decoded = json.loads(encoded)

    assert decoded[0]["databaseId"] == 60
    assert decoded[0]["bucket"] == "live_link_failure"
    assert decoded[0]["safeDeleteCandidate"] is False


def test_link_check_audit_cli_accepts_external_transient_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = tmp_path / "runs.json"
    accepted = tmp_path / "accepted.json"
    fixture.write_text(
        json.dumps([_run(70, "failure", "2026-05-06T00:00:00Z")]),
        encoding="utf-8",
    )
    accepted.write_text(
        json.dumps([{"databaseId": 70, "reason": "External HTTP 429 transient."}]),
        encoding="utf-8",
    )

    assert main(["--input", str(fixture), "--accepted", str(accepted), "--json"]) == 0
    decoded = json.loads(capsys.readouterr().out)
    assert decoded[0]["bucket"] == "accepted_external_transient"
    assert decoded[0]["reason"] == "External HTTP 429 transient."


def test_link_check_audit_accepts_empty_external_transient_file() -> None:
    assert accepted_failures_from_json("") == {}
    assert accepted_failures_from_json("   \n") == {}


def test_link_check_audit_cli_requires_repo_without_input() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_link_check_audit_cli_loads_runs_from_gh_when_input_is_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = workflow_runs_from_json(json.dumps([_run(75, "success", "2026-05-06T00:00:00Z")]))
    monkeypatch.setattr(_MODULE, "_load_runs_from_gh", lambda _repo, _limit: runs)

    assert main(["--repo", "owner/repo", "--json"]) == 0

    decoded = json.loads(capsys.readouterr().out)
    assert decoded[0]["databaseId"] == 75
    assert decoded[0]["bucket"] == "clean_success"


def test_link_check_audit_rejects_non_array_accepted_failure_json() -> None:
    with pytest.raises(ValueError, match="must be a JSON array"):
        accepted_failures_from_json(json.dumps({"databaseId": 80}))


def test_link_check_audit_rejects_non_object_accepted_failure_entries() -> None:
    with pytest.raises(ValueError, match="entries must be JSON objects"):
        accepted_failures_from_json(json.dumps([80]))
