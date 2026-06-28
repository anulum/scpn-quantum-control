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
import subprocess
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


def test_actions_audit_classifies_other_completed_conclusions() -> None:
    runs = workflow_runs_from_json(json.dumps([_run(25, "skipped", "2026-05-06T00:00:00Z")]))

    classified = classify_workflow_runs(runs)

    assert classified[0].bucket == "other_completed"
    assert not classified[0].safe_delete_candidate


def test_actions_audit_cli_reads_fixture_and_returns_nonzero_for_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = tmp_path / "runs.json"
    fixture.write_text(
        json.dumps([_run(70, "success", "2026-05-06T00:00:00Z")]),
        encoding="utf-8",
    )

    assert main(["--input", str(fixture), "--json"]) == 0
    decoded = json.loads(capsys.readouterr().out)
    assert decoded[0]["bucket"] == "clean_success"


def test_actions_audit_cli_requires_repo_without_input() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_actions_audit_cli_loads_runs_from_gh_when_input_is_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = workflow_runs_from_json(json.dumps([_run(78, "success", "2026-05-06T00:00:00Z")]))
    monkeypatch.setattr(_MODULE, "_load_runs_from_gh", lambda _repo, _limit: runs)

    assert main(["--repo", "owner/repo", "--json"]) == 0

    decoded = json.loads(capsys.readouterr().out)
    assert decoded[0]["databaseId"] == 78
    assert decoded[0]["bucket"] == "clean_success"


def test_actions_audit_loads_runs_through_admitted_gh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(_MODULE, "_resolve_gh_executable", lambda: sys.executable)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["shell"] = kwargs.get("shell")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps([_run(80, "success", "2026-05-06T00:00:00Z")]),
            stderr="",
        )

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    runs = _MODULE._load_runs_from_gh("owner/repo", 25)

    assert runs[0].database_id == 80
    assert observed["command"] == [
        sys.executable,
        "run",
        "list",
        "--repo",
        "owner/repo",
        "--limit",
        "25",
        "--json",
        _MODULE.RUN_LIST_FIELDS,
    ]
    assert observed["shell"] is False


def test_actions_audit_rejects_missing_gh_for_live_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "_resolve_gh_executable", lambda: None)

    with pytest.raises(RuntimeError, match="gh executable is required"):
        _MODULE._load_runs_from_gh("owner/repo", 25)


def test_actions_audit_gh_resolver_rejects_missing_stale_and_non_executable_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_MODULE.shutil, "which", lambda _name: None)
    assert _MODULE._resolve_gh_executable() is None

    missing = tmp_path / "missing-gh"
    monkeypatch.setattr(_MODULE.shutil, "which", lambda _name: str(missing))
    assert _MODULE._resolve_gh_executable() is None

    candidate = tmp_path / "gh"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(_MODULE.shutil, "which", lambda _name: str(candidate))
    assert _MODULE._resolve_gh_executable() is None


def test_actions_audit_gh_resolver_admits_current_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE.shutil, "which", lambda _name: sys.executable)

    resolved = _MODULE._resolve_gh_executable()

    assert resolved is not None
    assert Path(resolved).is_absolute()


def test_actions_audit_rejects_non_array_json() -> None:
    with pytest.raises(ValueError, match="must be an array"):
        workflow_runs_from_json(json.dumps({"databaseId": 80}))


def test_actions_audit_normalises_naive_timestamps_to_utc() -> None:
    run = workflow_run_from_mapping(_run(90, "success", "2026-05-06T00:00:00"))

    encoded = classified_runs_to_json(classify_workflow_runs((run,)))

    assert json.loads(encoded)[0]["createdAt"] == "2026-05-06T00:00:00Z"
