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


def test_link_check_audit_cli_returns_nonzero_for_live_failures(
    tmp_path: Path, capsys: object
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
