# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the advisory workflow step gate
"""Exercise the advisory workflow step gate with real workflows in real Git repositories."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import audit_advisory_workflow_steps as gate

REPOSITORY = Path(__file__).resolve().parents[1]

CHECKS = """\
name: Checks
on: push
jobs:
  staged:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - name: Measure
        run: |
          tool run || true
          tool report || :
      - name: Classify
        continue-on-error: ${{ github.event_name == 'schedule' }}
        run: tool classify
      - run: |
          set +e
          tool probe
  strict:
    runs-on: ubuntu-latest
    continue-on-error: false
    steps:
      - name: Enforce
        continue-on-error: false
        run: tool enforce || truename --flag
  reusable:
    uses: ./.github/workflows/other.yml
"""

RECORDED: list[dict[str, Any]] = [
    {
        "workflow": "checks.yml",
        "job": "staged",
        "kind": gate.JOB,
        "reason": "staged lane",
        "promotion": "three green runs",
    },
    {
        "workflow": "checks.yml",
        "job": "staged",
        "step": "Measure",
        "kind": gate.SHELL,
        "count": 2,
        "reason": "the report follows",
        "permanent": "decides nothing",
    },
    {
        "workflow": "checks.yml",
        "job": "staged",
        "step": "Classify",
        "kind": gate.STEP,
        "reason": "scheduled runs only report",
        "undecided": "no condition yet",
    },
    {
        "workflow": "checks.yml",
        "job": "staged",
        "step": "set +e",
        "kind": gate.SHELL,
        "reason": "probe may be absent",
        "undecided": "whether the probe is required",
    },
]


def _git(repo: Path, *arguments: str) -> None:
    """Run Git in ``repo`` and require success."""
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def _repository(root: Path, workflows: dict[str, str], sites: list[dict[str, Any]]) -> Path:
    """Create a repository with tracked workflows and an advisory policy.

    Parameters
    ----------
    root
        Empty directory that becomes the repository.
    workflows
        Workflow text by file name.
    sites
        Rows of the advisory policy.

    Returns
    -------
    Path
        The repository root.

    """
    directory = root / gate.WORKFLOW_ROOT
    directory.mkdir(parents=True)
    for name, text in workflows.items():
        (directory / name).write_text(text, encoding="utf-8")
    policy = root / gate.DEFAULT_POLICY
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({"schema": gate.SCHEMA, "sites": sites}), encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "add", "--all")
    return root


def test_scan_finds_job_step_and_shell_tolerances_in_tracked_workflows(tmp_path: Path) -> None:
    """All three kinds are found, with counts; explicit false and look-alike words are not.

    An untracked workflow and a file that is not a workflow are not read.
    """
    repo = _repository(tmp_path, {"checks.yml": CHECKS, "notes.txt": "|| true\n"}, RECORDED)
    (repo / gate.WORKFLOW_ROOT / "untracked.yml").write_text(
        "jobs:\n  loose:\n    continue-on-error: true\n", encoding="utf-8"
    )

    assert gate.tracked_workflows(repo) == [".github/workflows/checks.yml"]
    assert gate.scan(repo) == [
        gate.Site("checks.yml", "staged", "", gate.JOB, 1),
        gate.Site("checks.yml", "staged", "Measure", gate.SHELL, 2),
        gate.Site("checks.yml", "staged", "Classify", gate.STEP, 1),
        gate.Site("checks.yml", "staged", "set +e", gate.SHELL, 1),
    ]


def test_gate_passes_when_every_site_is_recorded_and_prints_undecided_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A complete policy passes; undecided rows are printed with what is missing."""
    repo = _repository(tmp_path, {"checks.yml": CHECKS}, RECORDED)

    assert gate.main(["--repo", str(repo)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == [
        "undecided: checks.yml / staged / Classify (step-continue-on-error): no condition yet",
        "undecided: checks.yml / staged / set +e (shell-tolerance): whether the probe is required",
        "Advisory workflow steps: 4 sites; 4 recorded (1 staged, 1 permanent, 2 undecided); "
        "0 problems",
    ]


def test_gate_reports_unrecorded_sites_changed_counts_and_rows_without_a_site(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A new tolerance, a changed count and a stale row are each one problem."""
    rows = [dict(row) for row in RECORDED[:2]]
    rows[1]["count"] = 1
    rows.append(
        {
            "workflow": "checks.yml",
            "job": "removed",
            "kind": gate.JOB,
            "reason": "gone",
            "permanent": "gone",
        }
    )
    repo = _repository(tmp_path, {"checks.yml": CHECKS}, rows)

    assert gate.main(["--repo", str(repo)]) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "tolerance count changed: checks.yml / staged / Measure (shell-tolerance): "
        "recorded 1, found 2",
        "tolerated failure outside the advisory policy: "
        "checks.yml / staged / Classify (step-continue-on-error)",
        "tolerated failure outside the advisory policy: "
        "checks.yml / staged / set +e (shell-tolerance)",
        "advisory policy row has no site: checks.yml / removed (job-continue-on-error)",
    ]
    assert captured.out.endswith("4 problems\n")


@pytest.mark.parametrize(
    ("workflow", "message"),
    [
        ("- not\n- a mapping\n", r"workflow is not a mapping: \.github/workflows/checks\.yml"),
        ("jobs: [unbalanced\n", "advisory workflow step gate failed"),
        (
            "jobs:\n  twins:\n    steps:\n      - name: Same\n        run: a || true\n"
            "      - name: Same\n        run: b || true\n",
            r"steps cannot be told apart, name them: checks\.yml / twins / Same",
        ),
    ],
)
def test_unreadable_workflows_and_indistinguishable_steps_fail_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], workflow: str, message: str
) -> None:
    """A workflow that is not a mapping, broken YAML and twin steps stop the gate.

    Parameters
    ----------
    tmp_path
        Directory that becomes the repository.
    capsys
        Captured output of the gate.
    workflow
        Text of the tracked workflow.
    message
        Pattern the failure line must match.

    """
    repo = _repository(tmp_path, {"checks.yml": workflow}, [])

    assert gate.main(["--repo", str(repo)]) == 1
    assert re.search(message, capsys.readouterr().err) is not None


def test_workflows_without_jobs_or_with_other_shapes_have_no_site(tmp_path: Path) -> None:
    """A workflow without jobs, a job that is not a mapping and odd step lists yield nothing."""
    repo = _repository(
        tmp_path,
        {
            "empty.yml": "name: Empty\non: push\n",
            "odd.yaml": (
                "jobs:\n  text: just a string\n  no-steps:\n    runs-on: x\n"
                "  odd-steps:\n    steps:\n      - plain string\n      - uses: actions/checkout@v7\n"
                "      - {}\n"
            ),
        },
        [],
    )

    assert gate.tracked_workflows(repo) == [
        ".github/workflows/empty.yml",
        ".github/workflows/odd.yaml",
    ]
    assert gate.scan(repo) == []
    assert gate.main(["--repo", str(repo)]) == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "other"}, "unknown advisory site kind: other"),
        ({"workflow": " "}, "needs a non-empty workflow"),
        ({"job": 3}, "needs a non-empty job"),
        ({"reason": ""}, "needs a non-empty reason"),
        ({"kind": gate.STEP}, "needs a non-empty step"),
        ({"count": 2}, "only a shell tolerance row records a count"),
        ({"count": 0}, "count must be a positive integer"),
        ({"count": True}, "count must be a positive integer"),
        ({"permanent": "also"}, "needs exactly one of promotion, permanent, undecided"),
        ({"promotion": " "}, "needs a non-empty promotion"),
    ],
)
def test_malformed_policy_rows_are_refused(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    """Every malformed field of a policy row is refused with its own message.

    Parameters
    ----------
    tmp_path
        Directory that receives the policy.
    changes
        Fields replaced in an otherwise valid job row.
    message
        Pattern the refusal must match.

    """
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({"schema": gate.SCHEMA, "sites": [{**RECORDED[0], **changes}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        gate.load_policy(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "unsupported advisory workflow policy schema"),
        ({"schema": "other", "sites": []}, "unsupported advisory workflow policy schema"),
        ({"schema": gate.SCHEMA, "sites": {}}, "sites must be a list of objects"),
        ({"schema": gate.SCHEMA, "sites": [1]}, "sites must be a list of objects"),
        (
            {"schema": gate.SCHEMA, "sites": [RECORDED[0], RECORDED[0]]},
            r"recorded twice: checks\.yml / staged \(job-continue-on-error\)",
        ),
        (
            {
                "schema": gate.SCHEMA,
                "sites": [{k: v for k, v in RECORDED[0].items() if k != "promotion"}],
            },
            "needs exactly one of promotion, permanent, undecided",
        ),
    ],
)
def test_malformed_policy_documents_are_refused(
    tmp_path: Path, document: Any, message: str
) -> None:
    """Another schema, a wrong site list, a repeated site and a missing outcome are refused.

    Parameters
    ----------
    tmp_path
        Directory that receives the policy.
    document
        The whole policy document.
    message
        Pattern the refusal must match.

    """
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        gate.load_policy(path)


def test_gate_fails_without_a_policy_or_outside_a_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing policy, a directory that is no repository and a missing Git fail the gate."""
    assert gate.main(["--repo", str(tmp_path)]) == 1
    assert "advisory workflow step gate failed" in capsys.readouterr().err

    with pytest.raises(ValueError, match="git ls-files failed"):
        gate.tracked_workflows(tmp_path)

    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="cannot run git"):
        gate.tracked_workflows(tmp_path)


def test_repository_policy_records_every_site_of_its_workflows() -> None:
    """The repository's own workflows and policy agree, row for row."""
    rows = gate.load_policy(REPOSITORY / gate.DEFAULT_POLICY)

    assert gate.compare(gate.scan(REPOSITORY), rows) == []
    assert {row.outcome for row in rows} <= set(gate.OUTCOMES)
