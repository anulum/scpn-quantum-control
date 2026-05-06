# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — GitHub Actions history audit helper
"""Classify GitHub Actions history into actionable hygiene buckets."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_LIST_FIELDS = "databaseId,conclusion,status,workflowName,headSha,headBranch,createdAt,event"


@dataclass(frozen=True)
class WorkflowRun:
    """Minimal workflow-run record required for history classification."""

    database_id: int
    conclusion: str
    status: str
    workflow_name: str
    head_sha: str
    head_branch: str
    created_at: datetime
    event: str

    @property
    def workflow_key(self) -> tuple[str, str]:
        """Return the workflow/branch key used for later-success evidence."""
        return (self.workflow_name, self.head_branch)


@dataclass(frozen=True)
class ClassifiedRun:
    """A workflow run with a release-hygiene classification."""

    run: WorkflowRun
    bucket: str
    reason: str
    safe_delete_candidate: bool


def _parse_timestamp(value: str) -> datetime:
    """Parse a GitHub timestamp into a timezone-aware datetime."""
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def workflow_run_from_mapping(raw: Mapping[str, Any]) -> WorkflowRun:
    """Convert a GitHub CLI JSON record into a WorkflowRun."""
    return WorkflowRun(
        database_id=int(raw["databaseId"]),
        conclusion=str(raw.get("conclusion") or ""),
        status=str(raw.get("status") or ""),
        workflow_name=str(raw.get("workflowName") or ""),
        head_sha=str(raw.get("headSha") or ""),
        head_branch=str(raw.get("headBranch") or ""),
        created_at=_parse_timestamp(str(raw["createdAt"])),
        event=str(raw.get("event") or ""),
    )


def workflow_runs_from_json(text: str) -> tuple[WorkflowRun, ...]:
    """Parse a JSON array produced by `gh run list --json ...`."""
    loaded = json.loads(text)
    if not isinstance(loaded, list):
        raise ValueError("Workflow history JSON must be an array of run records.")
    return tuple(workflow_run_from_mapping(item) for item in loaded)


def _later_success_exists(run: WorkflowRun, runs: Iterable[WorkflowRun]) -> bool:
    """Return True when a later successful run exists for the same workflow key."""
    return any(
        candidate.workflow_key == run.workflow_key
        and candidate.created_at > run.created_at
        and candidate.status == "completed"
        and candidate.conclusion == "success"
        for candidate in runs
    )


def classify_workflow_runs(runs: Sequence[WorkflowRun]) -> tuple[ClassifiedRun, ...]:
    """Classify workflow runs into resolved, unresolved, cancelled, and clean buckets."""
    sorted_runs = tuple(sorted(runs, key=lambda item: item.created_at))
    classified: list[ClassifiedRun] = []
    for run in sorted_runs:
        later_success = _later_success_exists(run, sorted_runs)
        if run.status != "completed":
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="in_progress",
                    reason="Run is not completed; do not delete.",
                    safe_delete_candidate=False,
                )
            )
        elif run.conclusion == "success":
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="clean_success",
                    reason="Successful run retained as current evidence.",
                    safe_delete_candidate=False,
                )
            )
        elif run.conclusion == "failure" and later_success:
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="resolved_failure",
                    reason="A later successful run exists for the same workflow and branch.",
                    safe_delete_candidate=True,
                )
            )
        elif run.conclusion == "failure":
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="unresolved_failure",
                    reason="No later successful run exists for the same workflow and branch.",
                    safe_delete_candidate=False,
                )
            )
        elif run.conclusion == "cancelled" and later_success:
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="superseded_cancelled",
                    reason="A later successful run exists for the same workflow and branch.",
                    safe_delete_candidate=True,
                )
            )
        elif run.conclusion == "cancelled":
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="unresolved_cancelled",
                    reason="Cancelled run has no later successful evidence.",
                    safe_delete_candidate=False,
                )
            )
        else:
            classified.append(
                ClassifiedRun(
                    run=run,
                    bucket="other_completed",
                    reason=f"Completed with conclusion {run.conclusion!r}.",
                    safe_delete_candidate=False,
                )
            )
    return tuple(classified)


def classified_runs_to_json(classified: Sequence[ClassifiedRun]) -> str:
    """Serialise classified runs as deterministic JSON."""
    rows = [
        {
            "databaseId": item.run.database_id,
            "workflowName": item.run.workflow_name,
            "headBranch": item.run.head_branch,
            "headSha": item.run.head_sha,
            "createdAt": item.run.created_at.isoformat().replace("+00:00", "Z"),
            "event": item.run.event,
            "status": item.run.status,
            "conclusion": item.run.conclusion,
            "bucket": item.bucket,
            "safeDeleteCandidate": item.safe_delete_candidate,
            "reason": item.reason,
        }
        for item in classified
    ]
    return json.dumps(rows, indent=2, sort_keys=True)


def format_classified_runs(classified: Sequence[ClassifiedRun]) -> str:
    """Render a compact human-readable audit summary."""
    counts: dict[str, int] = {}
    for item in classified:
        counts[item.bucket] = counts.get(item.bucket, 0) + 1
    lines = ["GitHub Actions history audit summary:"]
    for bucket in sorted(counts):
        lines.append(f"- {bucket}: {counts[bucket]}")
    deletable = [item for item in classified if item.safe_delete_candidate]
    if deletable:
        lines.append("Safe delete candidates:")
        lines.extend(
            f"- {item.run.database_id} {item.run.workflow_name} "
            f"{item.run.head_branch} {item.bucket}"
            for item in deletable
        )
    else:
        lines.append("Safe delete candidates: none")
    return "\n".join(lines)


def _load_runs_from_gh(repo: str, limit: int) -> tuple[WorkflowRun, ...]:
    """Load workflow history through the GitHub CLI."""
    command = [
        "gh",
        "run",
        "list",
        "--repo",
        repo,
        "--limit",
        str(limit),
        "--json",
        RUN_LIST_FIELDS,
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return workflow_runs_from_json(completed.stdout)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="GitHub repository as owner/name.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum runs to query via gh.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Read gh run-list JSON from this file instead of invoking gh.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.input is not None:
        runs = workflow_runs_from_json(args.input.read_text(encoding="utf-8"))
    else:
        if not args.repo:
            parser.error("--repo is required when --input is not provided.")
        runs = _load_runs_from_gh(args.repo, args.limit)

    classified = classify_workflow_runs(runs)
    print(classified_runs_to_json(classified) if args.json else format_classified_runs(classified))
    return 1 if any(item.bucket.startswith("unresolved") for item in classified) else 0


if __name__ == "__main__":
    raise SystemExit(main())
