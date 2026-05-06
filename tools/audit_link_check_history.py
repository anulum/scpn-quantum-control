# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — link-check history audit helper
"""Summarise Link Check workflow history without mutating GitHub state."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from audit_github_actions_history import (
    RUN_LIST_FIELDS,
    WorkflowRun,
    workflow_runs_from_json,
)


@dataclass(frozen=True)
class AcceptedLinkFailure:
    """Explicitly accepted external-transient link-check failure."""

    database_id: int
    reason: str


@dataclass(frozen=True)
class ClassifiedLinkRun:
    """A link-check workflow run with a documentation-hygiene classification."""

    run: WorkflowRun
    bucket: str
    reason: str
    safe_delete_candidate: bool


def accepted_failures_from_json(text: str) -> dict[int, AcceptedLinkFailure]:
    """Parse accepted external-transient link failures from JSON."""
    if not text.strip():
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, list):
        raise ValueError("Accepted link failures must be a JSON array.")
    accepted: dict[int, AcceptedLinkFailure] = {}
    for item in loaded:
        if not isinstance(item, Mapping):
            raise ValueError("Accepted link failure entries must be JSON objects.")
        database_id = int(item["databaseId"])
        accepted[database_id] = AcceptedLinkFailure(
            database_id=database_id,
            reason=str(item.get("reason") or "Accepted external-transient link failure."),
        )
    return accepted


def _later_success_exists(run: WorkflowRun, runs: Sequence[WorkflowRun]) -> bool:
    """Return True when a later successful link-check run exists for the branch."""
    return any(
        candidate.head_branch == run.head_branch
        and candidate.created_at > run.created_at
        and candidate.status == "completed"
        and candidate.conclusion == "success"
        for candidate in runs
    )


def classify_link_check_runs(
    runs: Sequence[WorkflowRun],
    *,
    workflow_name: str = "Link Check",
    accepted_failures: Mapping[int, AcceptedLinkFailure] | None = None,
) -> tuple[ClassifiedLinkRun, ...]:
    """Classify Link Check history into live, resolved, accepted, and clean buckets."""
    accepted = accepted_failures or {}
    link_runs = tuple(
        sorted(
            (run for run in runs if run.workflow_name == workflow_name),
            key=lambda item: item.created_at,
        )
    )
    classified: list[ClassifiedLinkRun] = []
    for run in link_runs:
        later_success = _later_success_exists(run, link_runs)
        if run.status != "completed":
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="in_progress",
                    reason="Link Check run is not completed.",
                    safe_delete_candidate=False,
                )
            )
        elif run.conclusion == "success":
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="clean_success",
                    reason="Successful Link Check evidence.",
                    safe_delete_candidate=False,
                )
            )
        elif run.database_id in accepted:
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="accepted_external_transient",
                    reason=accepted[run.database_id].reason,
                    safe_delete_candidate=False,
                )
            )
        elif run.conclusion == "failure" and later_success:
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="resolved_link_failure",
                    reason="A later successful Link Check run exists for the same branch.",
                    safe_delete_candidate=True,
                )
            )
        elif run.conclusion == "failure":
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="live_link_failure",
                    reason="No later successful Link Check run exists for the same branch.",
                    safe_delete_candidate=False,
                )
            )
        elif run.conclusion == "cancelled" and later_success:
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="superseded_cancelled",
                    reason="A later successful Link Check run exists for the same branch.",
                    safe_delete_candidate=True,
                )
            )
        elif run.conclusion == "cancelled":
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="unresolved_cancelled",
                    reason="Cancelled Link Check run has no later successful evidence.",
                    safe_delete_candidate=False,
                )
            )
        else:
            classified.append(
                ClassifiedLinkRun(
                    run=run,
                    bucket="other_completed",
                    reason=f"Completed with conclusion {run.conclusion!r}.",
                    safe_delete_candidate=False,
                )
            )
    return tuple(classified)


def classified_link_runs_to_json(classified: Sequence[ClassifiedLinkRun]) -> str:
    """Serialise classified Link Check runs as deterministic JSON."""
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


def format_classified_link_runs(classified: Sequence[ClassifiedLinkRun]) -> str:
    """Render a compact human-readable Link Check history summary."""
    counts: dict[str, int] = {}
    for item in classified:
        counts[item.bucket] = counts.get(item.bucket, 0) + 1
    lines = ["Link Check history audit summary:"]
    for bucket in sorted(counts):
        lines.append(f"- {bucket}: {counts[bucket]}")
    live = [item for item in classified if item.bucket == "live_link_failure"]
    if live:
        lines.append("Live link failures:")
        lines.extend(
            f"- {item.run.database_id} {item.run.head_branch} {item.run.head_sha}" for item in live
        )
    accepted = [item for item in classified if item.bucket == "accepted_external_transient"]
    if accepted:
        lines.append("Accepted external-transient failures:")
        lines.extend(f"- {item.run.database_id}: {item.reason}" for item in accepted)
    deletable = [item for item in classified if item.safe_delete_candidate]
    if deletable:
        lines.append("Safe delete candidates:")
        lines.extend(
            f"- {item.run.database_id} {item.run.head_branch} {item.bucket}" for item in deletable
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
    parser.add_argument(
        "--accepted",
        type=Path,
        help="Optional JSON file listing accepted external-transient failures.",
    )
    parser.add_argument("--workflow-name", default="Link Check", help="Workflow name to audit.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    if args.input is not None:
        runs = workflow_runs_from_json(args.input.read_text(encoding="utf-8"))
    else:
        if not args.repo:
            parser.error("--repo is required when --input is not provided.")
        runs = _load_runs_from_gh(args.repo, args.limit)

    accepted = (
        accepted_failures_from_json(args.accepted.read_text(encoding="utf-8"))
        if args.accepted
        else {}
    )
    classified = classify_link_check_runs(
        runs,
        workflow_name=args.workflow_name,
        accepted_failures=accepted,
    )
    print(
        classified_link_runs_to_json(classified)
        if args.json
        else format_classified_link_runs(classified)
    )
    return 1 if any(item.bucket == "live_link_failure" for item in classified) else 0


if __name__ == "__main__":
    raise SystemExit(main())
