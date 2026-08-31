# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CI Workflow Inventory
"""Read the distributed CI workflow as one ordered policy surface."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_POLICY = REPOSITORY_ROOT / "tools/ci_workflow_policy.json"
"""Versioned category ownership and workflow-size policy."""


class WorkflowCategory(TypedDict):
    """One reusable CI workflow and its exclusively owned jobs."""

    id: str
    workflow: str
    caller_needs: list[str]
    jobs: list[str]


class WorkflowLimits(TypedDict):
    """Repository-local limits that prevent workflow GodFiles."""

    coordinator_max_lines: int
    coordinator_max_bytes: int
    reusable_max_lines: int
    reusable_max_bytes: int
    max_reusable_workflows: int


class WorkflowPolicy(TypedDict):
    """Complete CI coordinator, category, order, and limit contract."""

    schema_version: int
    coordinator: str
    required_gate: str
    limits: WorkflowLimits
    categories: list[WorkflowCategory]
    job_order: list[str]
    optional_jobs: list[str]


def load_ci_workflow_policy() -> WorkflowPolicy:
    """Load the versioned CI workflow ownership policy."""
    payload = json.loads(CI_WORKFLOW_POLICY.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CI workflow policy must be a JSON object")
    return cast(WorkflowPolicy, payload)


def ci_workflow_paths(policy: WorkflowPolicy | None = None) -> tuple[Path, ...]:
    """Return the coordinator followed by reusable workflows in policy order."""
    resolved = load_ci_workflow_policy() if policy is None else policy
    paths = [REPOSITORY_ROOT / resolved["coordinator"]]
    paths.extend(REPOSITORY_ROOT / category["workflow"] for category in resolved["categories"])
    return tuple(paths)


def _job_blocks(workflow: str) -> dict[str, str]:
    """Extract top-level job blocks without normalising their YAML text."""
    lines = workflow.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    jobs_seen = False
    for index, line in enumerate(lines):
        if line.rstrip("\n") == "jobs:":
            jobs_seen = True
            continue
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if jobs_seen and match:
            starts.append((index, match.group(1)))
    blocks: dict[str, str] = {}
    for position, (start, job_id) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        if job_id in blocks:
            raise ValueError(f"CI job appears multiple times in one workflow: {job_id}")
        blocks[job_id] = "".join(lines[start:end]).strip("\n")
    return blocks


def read_ci_workflow_source() -> str:
    """Return all real CI jobs in their historical logical order.

    The compatibility view lets job-contract tests inspect the distributed
    workflow without binding themselves to a physical category file. It is
    assembled only from executable workflow files; no duplicate snapshot is
    stored.
    """
    policy = load_ci_workflow_policy()
    coordinator_path = REPOSITORY_ROOT / policy["coordinator"]
    coordinator = coordinator_path.read_text(encoding="utf-8")
    prefix, _separator, _jobs = coordinator.partition("jobs:\n")
    blocks: dict[str, str] = {}
    for path in ci_workflow_paths(policy)[1:]:
        for job_id, block in _job_blocks(path.read_text(encoding="utf-8")).items():
            if job_id in blocks:
                raise ValueError(f"CI job appears in multiple reusable workflows: {job_id}")
            blocks[job_id] = block
    coordinator_blocks = _job_blocks(coordinator)
    gate_id = policy["required_gate"]
    if gate_id not in coordinator_blocks:
        raise ValueError(f"CI coordinator is missing required gate {gate_id}")
    missing = set(policy["job_order"]) - blocks.keys()
    if missing:
        raise ValueError(f"CI workflow inventory is missing jobs: {sorted(missing)}")
    ordered: list[str] = []
    for job_id in policy["job_order"]:
        block = blocks[job_id]
        if job_id != "lint":
            needs = re.search(r"(?m)^    needs: (?P<value>.+)$", block)
            if needs is None:
                block = block.replace(f"  {job_id}:\n", f"  {job_id}:\n    needs: lint\n", 1)
            else:
                value = needs.group("value")
                dependencies = value[1:-1] if value.startswith("[") else value
                block = block[: needs.start()] + (
                    f"    needs: [lint, {dependencies}]" + block[needs.end() :]
                )
        ordered.append(block)
    compatibility_gate = re.sub(
        r"(?m)^    needs:.*$",
        "    needs: [" + ", ".join(policy["job_order"]) + "]",
        coordinator_blocks[gate_id],
        count=1,
    )
    ordered.append(compatibility_gate)
    return prefix + "jobs:\n" + "\n\n".join(ordered) + "\n"


def workflow_path_for_job(job_id: str) -> Path:
    """Resolve the reusable workflow that exclusively owns ``job_id``."""
    policy = load_ci_workflow_policy()
    for category in policy["categories"]:
        if job_id in category["jobs"]:
            return REPOSITORY_ROOT / category["workflow"]
    if job_id == policy["required_gate"]:
        return REPOSITORY_ROOT / policy["coordinator"]
    raise KeyError(job_id)


__all__ = [
    "CI_WORKFLOW_POLICY",
    "REPOSITORY_ROOT",
    "WorkflowCategory",
    "WorkflowLimits",
    "WorkflowPolicy",
    "ci_workflow_paths",
    "load_ci_workflow_policy",
    "read_ci_workflow_source",
    "workflow_path_for_job",
]
