# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CI Workflow Inventory
"""Read the distributed CI workflow as one ordered policy surface."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict, cast

from scpn_quantum_control.ci_workflow_ownership import (
    read_ci_job_blocks as _job_blocks,
)
from scpn_quantum_control.ci_workflow_ownership import (
    read_ci_workflow_policy,
    resolve_ci_workflow_owner,
)

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


def load_ci_workflow_policy(*, repo_root: Path | None = None) -> WorkflowPolicy:
    """Load the selected checkout's versioned CI ownership policy.

    Parameters
    ----------
    repo_root
        Explicit checkout root, or the current tool's repository configuration.

    Returns
    -------
    WorkflowPolicy
        Decoded policy; individual executable owners are checked on resolution.

    Raises
    ------
    ValueError
        If the policy is not a JSON object.
    OSError
        If the selected policy cannot be read.

    """
    path = CI_WORKFLOW_POLICY if repo_root is None else repo_root / "tools/ci_workflow_policy.json"
    return cast(WorkflowPolicy, read_ci_workflow_policy(path))


def ci_workflow_paths(
    policy: WorkflowPolicy | None = None,
    *,
    repo_root: Path | None = None,
) -> tuple[Path, ...]:
    """Return the selected checkout's declared workflows in policy order.

    Parameters
    ----------
    policy
        Explicit policy, or the selected checkout's policy.
    repo_root
        Checkout root; defaults to this tool's configured repository.

    Returns
    -------
    tuple[Path, ...]
        Coordinator followed by reusable workflows, without rewriting source.

    """
    root = REPOSITORY_ROOT if repo_root is None else repo_root
    resolved = load_ci_workflow_policy(repo_root=repo_root) if policy is None else policy
    paths = [root / resolved["coordinator"]]
    paths.extend(root / category["workflow"] for category in resolved["categories"])
    return tuple(paths)


def read_ci_workflow_source(*, repo_root: Path | None = None) -> str:
    """Return all real CI jobs in their historical logical order.

    The compatibility view lets job-contract tests inspect the distributed
    workflow without binding themselves to a physical category file. It is
    assembled only from executable workflow files; no duplicate snapshot is
    stored.

    Parameters
    ----------
    repo_root
        Checkout to inspect, or the current tool's repository configuration.

    Returns
    -------
    str
        Compatibility view assembled from real executable job blocks.

    Raises
    ------
    ValueError
        If jobs are duplicated, missing or the required gate is absent.
    OSError
        If a declared workflow cannot be read.

    """
    root = REPOSITORY_ROOT if repo_root is None else repo_root
    policy = load_ci_workflow_policy(repo_root=repo_root)
    coordinator_path = root / policy["coordinator"]
    coordinator = coordinator_path.read_text(encoding="utf-8")
    prefix, _separator, _jobs = coordinator.partition("jobs:\n")
    blocks: dict[str, str] = {}
    for path in ci_workflow_paths(policy, repo_root=root)[1:]:
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


def workflow_path_for_job(job_id: str, *, policy: WorkflowPolicy | None = None) -> Path:
    """Resolve one registered job to its exclusive executable owner.

    Parameters
    ----------
    job_id
        Exact job identifier in the versioned repository workflow policy.
    policy
        Explicit policy being reviewed, or the current repository policy.
        The selected workflow must contain the requested executable job.

    Returns
    -------
    Path
        Existing workflow below the canonical repository root.

    Raises
    ------
    KeyError
        If no owner is registered for the job.
    ValueError
        If ownership is duplicated, escapes the repository, or the declared
        workflow does not execute the job. No first-match fallback is allowed.

    """
    resolved = load_ci_workflow_policy() if policy is None else policy
    return resolve_ci_workflow_owner(job_id, repo_root=REPOSITORY_ROOT, policy=resolved)


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
