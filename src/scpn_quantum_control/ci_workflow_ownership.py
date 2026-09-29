# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — executable CI workflow ownership
"""Resolve source-bound CI ownership for package and repository consumers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path


def read_ci_workflow_policy(path: Path) -> dict[str, object]:
    """Read ownership declarations without allowing duplicate-key replacement.

    Parameters
    ----------
    path
        Actual selected checkout's policy file; reads only local bytes.

    Returns
    -------
    dict[str, object]
        Decoded policy, subsequently validated by the executable-owner resolver.

    Raises
    ------
    ValueError
        For malformed JSON, duplicate fields or a non-object policy.
    OSError
        If the selected policy cannot be read.

    """
    payload: object = json.loads(path.read_bytes(), object_pairs_hook=_unique_fields)
    if not isinstance(payload, dict):
        raise ValueError("CI workflow policy must be a JSON object")
    return payload


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate CI policy field: {key}")
        result[key] = value
    return result


def read_ci_job_blocks(workflow: str) -> dict[str, str]:
    """Extract executable top-level jobs without rewriting their commands.

    Parameters
    ----------
    workflow
        Actual workflow YAML source in the repository's two-space job format.

    Returns
    -------
    dict[str, str]
        Job identifiers and unchanged source blocks, in declaration order.

    Raises
    ------
    ValueError
        If a job is declared twice; neither declaration becomes an owner.

    """
    lines = workflow.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    jobs_seen = False
    jobs_end = len(lines)
    for index, line in enumerate(lines):
        if line.rstrip("\n") == "jobs:":
            jobs_seen = True
            continue
        if jobs_seen and re.match(r"^[^\s#]", line):
            # A later top-level key closes the jobs mapping; it is not job source.
            jobs_end = index
            break
        match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if jobs_seen and match:
            starts.append((index, match.group(1)))
    blocks: dict[str, str] = {}
    for position, (start, job_id) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else jobs_end
        if job_id in blocks:
            raise ValueError(f"CI job appears multiple times in one workflow: {job_id}")
        blocks[job_id] = "".join(lines[start:end]).strip("\n")
    return blocks


def resolve_ci_workflow_owner(
    job_id: str,
    *,
    repo_root: Path,
    policy: Mapping[str, object],
) -> Path:
    """Resolve a registered job to exactly one existing executable workflow.

    Parameters
    ----------
    job_id
        Exact job identifier; category and required-gate ownership are exclusive.
    repo_root
        Checkout against which relative paths and symlink containment are checked.
    policy
        Decoded version-one repository workflow policy. No file is modified.

    Returns
    -------
    Path
        Resolved workflow path containing the job's actual source block.

    Raises
    ------
    KeyError
        If the requested job has no declared owner.
    ValueError
        For an unsupported policy, malformed declaration, ambiguous ownership,
        unsafe path, absent executable job or duplicate workflow job.
    OSError
        If the declared workflow cannot be read.

    """
    categories = policy.get("categories")
    if type(policy.get("schema_version")) is not int or policy.get("schema_version") != 1:
        raise ValueError("unknown CI ownership policy")
    if not isinstance(categories, list):
        raise ValueError("unknown CI ownership policy")
    owners: list[object] = []
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("CI category must be a string-keyed object")
        jobs = category.get("jobs")
        if not isinstance(jobs, list) or any(not isinstance(item, str) for item in jobs):
            raise ValueError("CI jobs must be a string list")
        owners.extend(category.get("workflow") for item in jobs if item == job_id)
    if policy.get("required_gate") == job_id:
        owners.append(policy.get("coordinator"))
    if not owners:
        raise KeyError(job_id)
    if len(owners) != 1:
        raise ValueError(f"job requires exactly one CI owner: {job_id}")
    owner = owners[0]
    if not isinstance(owner, str) or not owner.strip() or owner != owner.strip():
        raise ValueError("CI workflow must be a nonempty canonical string")
    root = repo_root.resolve()
    relative = Path(owner)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not candidate.is_relative_to(root):
        raise ValueError("CI owner must be repository-relative")
    if not candidate.is_file() or job_id not in read_ci_job_blocks(candidate.read_text("utf-8")):
        raise ValueError(f"CI owner does not execute qualification job {job_id}: {relative}")
    return candidate
