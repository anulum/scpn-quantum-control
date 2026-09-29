# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — exclusive CI workflow ownership tests
"""Resolve actual repository jobs and reject ambiguous ownership declarations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import pytest

from tools.ci_workflow_inventory import (
    ci_workflow_paths,
    load_ci_workflow_policy,
    read_ci_workflow_source,
    workflow_path_for_job,
)


def test_real_inventory_preserves_job_commands_and_required_dependencies() -> None:
    """The compatibility view preserves executable commands and all gate owners."""
    policy = load_ci_workflow_policy()
    view = read_ci_workflow_source()
    paths = ci_workflow_paths(policy)
    assert ci_workflow_paths() == paths
    assert paths[0] == workflow_path_for_job(policy["required_gate"])
    assert len(paths) == len(policy["categories"]) + 1
    ordered = [view.index(f"  {job}:\n") for job in policy["job_order"]]
    assert ordered == sorted(ordered)
    gate = view[view.index(f"  {policy['required_gate']}:\n") :]
    assert f"    needs: [{', '.join(policy['job_order'])}]" in gate
    for job in policy["job_order"]:
        actual = workflow_path_for_job(job).read_text()
        start = actual.index(f"  {job}:\n")
        tail = actual[start:]
        lines = tail.splitlines()
        end = next(
            (
                i
                for i, line in enumerate(lines[1:], 1)
                if line.startswith("  ") and not line.startswith("   ")
            ),
            len(lines),
        )
        for line in lines[:end]:
            if "run:" in line:
                assert line in view
        if job != "lint":
            block = view[view.index(f"  {job}:\n") :]
            assert block.splitlines()[1] == "    needs: lint" or "lint" in block.splitlines()[1]


@pytest.mark.parametrize(
    "corruption", ("non-object", "missing-gate", "missing-job", "duplicate-job")
)
def test_real_inventory_rejects_corrupted_checkout(tmp_path: Path, corruption: str) -> None:
    """Corrupt copies of actual workflow files cannot produce a complete compatibility view."""
    from tools.ci_workflow_inventory import REPOSITORY_ROOT

    policy = load_ci_workflow_policy()
    for original in ci_workflow_paths(policy):
        target = tmp_path / original.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(original, target)
    policy_path = tmp_path / "tools/ci_workflow_policy.json"
    policy_path.parent.mkdir()
    copyfile(REPOSITORY_ROOT / "tools/ci_workflow_policy.json", policy_path)
    first, second = policy["categories"][:2]
    if corruption == "non-object":
        policy_path.write_text("[]")
    elif corruption == "missing-gate":
        path = tmp_path / policy["coordinator"]
        path.write_text(
            path.read_text().replace(f"  {policy['required_gate']}:\n", "  retired-gate:\n")
        )
    else:
        path = tmp_path / first["workflow"]
        job = first["jobs"][0]
        text = path.read_text()
        if corruption == "missing-job":
            path.write_text(text.replace(f"  {job}:\n", f"  {job}-retired:\n"))
        else:
            lines = text[text.index(f"  {job}:\n") :].splitlines(keepends=True)
            end = next(
                (
                    i
                    for i, line in enumerate(lines[1:], 1)
                    if line.startswith("  ") and not line.startswith("   ")
                ),
                len(lines),
            )
            target = tmp_path / second["workflow"]
            target.write_text(target.read_text() + "\n" + "".join(lines[:end]))
    with pytest.raises(
        ValueError, match="JSON object|missing required gate|missing jobs|multiple reusable"
    ):
        read_ci_workflow_source(repo_root=tmp_path)


def test_actual_ci_jobs_have_exclusive_executable_owners() -> None:
    """Every declared job resolves to its real executable category workflow."""
    policy = load_ci_workflow_policy()
    for category in policy["categories"]:
        for job in category["jobs"]:
            path = workflow_path_for_job(job, policy=policy)
            assert path.is_file()
            assert path.as_posix().endswith(category["workflow"])
            assert f"  {job}:" in path.read_text()
    assert workflow_path_for_job(policy["required_gate"], policy=policy).is_file()


def test_duplicate_ci_category_owner_is_refused() -> None:
    """A duplicated real category cannot qualify the first matching owner."""
    policy = deepcopy(load_ci_workflow_policy())
    category = policy["categories"][0]
    policy["categories"].append(deepcopy(category))
    with pytest.raises(ValueError, match="exactly one CI owner"):
        workflow_path_for_job(category["jobs"][0], policy=policy)


def test_duplicate_ci_job_in_one_category_is_refused() -> None:
    """Repeated declarations in one workflow are also ambiguous ownership."""
    policy = deepcopy(load_ci_workflow_policy())
    category = policy["categories"][0]
    job = category["jobs"][0]
    category["jobs"].append(job)
    with pytest.raises(ValueError, match="exactly one CI owner"):
        workflow_path_for_job(job, policy=policy)


def test_ci_gate_cannot_also_belong_to_a_category() -> None:
    """The aggregate gate and a reusable category cannot share ownership."""
    policy = deepcopy(load_ci_workflow_policy())
    policy["categories"][0]["jobs"].append(policy["required_gate"])
    with pytest.raises(ValueError, match="exactly one CI owner"):
        workflow_path_for_job(policy["required_gate"], policy=policy)


def test_unknown_ci_job_is_not_advertised() -> None:
    """A domain cannot invent a job that has no registered CI owner."""
    with pytest.raises(KeyError):
        workflow_path_for_job("unregistered-domain-qualification")


def test_declared_owner_must_execute_the_requested_job() -> None:
    """A real workflow for another category is not evidence for this job."""
    policy = deepcopy(load_ci_workflow_policy())
    first, second = policy["categories"][:2]
    first["workflow"] = second["workflow"]
    with pytest.raises(ValueError, match="does not execute"):
        workflow_path_for_job(first["jobs"][0], policy=policy)


@pytest.mark.parametrize("path", ("../outside.yml", "/tmp/outside.yml"))
def test_ci_owner_path_must_remain_in_the_repository(path: str) -> None:
    """Out-of-repository ownership is refused before reading another file."""
    policy = deepcopy(load_ci_workflow_policy())
    category = policy["categories"][0]
    category["workflow"] = path
    with pytest.raises(ValueError, match="repository-relative"):
        workflow_path_for_job(category["jobs"][0], policy=policy)
