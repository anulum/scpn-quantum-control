# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — executable CI ownership boundary tests
"""Exercise the canonical owner resolver against real copied workflow sources."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import pytest

from scpn_quantum_control.ci_workflow_ownership import (
    read_ci_job_blocks,
    read_ci_workflow_policy,
    resolve_ci_workflow_owner,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "tools/ci_workflow_policy.json"


@pytest.mark.parametrize("corruption", ("duplicate", "non-object"))
def test_policy_reader_refuses_ambiguous_bytes(tmp_path: Path, corruption: str) -> None:
    """Duplicate fields in an actual policy copy cannot replace its version contract."""
    target = tmp_path / "policy.json"
    raw = POLICY.read_text()
    assert read_ci_workflow_policy(POLICY)["schema_version"] == 1
    if corruption == "duplicate":
        raw = raw.replace('"schema_version": 1', '"schema_version": 2, "schema_version": 1')
    else:
        raw = "[]"
    target.write_text(raw)
    with pytest.raises(ValueError, match="duplicate CI policy|JSON object"):
        read_ci_workflow_policy(target)


def _checkout(tmp_path: Path) -> tuple[dict[str, object], str, str]:
    """Copy executable repository workflows while retaining the original policy."""
    policy: dict[str, object] = json.loads(POLICY.read_text())
    categories = policy["categories"]
    assert isinstance(categories, list)
    for category in categories:
        target = tmp_path / category["workflow"]
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / category["workflow"], target)
    coordinator = policy["coordinator"]
    assert isinstance(coordinator, str)
    copyfile(ROOT / coordinator, tmp_path / coordinator)
    return policy, categories[0]["jobs"][0], categories[0]["workflow"]


def test_all_registered_jobs_resolve_in_a_real_checkout(tmp_path: Path) -> None:
    """Every actual job and the aggregate gate retains its sole executable owner."""
    policy, _, _ = _checkout(tmp_path)
    categories = policy["categories"]
    assert isinstance(categories, list)
    for category in categories:
        for job in category["jobs"]:
            owner = resolve_ci_workflow_owner(job, repo_root=tmp_path, policy=policy)
            assert owner == (tmp_path / category["workflow"]).resolve()
            assert f"  {job}:\n" in owner.read_text()
    gate = policy["required_gate"]
    assert isinstance(gate, str)
    coordinator = policy["coordinator"]
    assert isinstance(coordinator, str)
    owner = resolve_ci_workflow_owner(gate, repo_root=tmp_path, policy=policy)
    assert owner == (tmp_path / coordinator).resolve()


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", True),
        ("schema_version", 2),
        ("categories", {}),
        ("categories", [None]),
        ("categories", [{"jobs": None}]),
        ("categories", [{"jobs": [1]}]),
    ),
)
def test_malformed_ownership_policy_is_refused(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Malformed declarations never inherit a former executable owner."""
    policy, job, _ = _checkout(tmp_path)
    policy[field] = value
    with pytest.raises(ValueError):
        resolve_ci_workflow_owner(job, repo_root=tmp_path, policy=policy)


@pytest.mark.parametrize("path", (None, "", " leading.yml", "../outside.yml", "/tmp/outside.yml"))
def test_invalid_owner_paths_are_refused(tmp_path: Path, path: object) -> None:
    """An owner must be canonical text and remain within the selected checkout."""
    policy, job, _ = _checkout(tmp_path)
    categories = policy["categories"]
    assert isinstance(categories, list)
    categories[0]["workflow"] = path
    with pytest.raises(ValueError):
        resolve_ci_workflow_owner(job, repo_root=tmp_path, policy=policy)


def test_duplicate_or_unknown_owner_cannot_resolve(tmp_path: Path) -> None:
    """First-match selection cannot hide repeated jobs or an invented job."""
    policy, job, _ = _checkout(tmp_path)
    with pytest.raises(KeyError):
        resolve_ci_workflow_owner(
            "unregistered-domain-qualification", repo_root=tmp_path, policy=policy
        )
    categories = policy["categories"]
    assert isinstance(categories, list)
    categories[0]["jobs"].append(job)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_ci_workflow_owner(job, repo_root=tmp_path, policy=policy)


@pytest.mark.parametrize("corruption", ("missing-file", "missing-job", "duplicate-job"))
def test_declared_workflow_requires_one_actual_job(tmp_path: Path, corruption: str) -> None:
    """File absence and edited executable declarations cannot retain ownership."""
    policy, job, path = _checkout(tmp_path)
    workflow = tmp_path / path
    if corruption == "missing-file":
        workflow.unlink()
    elif corruption == "missing-job":
        workflow.write_text(workflow.read_text().replace(f"  {job}:\n", f"  {job}-retired:\n"))
    else:
        text = workflow.read_text()
        start = text.index(f"  {job}:\n")
        tail = text[start:].splitlines(keepends=True)
        end = next(
            (
                i
                for i, line in enumerate(tail[1:], 1)
                if line.startswith("  ") and not line.startswith("   ")
            ),
            len(tail),
        )
        workflow.write_text(text + "\n" + "".join(tail[:end]))
    with pytest.raises(ValueError, match="does not execute|multiple times"):
        resolve_ci_workflow_owner(job, repo_root=tmp_path, policy=policy)


def test_symlink_owner_cannot_escape_the_checkout(tmp_path: Path) -> None:
    """A local-looking owner cannot resolve to an executable file outside its root."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    policy, job, path = _checkout(checkout)
    workflow = checkout / path
    external = tmp_path / "external.yml"
    copyfile(workflow, external)
    workflow.unlink()
    workflow.symlink_to(external)
    with pytest.raises(ValueError, match="repository-relative"):
        resolve_ci_workflow_owner(job, repo_root=checkout, policy=policy)


def test_job_blocks_end_at_the_next_top_level_key(tmp_path: Path) -> None:
    """Keys after the jobs mapping never extend or impersonate the final job."""
    policy, job, path = _checkout(tmp_path)
    workflow = tmp_path / path
    source = workflow.read_text()
    workflow.write_text(source.rstrip("\n") + "\nconcurrency:\n  impostor-job:\n    group: x\n")
    blocks = read_ci_job_blocks(workflow.read_text())
    assert "impostor-job" not in blocks
    assert all("concurrency:" not in block for block in blocks.values())
    assert blocks == read_ci_job_blocks(source)
    assert resolve_ci_workflow_owner(job, repo_root=tmp_path, policy=policy) == workflow.resolve()
    with pytest.raises(ValueError, match="does not execute"):
        resolve_ci_workflow_owner(
            "impostor-job",
            repo_root=tmp_path,
            policy={**policy, "categories": [{"workflow": path, "jobs": ["impostor-job"]}]},
        )
