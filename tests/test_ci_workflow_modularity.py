# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CI Workflow Modularity Tests
"""Exercise the distributed CI inventory and fail-closed GodFile guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_ci_workflow_modularity as modularity
from tools import ci_workflow_inventory as inventory
from tools.ci_workflow_inventory import WorkflowCategory, WorkflowPolicy

_ACTION_SHA = "0123456789abcdef0123456789abcdef01234567"


def _policy() -> WorkflowPolicy:
    """Return a one-category policy suitable for isolated mutation tests."""
    return {
        "schema_version": 1,
        "coordinator": ".github/workflows/ci.yml",
        "required_gate": "ci-gate",
        "limits": {
            "coordinator_max_lines": 80,
            "coordinator_max_bytes": 8_192,
            "reusable_max_lines": 80,
            "reusable_max_bytes": 8_192,
            "max_reusable_workflows": 4,
        },
        "categories": [
            {
                "id": "unit-quality",
                "workflow": ".github/workflows/ci-unit-quality.yml",
                "caller_needs": [],
                "jobs": ["unit"],
            }
        ],
        "job_order": ["unit"],
        "optional_jobs": [],
    }


def _write_fixture(root: Path, policy: WorkflowPolicy) -> None:
    """Write one valid coordinator/category pair below ``root``."""
    workflow_root = root / ".github" / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)
    (workflow_root / "ci.yml").write_text(
        """name: CI
on:
  push:
jobs:
  unit-quality:
    uses: ./.github/workflows/ci-unit-quality.yml
  ci-gate:
    needs: [unit-quality]
    runs-on: ubuntu-latest
    if: always()
    steps:
      - env:
          CATEGORY_RESULTS: ${{ toJSON(needs) }}
        run: |
          failures = {name: value["result"] for name, value in results.items() if value["result"] != "success"}
""",
        encoding="utf-8",
    )
    (workflow_root / "ci-unit-quality.yml").write_text(
        f"""name: CI / Unit Quality
on:
  workflow_call:
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@{_ACTION_SHA}
""",
        encoding="utf-8",
    )
    (root / "tools").mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    (root / "tools" / "ci_workflow_policy.json").write_text(json.dumps(policy), encoding="utf-8")


def test_live_inventory_is_complete_unique_and_bounded() -> None:
    """Require the real coordinator and every reusable category to pass."""
    policy = inventory.load_ci_workflow_policy()
    jobs = [job for category in policy["categories"] for job in category["jobs"]]

    assert len(policy["categories"]) == 20
    assert len(jobs) == len(set(jobs)) == len(policy["job_order"]) == 189
    assert set(jobs) == set(policy["job_order"])
    assert modularity.audit_ci_workflow_modularity(policy) == []
    assert modularity.main() == 0


def test_inventory_reconstructs_real_jobs_and_resolves_owners() -> None:
    """Expose physical category ownership through one ordered compatibility view."""
    source = inventory.read_ci_workflow_source()
    policy = inventory.load_ci_workflow_policy()

    assert source.count("  lint:\n") == 1
    assert source.count("  ci-gate:\n") == 1
    assert source.index("  lint:\n") < source.index("  native-wheels:\n")
    assert inventory.workflow_path_for_job("lint").name == "ci-static-analysis.yml"
    assert inventory.workflow_path_for_job("ci-gate").name == "ci.yml"
    assert inventory.ci_workflow_paths(policy)[0].name == "ci.yml"
    with pytest.raises(KeyError):
        inventory.workflow_path_for_job("unknown-job")


def test_inventory_rejects_non_object_duplicate_and_missing_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed on malformed policy, duplicate jobs, and absent gate ownership."""
    policy = _policy()
    _write_fixture(tmp_path, policy)
    policy_path = tmp_path / "tools" / "ci_workflow_policy.json"
    monkeypatch.setattr(inventory, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(inventory, "CI_WORKFLOW_POLICY", policy_path)

    policy_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        inventory.load_ci_workflow_policy()
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    category.write_text(
        category.read_text(encoding="utf-8") + "  unit:\n    runs-on: ubuntu-latest\n"
    )
    with pytest.raises(ValueError, match="multiple times in one workflow"):
        inventory.read_ci_workflow_source()

    _write_fixture(tmp_path, policy)
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text("name: CI\non:\n  push:\njobs:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required gate"):
        inventory.read_ci_workflow_source()

    _write_fixture(tmp_path, policy)
    policy["job_order"] = ["unit", "absent"]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="missing jobs"):
        inventory.read_ci_workflow_source()


def test_modularity_audit_reports_policy_size_and_coordinator_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unsupported schemas, excessive counts, size growth, and call drift."""
    policy = _policy()
    _write_fixture(tmp_path, policy)
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", tmp_path)
    policy["schema_version"] = 2
    policy["limits"]["max_reusable_workflows"] = 0
    policy["limits"]["coordinator_max_lines"] = 1
    policy["limits"]["coordinator_max_bytes"] = 1
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text(
        coordinator.read_text(encoding="utf-8")
        .replace("uses: ./.github/workflows/ci-unit-quality.yml", "uses: ./wrong.yml")
        .replace("needs: [unit-quality]", "needs: []")
        .replace("if: always()", "if: success()")
        .replace("toJSON(needs)", "needs")
        .replace('value["result"] != "success"', "False"),
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("schema_version" in error for error in errors)
    assert any("reusable workflow count" in error for error in errors)
    assert any("lines exceed" in error for error in errors)
    assert any("bytes exceed" in error for error in errors)
    assert any("targets the wrong workflow" in error for error in errors)
    assert any("does not aggregate" in error for error in errors)
    assert any("if: always" in error for error in errors)
    assert any("fail closed" in error for error in errors)


def test_modularity_audit_reports_category_ownership_and_security_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject direct triggers, oversize categories, cross-needs, and mutable actions."""
    policy = _policy()
    _write_fixture(tmp_path, policy)
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", tmp_path)
    policy["limits"]["reusable_max_lines"] = 1
    policy["limits"]["reusable_max_bytes"] = 1
    policy["categories"][0]["jobs"] = ["expected"]
    category = tmp_path / ".github/workflows/ci-unit-quality.yml"
    category.write_text(
        category.read_text(encoding="utf-8")
        .replace("workflow_call:", "push:")
        .replace("jobs:", "permissions:\n  contents: write\njobs:")
        .replace("runs-on: ubuntu-latest", "needs: foreign\n    runs-on: ubuntu-latest")
        .replace(f"actions/checkout@{_ACTION_SHA}", "actions/checkout@main"),
        encoding="utf-8",
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("reusable category" in error for error in errors)
    assert any("read-only contents permission" in error for error in errors)
    assert any("lines exceed" in error for error in errors)
    assert any("bytes exceed" in error for error in errors)
    assert any("job order/ownership" in error for error in errors)
    assert any("cross-category needs" in error for error in errors)
    assert any("unpinned action" in error for error in errors)
    assert any("incomplete or contains undeclared" in error for error in errors)


def test_modularity_audit_rejects_duplicate_jobs_and_direct_coordinator_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject shared job ownership and renewed test coupling to the coordinator."""
    policy = _policy()
    duplicate: WorkflowCategory = {
        "id": "duplicate-quality",
        "workflow": ".github/workflows/ci-duplicate-quality.yml",
        "caller_needs": [],
        "jobs": ["unit"],
    }
    policy["categories"].append(duplicate)
    policy["optional_jobs"] = ["absent"]
    _write_fixture(tmp_path, policy)
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", tmp_path)
    source = tmp_path / ".github/workflows/ci-unit-quality.yml"
    (tmp_path / duplicate["workflow"]).write_text(source.read_text(encoding="utf-8"))
    coordinator = tmp_path / ".github/workflows/ci.yml"
    coordinator.write_text(
        coordinator.read_text(encoding="utf-8").replace(
            "  ci-gate:",
            "  duplicate-quality:\n"
            "    uses: ./.github/workflows/ci-duplicate-quality.yml\n"
            "  ci-gate:",
        ),
        encoding="utf-8",
    )
    coordinator_literal = ".github/workflows/" + "ci.yml"
    (tmp_path / "tests/bad_reader.py").write_text(
        f'Path("{coordinator_literal}").read_text()\n', encoding="utf-8"
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("duplicated" in error for error in errors)
    assert any("distributed CI inventory" in error for error in errors)
    assert any("optional CI job inventory" in error for error in errors)


def test_modularity_audit_rejects_unregistered_categories_and_unguarded_optional_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject policy bypass files and optional jobs without explicit semantics."""
    policy = _policy()
    policy["optional_jobs"] = ["unit"]
    _write_fixture(tmp_path, policy)
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / ".github/workflows/ci-unregistered.yml").write_text(
        "name: Undeclared\non:\n  workflow_call:\njobs: {}\n", encoding="utf-8"
    )

    errors = modularity.audit_ci_workflow_modularity(policy)

    assert any("physical CI category workflows" in error for error in errors)
    assert any("optional job unit lacks" in error for error in errors)


def test_modularity_audit_fails_loudly_on_non_mapping_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject syntactically valid YAML that is not a workflow mapping."""
    policy = _policy()
    _write_fixture(tmp_path, policy)
    monkeypatch.setattr(modularity, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / ".github/workflows/ci-unit-quality.yml").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="workflow must be a mapping"):
        modularity.audit_ci_workflow_modularity(policy)
