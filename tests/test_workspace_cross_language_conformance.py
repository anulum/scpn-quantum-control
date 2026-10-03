# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared contract ownership and conformance
"""Bind real shared corpus consumers to mandatory executable CI ownership."""

from __future__ import annotations

import ast
import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from tools.ci_workflow_inventory import (
    REPOSITORY_ROOT,
    ci_workflow_paths,
    load_ci_workflow_policy,
    load_contract_cohorts,
    validate_contract_workflow,
)

_POLICY = "tools/studio_contract_policy.json"


def _checkout(root: Path) -> Path:
    """Copy actual workflow declarations and named contract owners for mutations."""
    root.mkdir()
    paths = {Path(_POLICY), Path("tools/ci_workflow_policy.json")}
    paths.update(p.relative_to(REPOSITORY_ROOT) for p in ci_workflow_paths())
    for row in load_contract_cohorts():
        paths.update(Path(p) for p in row["fixtures"])
        for consumer in row["consumers"]:
            paths.update(Path(p) for p in [*consumer["source"], *consumer["tests"]])
    for p in paths:
        target = root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / p, target)
    return root


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _yaml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.load(path.read_text(), Loader=yaml.BaseLoader))


def test_required_contract_cohorts() -> None:
    """Every actual language consumer resolves through the existing public CI facade."""
    assert {row["id"]: {c["id"] for c in row["consumers"]} for row in load_contract_cohorts()} == {
        "workspace": {"python", "typescript"},
        "program-source": {"python", "rust", "wasm"},
    }
    validate_contract_workflow()


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "boolean-version",
        "cohorts-shape",
        "missing-cohort",
        "duplicate-cohort",
        "unknown-cohort",
        "cohort-shape",
        "consumers-shape",
        "missing-language",
        "duplicate-language",
        "invalid-language",
        "unknown-language",
        "no-fixture",
        "empty-list",
        "invalid-list",
        "missing-file",
        "absolute",
        "traversal",
        "empty-path",
        "space-path",
        "no-docs",
        "missing-doc-owner",
        "command-language",
        "missing-cwd",
        "directory-as-file",
        "symlink-outside",
        "non-string-path",
        "consumer-shape",
    ],
)
def test_missing_cohort_refused(tmp_path: Path, mutation: str) -> None:
    """Mutate real declarations without launching a consumer or rewriting admitted inputs."""
    root = _checkout(tmp_path / "copy")
    path = root / _POLICY
    p = _json(path)
    row, c = p["cohorts"][0], p["cohorts"][0]["consumers"][0]
    changes: dict[str, tuple[dict[str, Any], str, object]] = {
        "version": (p, "schema_version", 2),
        "boolean-version": (p, "schema_version", True),
        "cohorts-shape": (p, "cohorts", {}),
        "unknown-cohort": (row, "id", "unknown"),
        "consumers-shape": (row, "consumers", {}),
        "invalid-language": (c, "id", None),
        "unknown-language": (c, "id", "unknown"),
        "empty-list": (c, "source", []),
        "invalid-list": (c, "tests", [3]),
        "empty-path": (c, "cwd", ""),
        "space-path": (c, "cwd", " . "),
        "no-docs": (c, "native_docs", []),
        "missing-doc-owner": (c, "native_docs", c["source"][:1]),
        "missing-cwd": (c, "cwd", "missing"),
        "non-string-path": (c, "cwd", 3),
    }
    if mutation in changes:
        target, key, value = changes[mutation]
        target[key] = value
    elif mutation == "missing-cohort":
        p["cohorts"].pop()
    elif mutation == "duplicate-cohort":
        p["cohorts"].append(deepcopy(row))
    elif mutation == "cohort-shape":
        p["cohorts"][0] = None
    elif mutation == "missing-language":
        row["consumers"].pop()
    elif mutation == "duplicate-language":
        row["consumers"].append(deepcopy(c))
    elif mutation == "no-fixture":
        row.pop("fixtures")
    elif mutation == "missing-file":
        (root / row["fixtures"][0]).unlink()
    elif mutation == "absolute":
        row["fixtures"][0] = str(REPOSITORY_ROOT / row["fixtures"][0])
    elif mutation == "traversal":
        row["fixtures"][0] = "../outside.json"
    elif mutation == "command-language":
        c["command"][0] = "echo"
    elif mutation == "directory-as-file":
        row["fixtures"][0] = "tools"
    elif mutation == "consumer-shape":
        row["consumers"][0] = None
    else:
        fixture = root / row["fixtures"][0]
        fixture.unlink()
        fixture.symlink_to(REPOSITORY_ROOT / row["fixtures"][0])
    path.write_text(json.dumps(p))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        load_contract_cohorts(repo_root=root)
    assert path.read_bytes() == before


def test_duplicate_policy_keys_refused(tmp_path: Path) -> None:
    """Duplicate JSON fields cannot replace required consumer declarations."""
    root = _checkout(tmp_path / "copy")
    (root / _POLICY).write_text('{"schema_version":1,"schema_version":1,"cohorts":[]}')
    with pytest.raises(ValueError, match="duplicate"):
        load_contract_cohorts(repo_root=root)


@pytest.mark.parametrize(
    "mutation", ["no-tests-fallback", "missing-test-argument", "wrong-rust-target"]
)
def test_command_cannot_omit_required_owner(tmp_path: Path, mutation: str) -> None:
    """Actual consumer commands cannot omit their dedicated corpus test owner."""
    root = _checkout(tmp_path / "copy")
    path = root / _POLICY
    policy = _json(path)
    if mutation == "wrong-rust-target":
        consumer = policy["cohorts"][1]["consumers"][1]
        consumer["command"][4] = "unrelated"
    else:
        consumer = policy["cohorts"][0]["consumers"][1]
        if mutation == "no-tests-fallback":
            consumer["command"].append("--passWithNoTests")
        else:
            consumer["command"].remove("src/shared/contracts/canonical.test.ts")
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="required"):
        load_contract_cohorts(repo_root=root)


@pytest.mark.parametrize(
    "mutation",
    [
        "job-if",
        "job-tolerant",
        "step-if",
        "step-tolerant",
        "missing-step",
        "duplicate-step",
        "no-steps",
        "wrong-cwd",
        "run-shape",
        "comment-only",
        "swallow-failure",
        "caller-if",
        "caller-tolerant",
        "caller-target",
        "optional-job",
        "aggregate-omit",
        "aggregate-if",
        "aggregate-tolerant",
        "aggregate-no-steps",
        "aggregate-no-run",
        "aggregate-bad-shell",
        "aggregate-syntax",
        "aggregate-predicate",
        "aggregate-no-raise",
        "aggregate-env",
        "aggregate-step-if",
        "aggregate-step-tolerant",
        "jobs-shape",
    ],
)
def test_detached_required_gate_refused(tmp_path: Path, mutation: str) -> None:
    """Actual YAML mutations cannot qualify a skipped, swallowed or unaggregated failure."""
    root = _checkout(tmp_path / "copy")
    policy = load_ci_workflow_policy(repo_root=root)
    coordinator_path = root / policy["coordinator"]
    coordinator = _yaml(coordinator_path)
    workflow_path = root / ".github/workflows/ci-studio.yml"
    workflow = _yaml(workflow_path)
    job = workflow["jobs"]["studio-web"]
    step = next(item for item in job["steps"] if item.get("id") == "shared-contract-conformance")
    caller = coordinator["jobs"]["studio"]
    gate = coordinator["jobs"][policy["required_gate"]]
    gate_step = gate["steps"][0]
    changes: dict[str, tuple[dict[str, Any], str, object]] = {
        "job-if": (job, "if", "false"),
        "step-if": (step, "if", "false"),
        "caller-if": (caller, "if", "false"),
        "job-tolerant": (job, "continue-on-error", "true"),
        "step-tolerant": (step, "continue-on-error", "true"),
        "caller-tolerant": (caller, "continue-on-error", "true"),
        "no-steps": (job, "steps", {}),
        "wrong-cwd": (step, "working-directory", "studio-web"),
        "run-shape": (step, "run", ["python"]),
        "comment-only": (step, "run", "# python -m tools.studio_contract_quality_gates --run"),
        "swallow-failure": (step, "run", step["run"] + " || true"),
        "caller-target": (caller, "uses", "./wrong.yml"),
        "aggregate-if": (gate, "if", "success()"),
        "aggregate-tolerant": (gate, "continue-on-error", "true"),
        "aggregate-no-steps": (gate, "steps", {}),
        "aggregate-bad-shell": (gate_step, "run", "echo green"),
        "aggregate-syntax": (gate_step, "run", "python - <<'PY'\ninvalid ! syntax\nPY"),
        "aggregate-predicate": (
            gate_step,
            "run",
            gate_step["run"].replace('value["result"] != "success"', "False"),
        ),
        "aggregate-no-raise": (
            gate_step,
            "run",
            gate_step["run"].replace("raise SystemExit", "print"),
        ),
        "aggregate-step-if": (gate_step, "if", "false"),
        "aggregate-step-tolerant": (gate_step, "continue-on-error", "true"),
        "jobs-shape": (workflow, "jobs", []),
    }
    if mutation in changes:
        target, key, value = changes[mutation]
        target[key] = value
    elif mutation == "missing-step":
        job["steps"].remove(step)
    elif mutation == "duplicate-step":
        job["steps"].append(deepcopy(step))
    elif mutation == "optional-job":
        policy["optional_jobs"].append("studio-web")
        (root / "tools/ci_workflow_policy.json").write_text(json.dumps(policy))
    elif mutation == "aggregate-omit":
        gate["needs"].remove("studio")
    elif mutation == "aggregate-no-run":
        gate_step.pop("run")
    else:
        gate_step["env"]["CATEGORY_RESULTS"] = "{}"
    for path, data in [(workflow_path, workflow), (coordinator_path, coordinator)]:
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    before = {p: p.read_bytes() for p in [workflow_path, coordinator_path]}
    with pytest.raises(ValueError):
        validate_contract_workflow(repo_root=root)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_wire_version_refused_by_real_consumers() -> None:
    """Real Python document consumers reject unknown major versions without mutation."""
    from scpn_quantum_control.studio_workspace import parse_document

    corpus = _json(REPOSITORY_ROOT / "tests/data/studio_workspace/documents.json")
    for name in ["workspace", "revision_root", "parameter", "settings", "run"]:
        payload = deepcopy(corpus["fixtures"][name])
        payload["schema"] = payload["schema"].rsplit(".v", 1)[0] + ".v999"
        before = deepcopy(payload)
        with pytest.raises(ValueError):
            parse_document(payload)
        assert payload == before


def test_public_contract_names_exclude_private_ids() -> None:
    """Production declarations retain descriptive public identities."""
    import tools.studio_contract_ownership as ownership
    import tools.studio_contract_quality_gates as quality

    private = re.compile(r"(?:CORE-|STUDIO-|ST-)\d|(?:^|_)Q0[1-7](?:_|$)")
    assert all(
        private.search(name) is None for module in [quality, ownership] for name in module.__all__
    )
    assert all(private.search(row["id"]) is None for row in load_contract_cohorts())


def test_source_owners_have_native_docs_and_dedicated_tests() -> None:
    """Each declared actual owner retains native docs and real behavioural test paths."""
    for row in load_contract_cohorts():
        for consumer in row["consumers"]:
            assert set(consumer["native_docs"]) == set(consumer["source"])
            for source in consumer["native_docs"]:
                text = (REPOSITORY_ROOT / source).read_text()
                if source.endswith(".py"):
                    assert ast.get_docstring(ast.parse(text))
                elif source.endswith(".rs"):
                    assert "//!" in text or "///" in text
                else:
                    assert "/**" in text
            assert all((REPOSITORY_ROOT / test).is_file() for test in consumer["tests"])


@pytest.mark.parametrize(
    "mutation", ["zero-exit", "empty-results", "overwrite-failures", "early-success"]
)
def test_aggregate_cannot_mask_actual_consumer_failure(tmp_path: Path, mutation: str) -> None:
    """Public workflow admission refuses a success-shaped aggregate after corpus failure."""
    root = _checkout(tmp_path / "copy")
    policy = load_ci_workflow_policy(repo_root=root)
    path = root / policy["coordinator"]
    workflow = _yaml(path)
    step = workflow["jobs"][policy["required_gate"]]["steps"][0]
    run = step["run"]
    if mutation == "zero-exit":
        run = run.replace(
            'raise SystemExit(f"CI category gate failed: {failures}")', "raise SystemExit(0)"
        )
    elif mutation == "empty-results":
        run = run.replace('results = json.loads(os.environ["CATEGORY_RESULTS"])', "results = {}")
    elif mutation == "overwrite-failures":
        run = run.replace("if failures:", "failures = {}\nif failures:")
    else:
        run = run.replace("import json", "raise SystemExit(0)\nimport json")
    step["run"] = run
    path.write_text(yaml.safe_dump(workflow, sort_keys=False))
    with pytest.raises(ValueError, match="aggregate"):
        validate_contract_workflow(repo_root=root)
