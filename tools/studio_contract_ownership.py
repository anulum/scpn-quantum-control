# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared contract ownership and conformance
"""Validate required shared contract consumers and their executable CI owner."""

from __future__ import annotations

import ast
import shlex
from pathlib import Path
from typing import TypedDict, cast

import yaml

from scpn_quantum_control.ci_workflow_ownership import (
    read_ci_workflow_policy,
    resolve_ci_workflow_owner,
)

_REQUIRED = {"workspace": {"python", "typescript"}, "program-source": {"python", "rust", "wasm"}}
_OWNER_JOB = "studio-web"
_STEP_ID = "shared-contract-conformance"
_COMMAND = ["python", "-m", "tools.studio_contract_quality_gates", "--run"]


class ContractConsumer(TypedDict):
    """Actual language consumer, dedicated test command and native source docs."""

    id: str
    source: list[str]
    tests: list[str]
    native_docs: list[str]
    command: list[str]
    cwd: str


class ContractCohort(TypedDict):
    """One immutable shared fixture family and all existing language consumers."""

    id: str
    fixtures: list[str]
    consumers: list[ContractConsumer]


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("contract declaration must be an object")
    return cast(dict[str, object], value)


def _strings(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError("nonempty contract string list required")
    return cast(list[str], value)


def _path(root: Path, value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("canonical repository path required")
    path = Path(value)
    target = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not target.is_relative_to(root):
        raise ValueError("contract path must remain repository-relative")
    exists = target.is_dir() if directory else target.is_file()
    if not exists:
        raise ValueError(f"missing contract owner: {value}")
    return target


def load_contract_cohorts(*, repo_root: Path | None = None) -> list[ContractCohort]:
    """Load all mandatory actual consumers without silently omitting a language.

    Parameters
    ----------
    repo_root
        Selected local checkout, defaulting to this tool's repository.

    Returns
    -------
    list[ContractCohort]
        Version-one fixture families with validated source, docs, test and cwd paths.

    Raises
    ------
    ValueError
        For unknown versions, missing/duplicate cohorts or consumers, unsafe paths,
        missing owners, malformed commands or documentation not bound to source.
    OSError
        If the local registry cannot be read. No files or processes are changed.

    """
    root = (Path(__file__).resolve().parents[1] if repo_root is None else repo_root).resolve()
    payload = read_ci_workflow_policy(root / "tools/studio_contract_policy.json")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise ValueError("unsupported shared contract policy version")
    rows = payload.get("cohorts")
    if not isinstance(rows, list):
        raise ValueError("required contract cohorts absent")
    seen: set[str] = set()
    for raw in rows:
        row = _mapping(raw)
        identity = row.get("id")
        if not isinstance(identity, str) or identity not in _REQUIRED or identity in seen:
            raise ValueError("missing, unknown or duplicate contract cohort")
        seen.add(identity)
        for fixture in _strings(row.get("fixtures")):
            _path(root, fixture)
        consumers = row.get("consumers")
        if not isinstance(consumers, list):
            raise ValueError("required contract consumers absent")
        languages: set[str] = set()
        for raw_consumer in consumers:
            consumer = _mapping(raw_consumer)
            language = consumer.get("id")
            if not isinstance(language, str) or language in languages:
                raise ValueError("duplicate or invalid contract consumer")
            languages.add(language)
            sources = _strings(consumer.get("source"))
            docs = _strings(consumer.get("native_docs"))
            tests = _strings(consumer.get("tests"))
            command = _strings(consumer.get("command"))
            for value in [*sources, *docs, *tests]:
                _path(root, value)
            if set(docs) != set(sources):
                raise ValueError("each source owner needs native documentation")
            _path(root, consumer.get("cwd"), directory=True)
            executable = {
                "python": "{python}",
                "typescript": "node",
                "rust": "cargo",
                "wasm": "node",
            }
            if language not in executable or command[0] != executable[language]:
                raise ValueError("contract command does not execute its actual language")
            if "--passWithNoTests" in command:
                raise ValueError("required consumer cannot pass without its tests")
            if language == "rust":
                targets = [Path(test).stem for test in tests]
                if command[1:4] != ["test", "--locked", "--test"] or command[4:5] != targets:
                    raise ValueError("required Rust test target absent from command")
            else:
                targets = [
                    Path(test).relative_to(Path(str(consumer["cwd"]))).as_posix() for test in tests
                ]
                prefix = (
                    ["-m", "pytest"]
                    if language == "python"
                    else ["node_modules/vitest/vitest.mjs", "run"]
                )
                if command[1:3] != prefix or any(target not in command for target in targets):
                    raise ValueError("required corpus test owner absent from command")
        if languages != _REQUIRED[identity]:
            raise ValueError(f"required language cohort absent: {identity}")
    if seen != set(_REQUIRED):
        raise ValueError("required contract cohort absent")
    return cast(list[ContractCohort], rows)


def _jobs(path: Path) -> dict[str, object]:
    payload = _mapping(yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader))
    return _mapping(payload.get("jobs"))


def _mandatory(job: dict[str, object]) -> None:
    if "if" in job or job.get("continue-on-error", "false") != "false":
        raise ValueError("required contract owner must not be conditional or error-tolerant")


def _aggregate_refuses_failure(run: object) -> bool:
    if not isinstance(run, str):
        return False
    lines = run.splitlines()
    if len(lines) < 3 or lines[0] != "python - <<'PY'" or lines[-1] != "PY":
        return False
    try:
        tree = ast.parse("\n".join(lines[1:-1]))
    except SyntaxError:
        return False
    # Bind the whole executable gate, including its nonzero refusal and inputs.
    # A matching predicate alone can be bypassed by overwriting its state.
    expected = ast.parse(
        "import json\n"
        "import os\n"
        'results = json.loads(os.environ["CATEGORY_RESULTS"])\n'
        'failures = {name: value["result"] for name, value in results.items() '
        'if value["result"] != "success"}\n'
        "if failures:\n"
        '    raise SystemExit(f"CI category gate failed: {failures}")\n'
        'print("CI category gate passed")\n'
    )
    return ast.dump(tree) == ast.dump(expected)


def validate_contract_workflow(*, repo_root: Path | None = None) -> None:
    """Require the actual conformance runner to feed the existing fail-closed gate.

    Parameters
    ----------
    repo_root
        Local checkout to inspect; defaults to this tool's repository.

    Raises
    ------
    ValueError
        If the consumer job, category, executable step or aggregate can omit or
        tolerate conformance failure.
    KeyError
        If exclusive CI ownership is unregistered.
    OSError
        If an actual local workflow or policy is missing. No file is modified.

    """
    root = (Path(__file__).resolve().parents[1] if repo_root is None else repo_root).resolve()
    policy = read_ci_workflow_policy(root / "tools/ci_workflow_policy.json")
    owner = resolve_ci_workflow_owner(_OWNER_JOB, repo_root=root, policy=policy)
    job = _mapping(_jobs(owner).get(_OWNER_JOB))
    _mandatory(job)
    categories = cast(list[dict[str, object]], policy["categories"])
    category = next(row for row in categories if _OWNER_JOB in cast(list[str], row["jobs"]))
    coordinator = _jobs(_path(root, policy.get("coordinator")))
    caller = _mapping(coordinator.get(str(category["id"])))
    _mandatory(caller)
    if caller.get("uses") != "./" + str(category["workflow"]):
        raise ValueError("contract category must call its actual executable owner")
    if _OWNER_JOB in cast(list[str], policy["optional_jobs"]):
        raise ValueError("required contract cohort cannot be optional")
    steps = job.get("steps")
    if not isinstance(steps, list):
        raise ValueError("required conformance step absent")
    selected = [
        _mapping(step) for step in steps if isinstance(step, dict) and step.get("id") == _STEP_ID
    ]
    if len(selected) != 1:
        raise ValueError("required conformance step absent or duplicated")
    step = selected[0]
    _mandatory(step)
    if step.get("working-directory") != "." or not isinstance(step.get("run"), str):
        raise ValueError("conformance must execute at the repository root")
    if shlex.split(cast(str, step["run"]), comments=True) != _COMMAND:
        raise ValueError("required conformance command missing or bypassed")
    gate = _mapping(coordinator.get(str(policy["required_gate"])))
    if gate.get("needs") != [row["id"] for row in categories] or gate.get("if") != "always()":
        raise ValueError("required aggregate must include every category")
    if gate.get("continue-on-error", "false") != "false":
        raise ValueError("required aggregate cannot tolerate failure")
    gate_steps = gate.get("steps")
    if not isinstance(gate_steps, list) or not any(
        isinstance(item, dict)
        and _aggregate_refuses_failure(item.get("run"))
        and _mapping(item.get("env")).get("CATEGORY_RESULTS") == "${{ toJSON(needs) }}"
        and "if" not in item
        and item.get("continue-on-error", "false") == "false"
        for item in gate_steps
    ):
        raise ValueError("required aggregate must refuse actual category failure")


__all__ = [
    "ContractCohort",
    "ContractConsumer",
    "load_contract_cohorts",
    "validate_contract_workflow",
]
