# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — additive strict-test-typing policy audit
"""Validate and execute the additive strict-mypy cohort for repository tests.

The whole test tree has measured legacy typing debt, so the repository uses an
additive cohort ratchet instead of an unsafe all-or-nothing configuration flip.
The tracked policy records the baseline, the enforced files, and the ordered
migration schedule. This audit rejects malformed or missing cohort paths and,
by default, runs strict mypy over the exact enforced set.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "tools" / "test_typing_policy.json"

CohortStatus = Literal["enforced", "planned", "deferred"]
_COHORT_STATUSES = frozenset({"enforced", "planned", "deferred"})
_REQUIRED_VERIFICATION = frozenset(
    {"focused pytest", "mypy --strict", "ruff check", "ruff format --check"}
)


@dataclass(frozen=True)
class DebtBaseline:
    """Measured whole-test-tree strict-mypy debt before this ratchet."""

    measured_at: str
    command: str
    tracked_python_files: int
    errors: int
    files_with_errors: int
    dominant_error_code: str
    dominant_error_count: int
    scope_note: str


@dataclass(frozen=True)
class MigrationRules:
    """Constraints for growing the enforced cohort."""

    additions_only: bool
    maximum_files_per_slice: int
    minimum_enforced_files: int
    required_verification: tuple[str, ...]
    invalid_input_policy: str


@dataclass(frozen=True)
class TestTypingCohort:
    """One ordered test-typing migration cohort."""

    order: int
    cohort_id: str
    status: CohortStatus
    selection: str
    activation_criterion: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class TestTypingPolicy:
    """Parsed strict-test-typing policy registry."""

    schema_version: int
    baseline: DebtBaseline
    rules: MigrationRules
    cohorts: tuple[TestTypingCohort, ...]

    @property
    def enforced_cohorts(self) -> tuple[TestTypingCohort, ...]:
        """Return cohorts currently executed by the strict gate."""
        return tuple(cohort for cohort in self.cohorts if cohort.status == "enforced")

    @property
    def enforced_paths(self) -> tuple[str, ...]:
        """Return the deterministic file set executed by strict mypy."""
        return tuple(path for cohort in self.enforced_cohorts for path in cohort.files)

    @property
    def registered_paths(self) -> tuple[str, ...]:
        """Return every concrete test path assigned to any cohort."""
        return tuple(path for cohort in self.cohorts for path in cohort.files)


@dataclass(frozen=True)
class AuditResult:
    """Policy, repository, and optional mypy execution result."""

    policy: TestTypingPolicy
    errors: tuple[str, ...]
    command: tuple[str, ...]
    mypy_returncode: int | None
    mypy_output: str

    @property
    def passed(self) -> bool:
        """Return whether registry validation and requested typing passed."""
        return not self.errors and self.mypy_returncode in {None, 0}


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return cast(list[object], value)


def _text(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _integer(mapping: dict[str, object], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _boolean(mapping: dict[str, object], key: str, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _string_sequence(mapping: dict[str, object], key: str, context: str) -> tuple[str, ...]:
    values = _sequence(mapping.get(key), f"{context}.{key}")
    parsed: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context}.{key}[{index}] must be a non-empty string")
        parsed.append(value.strip())
    return tuple(parsed)


def _parse_baseline(value: object) -> DebtBaseline:
    context = "registry.baseline"
    row = _mapping(value, context)
    tracked_python_files = _integer(row, "tracked_python_files", context)
    errors = _integer(row, "errors", context)
    files_with_errors = _integer(row, "files_with_errors", context)
    dominant_error_count = _integer(row, "dominant_error_count", context)
    if tracked_python_files <= 0:
        raise ValueError(f"{context}.tracked_python_files must be positive")
    if not 0 < files_with_errors <= tracked_python_files:
        raise ValueError(f"{context}.files_with_errors must be within the measured file count")
    if errors < files_with_errors:
        raise ValueError(f"{context}.errors cannot be below files_with_errors")
    if not 0 < dominant_error_count <= errors:
        raise ValueError(f"{context}.dominant_error_count must be within the error count")
    return DebtBaseline(
        measured_at=_text(row, "measured_at", context),
        command=_text(row, "command", context),
        tracked_python_files=tracked_python_files,
        errors=errors,
        files_with_errors=files_with_errors,
        dominant_error_code=_text(row, "dominant_error_code", context),
        dominant_error_count=dominant_error_count,
        scope_note=_text(row, "scope_note", context),
    )


def _parse_rules(value: object) -> MigrationRules:
    context = "registry.migration_rules"
    row = _mapping(value, context)
    additions_only = _boolean(row, "additions_only", context)
    maximum_files_per_slice = _integer(row, "maximum_files_per_slice", context)
    minimum_enforced_files = _integer(row, "minimum_enforced_files", context)
    required_verification = _string_sequence(row, "required_verification", context)
    if not additions_only:
        raise ValueError(f"{context}.additions_only must remain true")
    if maximum_files_per_slice <= 0:
        raise ValueError(f"{context}.maximum_files_per_slice must be positive")
    if minimum_enforced_files <= 0:
        raise ValueError(f"{context}.minimum_enforced_files must be positive")
    missing_verification = sorted(_REQUIRED_VERIFICATION - set(required_verification))
    if missing_verification:
        raise ValueError(
            f"{context}.required_verification misses required gates: {missing_verification}"
        )
    return MigrationRules(
        additions_only=additions_only,
        maximum_files_per_slice=maximum_files_per_slice,
        minimum_enforced_files=minimum_enforced_files,
        required_verification=required_verification,
        invalid_input_policy=_text(row, "invalid_input_policy", context),
    )


def _test_path(value: str, context: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be repository-relative")
    if len(path.parts) < 2 or path.parts[0] != "tests" or path.suffix != ".py":
        raise ValueError(f"{context} must name a Python file below tests/")
    return path.as_posix()


def _parse_cohort(value: object, index: int) -> TestTypingCohort:
    context = f"registry.cohorts[{index}]"
    row = _mapping(value, context)
    order = _integer(row, "order", context)
    cohort_id = _text(row, "id", context)
    if not cohort_id.replace("_", "").isalnum() or cohort_id.casefold() != cohort_id:
        raise ValueError(f"{context}.id must be a lowercase alphanumeric identifier")
    status_value = _text(row, "status", context)
    if status_value not in _COHORT_STATUSES:
        raise ValueError(f"{context}.status is unsupported: {status_value}")
    status = cast(CohortStatus, status_value)
    files = tuple(
        _test_path(path, f"{context}.files[{file_index}]")
        for file_index, path in enumerate(_string_sequence(row, "files", context))
    )
    if files != tuple(sorted(files)):
        raise ValueError(f"{context}.files must be sorted")
    if len(files) != len(set(files)):
        raise ValueError(f"{context}.files contains duplicates")
    if status == "enforced" and not files:
        raise ValueError(f"{context}: enforced cohorts must contain files")
    return TestTypingCohort(
        order=order,
        cohort_id=cohort_id,
        status=status,
        selection=_text(row, "selection", context),
        activation_criterion=_text(row, "activation_criterion", context),
        files=files,
    )


def parse_policy(payload: object) -> TestTypingPolicy:
    """Parse and validate a decoded strict-test-typing registry."""
    root = _mapping(payload, "registry")
    schema_version = _integer(root, "schema_version", "registry")
    if schema_version != 1:
        raise ValueError("registry.schema_version must be 1")
    rules = _parse_rules(root.get("migration_rules"))
    cohort_values = _sequence(root.get("cohorts"), "registry.cohorts")
    cohorts = tuple(_parse_cohort(value, index) for index, value in enumerate(cohort_values))
    if not cohorts:
        raise ValueError("registry.cohorts must not be empty")
    if tuple(cohort.order for cohort in cohorts) != tuple(range(len(cohorts))):
        raise ValueError("registry.cohorts orders must be consecutive from zero")
    cohort_ids = [cohort.cohort_id for cohort in cohorts]
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("registry.cohorts contains duplicate ids")
    statuses = tuple(cohort.status for cohort in cohorts)
    enforced_count = statuses.count("enforced")
    if enforced_count == 0 or statuses[:enforced_count] != ("enforced",) * enforced_count:
        raise ValueError("registry.cohorts must start with all enforced cohorts")
    status_rank = {"enforced": 0, "planned": 1, "deferred": 2}
    if tuple(status_rank[status] for status in statuses) != tuple(
        sorted(status_rank[status] for status in statuses)
    ):
        raise ValueError("registry.cohorts statuses must follow enforced, planned, deferred order")
    all_paths = [path for cohort in cohorts for path in cohort.files]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("registry.cohorts contains a path in more than one cohort")
    enforced_paths = tuple(path for cohort in cohorts[:enforced_count] for path in cohort.files)
    if len(enforced_paths) < rules.minimum_enforced_files:
        raise ValueError("enforced file count is below migration_rules.minimum_enforced_files")
    return TestTypingPolicy(
        schema_version=schema_version,
        baseline=_parse_baseline(root.get("baseline")),
        rules=rules,
        cohorts=cohorts,
    )


def load_policy(path: Path) -> TestTypingPolicy:
    """Load a UTF-8 JSON strict-test-typing registry."""
    return parse_policy(cast(object, json.loads(path.read_text(encoding="utf-8"))))


def tracked_test_paths(repo_root: Path) -> tuple[str, ...]:
    """Return Git-tracked paths below the repository test directory."""
    completed = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "-z", "--", "tests"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(sorted(path.decode("utf-8") for path in completed.stdout.split(b"\0") if path))


def audit_policy(policy: TestTypingPolicy, tracked_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Compare every registered cohort path with the tracked test inventory."""
    tracked = set(tracked_paths)
    return tuple(
        f"registered test is not tracked: {path}"
        for path in policy.registered_paths
        if path not in tracked
    )


def mypy_command(policy: TestTypingPolicy, python_executable: str) -> tuple[str, ...]:
    """Build the exact strict-mypy command for all enforced cohorts."""
    return (python_executable, "-m", "mypy", "--strict", *policy.enforced_paths)


def execute_mypy(command: tuple[str, ...], repo_root: Path) -> tuple[int, str]:
    """Execute the cohort type check and return its code and combined output."""
    completed = subprocess.run(  # nosec B603
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(
        text.rstrip() for text in (completed.stdout, completed.stderr) if text.rstrip()
    )
    return completed.returncode, output


def audit_repository(
    repo_root: Path,
    registry_path: Path,
    *,
    run_type_check: bool = True,
    python_executable: str = sys.executable,
) -> AuditResult:
    """Validate the registry and optionally execute its strict-mypy cohort."""
    policy = load_policy(registry_path)
    errors = audit_policy(policy, tracked_test_paths(repo_root))
    command = mypy_command(policy, python_executable)
    if errors or not run_type_check:
        return AuditResult(policy, errors, command, None, "")
    returncode, output = execute_mypy(command, repo_root)
    return AuditResult(policy, errors, command, returncode, output)


def format_result(result: AuditResult) -> str:
    """Render a deterministic human-readable audit summary."""
    if result.errors:
        return "test-typing policy audit failed:\n" + "\n".join(
            f"- {error}" for error in result.errors
        )
    mode = "validated; mypy skipped" if result.mypy_returncode is None else "strict mypy passed"
    if result.mypy_returncode not in {None, 0}:
        return "test-typing policy audit failed:\n" + result.mypy_output
    return (
        "test-typing policy current: "
        f"{len(result.policy.enforced_paths)} file(s) across "
        f"{len(result.policy.enforced_cohorts)} enforced cohort(s); {mode}; "
        f"baseline {result.policy.baseline.errors} errors in "
        f"{result.policy.baseline.files_with_errors} file(s)"
    )


def _json_result(result: AuditResult) -> str:
    return json.dumps(
        {
            "baseline": {
                "errors": result.policy.baseline.errors,
                "files_with_errors": result.policy.baseline.files_with_errors,
                "tracked_python_files": result.policy.baseline.tracked_python_files,
            },
            "command": list(result.command),
            "enforced_cohorts": [cohort.cohort_id for cohort in result.policy.enforced_cohorts],
            "enforced_files": list(result.policy.enforced_paths),
            "errors": list(result.errors),
            "mypy_returncode": result.mypy_returncode,
            "status": "pass" if result.passed else "fail",
        },
        indent=2,
        sort_keys=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the registry and tracked paths without invoking mypy",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the additive strict-test-typing audit."""
    args = _parser().parse_args(argv)
    try:
        result = audit_repository(
            args.repo_root.resolve(),
            args.registry.resolve(),
            run_type_check=not args.validate_only,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        if args.json:
            print(json.dumps({"errors": [str(exc)], "status": "error"}, indent=2, sort_keys=True))
        else:
            print(f"test-typing policy audit failed:\n- {exc}")
        return 2
    print(_json_result(result) if args.json else format_result(result))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
