# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — additive strict-test-typing policy audit tests
"""Tests for the machine-readable strict-test-typing cohort ratchet."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    """Load the actual audit owner under its isolated test module identity."""
    path = Path(__file__).resolve().parents[1] / "tools" / "audit_test_typing_policy.py"
    spec = importlib.util.spec_from_file_location("audit_test_typing_policy_for_tests", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test-typing policy audit from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_tool()


def _baseline() -> dict[str, object]:
    """Return a consistent historical-debt fixture for ten tracked test files."""
    return {
        "measured_at": "2026-07-13",
        "command": "mypy --strict tests/",
        "tracked_python_files": 10,
        "errors": 7,
        "files_with_errors": 3,
        "dominant_error_code": "no-untyped-def",
        "dominant_error_count": 5,
        "scope_note": "bounded test fixture",
    }


def _rules(minimum_enforced_files: int = 1) -> dict[str, object]:
    """Build additive migration rules with an explicit minimum enrolment.

    Parameters
    ----------
    minimum_enforced_files
        Minimum number of enforced paths admitted by this fixture.

    Returns
    -------
    dict
        Required verification gates and bounded migration settings.

    """
    return {
        "additions_only": True,
        "maximum_files_per_slice": 5,
        "minimum_enforced_files": minimum_enforced_files,
        "required_verification": [
            "focused pytest",
            "mypy --strict",
            "ruff check",
            "ruff format --check",
        ],
        "invalid_input_policy": "use narrow suppressions",
    }


def _cohort(
    cohort_id: str = "policy",
    *,
    order: int = 0,
    status: str = "enforced",
    files: list[object] | None = None,
) -> dict[str, object]:
    """Build one cohort record, allowing intentional invalid field values.

    Parameters
    ----------
    cohort_id
        Registry identity under test.
    order
        Proposed schedule position.
    status
        Enrolment state to validate.
    files
        Explicit path values, or a single valid default path.

    Returns
    -------
    dict
        Raw cohort input for the public policy parser.

    """
    return {
        "order": order,
        "id": cohort_id,
        "status": status,
        "selection": "bounded owner tests",
        "activation_criterion": "strict typing and focused tests pass",
        "files": ["tests/test_policy.py"] if files is None else files,
    }


def _payload(
    *cohorts: dict[str, object],
    baseline: object | None = None,
    rules: object | None = None,
    schema_version: object = 1,
) -> dict[str, object]:
    """Assemble a policy input with replaceable validation boundaries.

    Parameters
    ----------
    cohorts
        Ordered cohort records; omission uses one valid enforced cohort.
    baseline
        Historical measurement input, or the consistent default fixture.
    rules
        Migration rules input, or the additive default fixture.
    schema_version
        Schema value, including deliberate invalid types in refusal tests.

    Returns
    -------
    dict
        Complete raw policy input without pre-validating the test fault.

    """
    return {
        "schema_version": schema_version,
        "baseline": _baseline() if baseline is None else baseline,
        "migration_rules": _rules() if rules is None else rules,
        "cohorts": list(cohorts) if cohorts else [_cohort()],
    }


def test_parse_policy_exposes_enforced_cohort_and_paths() -> None:
    """Preserve enforced path order while excluding a planned empty cohort."""
    policy = _audit.parse_policy(
        _payload(
            _cohort(files=["tests/test_a.py", "tests/test_b.py"]),
            _cohort("next", order=1, status="planned", files=[]),
        )
    )

    assert policy.schema_version == 1
    assert tuple(cohort.cohort_id for cohort in policy.enforced_cohorts) == ("policy",)
    assert policy.enforced_paths == ("tests/test_a.py", "tests/test_b.py")
    assert policy.registered_paths == ("tests/test_a.py", "tests/test_b.py")
    assert policy.baseline.errors == 7
    assert policy.rules.maximum_files_per_slice == 5


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "registry must be an object"),
        (_payload(baseline=[]), "registry.baseline must be an object"),
        (_payload(rules=[]), "registry.migration_rules must be an object"),
        ({**_payload(), "cohorts": {}}, "registry.cohorts must be an array"),
        (
            _payload(baseline={**_baseline(), "measured_at": ""}),
            "measured_at must be a non-empty string",
        ),
        (
            _payload(rules={**_rules(), "required_verification": {}}),
            "required_verification must be an array",
        ),
        (
            _payload(rules={**_rules(), "required_verification": [""]}),
            r"required_verification\[0\] must be a non-empty string",
        ),
        (_payload(schema_version=True), "schema_version must be an integer"),
        (
            _payload(rules={**_rules(), "additions_only": 1}),
            "additions_only must be a boolean",
        ),
    ],
)
def test_parse_policy_rejects_structural_type_errors(payload: object, message: str) -> None:
    """Refuse malformed records before interpreting policy semantics.

    Parameters
    ----------
    payload
        Policy input with one structural type fault.
    message
        Required field-specific rejection diagnostic.

    """
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tracked_python_files", 0, "tracked_python_files must be positive"),
        ("files_with_errors", 0, "files_with_errors must be within"),
        ("files_with_errors", 11, "files_with_errors must be within"),
        ("errors", 2, "errors cannot be below"),
        ("dominant_error_count", 0, "dominant_error_count must be within"),
        ("dominant_error_count", 8, "dominant_error_count must be within"),
    ],
)
def test_parse_policy_rejects_impossible_baselines(field: str, value: int, message: str) -> None:
    """Reject measurement totals that cannot describe the declared inventory.

    Parameters
    ----------
    field
        Baseline field to replace.
    value
        Inconsistent measurement value.
    message
        Required rejection diagnostic.

    """
    baseline = _baseline()
    baseline[field] = value

    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_payload(baseline=baseline))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("additions_only", False, "additions_only must remain true"),
        ("maximum_files_per_slice", 0, "maximum_files_per_slice must be positive"),
        ("minimum_enforced_files", 0, "minimum_enforced_files must be positive"),
        (
            "required_verification",
            ["mypy --strict"],
            "required_verification misses required gates",
        ),
        ("invalid_input_policy", "", "invalid_input_policy must be a non-empty string"),
    ],
)
def test_parse_policy_rejects_unsafe_migration_rules(
    field: str, value: object, message: str
) -> None:
    """Keep additive enrolment, positive limits and required gates mandatory.

    Parameters
    ----------
    field
        Migration rule to corrupt.
    value
        Proposed unsafe rule value.
    message
        Required rejection diagnostic.

    """
    rules = _rules()
    rules[field] = value

    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_payload(rules=rules))


@pytest.mark.parametrize(
    ("cohort", "message"),
    [
        (_cohort("Not-valid"), "id must be a lowercase"),
        (_cohort(status="active"), "status is unsupported"),
        (_cohort(files=["tests/test_z.py", "tests/test_a.py"]), "files must be sorted"),
        (_cohort(files=["tests/test_a.py", "tests/test_a.py"]), "files contains duplicates"),
        (_cohort(files=[]), "enforced cohorts must contain files"),
        (_cohort(files=["/tests/test_a.py"]), "must be repository-relative"),
        (_cohort(files=["tests/../test_a.py"]), "must be repository-relative"),
        (_cohort(files=["src/test_a.py"]), "must name a Python file below tests"),
        (_cohort(files=["tests/data.json"]), "must name a Python file below tests"),
        (_cohort(files=[1]), r"files\[0\] must be a non-empty string"),
    ],
)
def test_parse_policy_rejects_invalid_cohort_rows(cohort: dict[str, object], message: str) -> None:
    """Reject ambiguous identities, invalid states and unsafe test paths.

    Parameters
    ----------
    cohort
        Raw cohort carrying the named fault.
    message
        Required rejection diagnostic.

    """
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_payload(cohort))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(schema_version=2), "schema_version must be 1"),
        ({**_payload(), "cohorts": []}, "cohorts must not be empty"),
        (_payload(_cohort(order=1)), "orders must be consecutive"),
        (
            _payload(_cohort(), _cohort(order=1)),
            "contains duplicate ids",
        ),
        (_payload(_cohort(status="planned", files=[])), "must start with all enforced"),
        (
            _payload(
                _cohort("first", status="planned", files=[]),
                _cohort("second", order=1),
            ),
            "must start with all enforced",
        ),
        (
            _payload(
                _cohort(),
                _cohort("later", order=1, status="deferred", files=[]),
                _cohort("middle", order=2, status="planned", files=[]),
            ),
            "statuses must follow enforced, planned, deferred order",
        ),
        (
            _payload(
                _cohort(files=["tests/test_same.py"]),
                _cohort("next", order=1, status="planned", files=["tests/test_same.py"]),
            ),
            "path in more than one cohort",
        ),
        (
            _payload(rules=_rules(minimum_enforced_files=2)),
            "enforced file count is below",
        ),
    ],
)
def test_parse_policy_rejects_invalid_schedule(payload: object, message: str) -> None:
    """Reject unsupported versions, duplicate ownership and invalid migration order.

    Parameters
    ----------
    payload
        Whole policy with the schedule or version fault.
    message
        Required rejection diagnostic.

    """
    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(payload)


def test_load_policy_and_tracked_paths_use_utf8_and_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load JSON and query sorted tracked paths from the selected repository.

    Parameters
    ----------
    tmp_path
        Isolated directory holding the policy fixture.
    monkeypatch
        Replace Git execution to inspect its exact command and working path.

    """
    registry = tmp_path / "policy.json"
    registry.write_text(json.dumps(_payload()), encoding="utf-8")
    policy = _audit.load_policy(registry)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        """Require the actual Git inventory command and supply unsorted NUL paths."""
        assert args[0] == ["git", "ls-files", "-z", "--", "tests"]
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess([], 0, stdout=b"tests/z.py\0tests/a.py\0", stderr=b"")

    monkeypatch.setattr(_audit.subprocess, "run", fake_run)

    assert policy.enforced_paths == ("tests/test_policy.py",)
    assert _audit.tracked_test_paths(tmp_path) == ("tests/a.py", "tests/z.py")


def test_audit_policy_and_command_report_exact_enforced_set() -> None:
    """Reject untracked ownership and pass only enforced paths to strict mypy."""
    policy = _audit.parse_policy(_payload())

    assert _audit.audit_policy(policy, ()) == (
        "registered test is not tracked: tests/test_policy.py",
    )
    assert _audit.audit_policy(policy, ("tests/test_policy.py",)) == ()
    assert _audit.mypy_command(policy, "/python") == (
        "/python",
        "-m",
        "mypy",
        "--strict",
        "tests/test_policy.py",
    )


def test_execute_mypy_combines_nonempty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain mypy failure status and both diagnostic streams.

    Parameters
    ----------
    tmp_path
        Expected checker working directory.
    monkeypatch
        Replace the subprocess with a controlled failed checker result.

    """

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Inspect checker invocation and emit distinct stdout/stderr diagnostics."""
        assert args[0] == ("python", "-m", "mypy")
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess([], 1, stdout="error\n", stderr="note\n")

    monkeypatch.setattr(_audit.subprocess, "run", fake_run)

    assert _audit.execute_mypy(("python", "-m", "mypy"), tmp_path) == (
        1,
        "error\nnote",
    )


def test_audit_repository_skips_mypy_for_validation_errors_or_validate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep inventory-only validation and malformed policy from launching mypy.

    Parameters
    ----------
    tmp_path
        Isolated repository path passed to the audit.
    monkeypatch
        Supply controlled tracked paths and forbid checker execution.

    """
    policy = _audit.parse_policy(_payload())
    monkeypatch.setattr(_audit, "load_policy", lambda path: policy)
    monkeypatch.setattr(_audit, "tracked_test_paths", lambda root: ())
    monkeypatch.setattr(
        _audit,
        "execute_mypy",
        lambda command, root: pytest.fail("mypy must be skipped for validation errors"),
    )

    invalid = _audit.audit_repository(tmp_path, tmp_path / "policy.json")
    assert invalid.mypy_returncode is None
    assert invalid.errors

    monkeypatch.setattr(
        _audit,
        "tracked_test_paths",
        lambda root: ("tests/test_policy.py",),
    )
    validated = _audit.audit_repository(
        tmp_path,
        tmp_path / "policy.json",
        run_type_check=False,
        python_executable="/python",
    )
    assert validated.errors == ()
    assert validated.mypy_returncode is None
    assert validated.command[0] == "/python"


def test_audit_repository_executes_mypy_for_valid_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve a successful checker result when the enforced policy validates.

    Parameters
    ----------
    tmp_path
        Isolated repository path passed to the audit.
    monkeypatch
        Supply valid policy ownership and a controlled checker result.

    """
    policy = _audit.parse_policy(_payload())
    monkeypatch.setattr(_audit, "load_policy", lambda path: policy)
    monkeypatch.setattr(
        _audit,
        "tracked_test_paths",
        lambda root: ("tests/test_policy.py",),
    )
    monkeypatch.setattr(_audit, "execute_mypy", lambda command, root: (0, "Success"))

    result = _audit.audit_repository(tmp_path, tmp_path / "policy.json")

    assert result.passed is True
    assert result.mypy_returncode == 0
    assert result.mypy_output == "Success"


def test_result_formatters_cover_errors_skip_failure_and_success() -> None:
    """Distinguish invalid inventory, skipped checks, failure and actual success."""
    policy = _audit.parse_policy(_payload())
    command = _audit.mypy_command(policy, "python")
    invalid = _audit.AuditResult(policy, ("missing",), command, None, "")
    skipped = _audit.AuditResult(policy, (), command, None, "")
    failed = _audit.AuditResult(policy, (), command, 1, "typing failed")
    passed = _audit.AuditResult(policy, (), command, 0, "Success")

    assert invalid.passed is False
    assert "- missing" in _audit.format_result(invalid)
    assert "validated; mypy skipped" in _audit.format_result(skipped)
    assert _audit.format_result(failed) == "test-typing policy audit failed:\ntyping failed"
    assert "strict mypy passed" in _audit.format_result(passed)
    assert json.loads(_audit._json_result(passed))["status"] == "pass"
    assert json.loads(_audit._json_result(failed))["status"] == "fail"


def test_main_supports_text_json_failure_and_registry_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Expose validation, checker and registry outcomes with distinct CLI statuses.

    Parameters
    ----------
    tmp_path
        Repository argument used in the CLI calls.
    monkeypatch
        Supply controlled audit outcomes without running a full type check.
    capsys
        Capture and verify human-readable and JSON diagnostics.

    """
    policy = _audit.parse_policy(_payload())
    command = _audit.mypy_command(policy, "python")
    passed = _audit.AuditResult(policy, (), command, None, "")
    failed = _audit.AuditResult(policy, (), command, 1, "typing failed")
    monkeypatch.setattr(_audit, "audit_repository", lambda *args, **kwargs: passed)

    assert _audit.main(["--repo-root", str(tmp_path), "--validate-only"]) == 0
    assert "mypy skipped" in capsys.readouterr().out

    assert _audit.main(["--repo-root", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"

    monkeypatch.setattr(_audit, "audit_repository", lambda *args, **kwargs: failed)
    assert _audit.main([]) == 1
    assert "typing failed" in capsys.readouterr().out

    def raise_value_error(*args: object, **kwargs: object) -> object:
        """Simulate a malformed registry before any checker outcome exists."""
        del args, kwargs
        raise ValueError("bad registry")

    monkeypatch.setattr(_audit, "audit_repository", raise_value_error)
    assert _audit.main([]) == 2
    assert "bad registry" in capsys.readouterr().out
    assert _audit.main(["--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_live_registry_validates_all_enforced_paths() -> None:
    """Keep the accepted enrolment floor while allowing additive policy growth."""
    repo_root = Path(__file__).resolve().parents[1]
    policy = _audit.load_policy(repo_root / "tools" / "test_typing_policy.json")
    tracked = set(_audit.tracked_test_paths(repo_root))

    assert policy.rules.minimum_enforced_files >= 1373
    assert len(policy.enforced_paths) >= policy.rules.minimum_enforced_files
    assert _audit.audit_policy(policy, tuple(sorted(tracked))) == ()


def test_test_infrastructure_documents_live_policy_baseline() -> None:
    """Resolve cohort membership and dated baseline through the canonical policy."""
    repo_root = Path(__file__).resolve().parents[1]
    policy = _audit.load_policy(repo_root / "tools" / "test_typing_policy.json")
    documentation = (repo_root / "docs" / "test_infrastructure.md").read_text(encoding="utf-8")

    assert (
        "https://github.com/anulum/scpn-quantum-control/blob/main/tools/test_typing_policy.json"
        in documentation
    )
    assert "](../tools/test_typing_policy.json)" not in documentation
    assert "dated baseline in the policy records historical debt" in documentation
    assert "--validate-only --json" in documentation
    for cohort in policy.enforced_cohorts:
        assert documentation.count(f"| `{cohort.cohort_id}` |") == 1
