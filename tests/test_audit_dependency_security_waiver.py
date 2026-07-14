# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — constrained dependency advisory waiver tests
"""Tests for the fail-closed dependency advisory waiver audit."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from tools import audit_dependency_security_waiver as waiver

REPO_ROOT = Path(__file__).resolve().parents[1]


def _distribution_record(
    name: str,
    *,
    version: str = "1.39.5",
    setuptools_requirement: str = "setuptools==81.0.0",
    extra_requirements: tuple[str, ...] = ("numpy",),
) -> waiver.DistributionRecord:
    requirements = (*extra_requirements, setuptools_requirement)
    return waiver.DistributionRecord(name=name, version=version, requirements=requirements)


def _valid_records() -> tuple[waiver.DistributionRecord, ...]:
    return tuple(_distribution_record(name) for name in waiver.BRAKET_DISTRIBUTIONS)


def _lock_text(
    *,
    version: str = waiver.SETUPTOOLS_VERSION,
    hashes: tuple[str, ...] = tuple(sorted(waiver.SETUPTOOLS_HASHES)),
    via: tuple[str, ...] = waiver.BRAKET_DISTRIBUTIONS,
) -> str:
    lines = [f"setuptools=={version} \\"]
    for index, digest in enumerate(hashes):
        suffix = " \\" if index < len(hashes) - 1 else ""
        lines.append(f"    --hash=sha256:{digest}{suffix}")
    lines.extend(("    # via", *(f"    #   {owner}" for owner in via), "wheel==0.46.3"))
    return "\n".join(lines) + "\n"


def _lock_mapping(**overrides: str) -> dict[str, str]:
    return {path: overrides.get(path, _lock_text()) for path in waiver.LOCK_PATHS}


def _workflow(
    *,
    gate_command: str = waiver.WAIVER_GATE_COMMAND,
    pip_audit_command: tuple[str, ...] = waiver.EXPECTED_PIP_AUDIT_COMMAND,
) -> str:
    return (
        "security:\n"
        "  steps:\n"
        f"    - run: {gate_command}\n"
        f"    - run: {_shlex_join(pip_audit_command)}\n"
    )


def _shlex_join(parts: tuple[str, ...]) -> str:
    """Join the repository's shell-safe alphanumeric audit arguments."""
    return " ".join(parts)


def _documentation() -> str:
    return "\n".join(
        (
            waiver.WAIVER_DOC_HEADING,
            waiver.ADVISORY_ID,
            f"setuptools=={waiver.SETUPTOOLS_VERSION}",
            waiver.BUILD_BACKEND,
            "Remove the waiver when both pin owners permit the fixed version.",
        )
    )


def _write_replay_repository(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "tools").mkdir()
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n',
        encoding="utf-8",
    )
    for path, text in _lock_mapping().items():
        (root / path).write_text(text, encoding="utf-8")
    (root / ".github" / "workflows" / "ci.yml").write_text(_workflow(), encoding="utf-8")
    (root / "docs" / "test_infrastructure.md").write_text(_documentation(), encoding="utf-8")
    (root / "src" / "runtime.py").write_text("import pathlib\n", encoding="utf-8")


def test_live_repository_waiver_matches_installed_braket_metadata() -> None:
    """The checked-out CI boundary must match installed package metadata."""
    result = waiver.audit_repository(REPO_ROOT)

    assert result.passed
    assert result.errors == ()


def test_recorded_metadata_replay_exercises_the_complete_repository_contract(
    tmp_path: Path,
) -> None:
    """A complete recorded repository replay must pass without network access."""
    _write_replay_repository(tmp_path)

    result = waiver.audit_repository(tmp_path, _valid_records())

    assert result == waiver.WaiverAuditResult(errors=())


def test_main_reports_pass_for_a_complete_repository(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator CLI must report the advisory and removal condition."""
    _write_replay_repository(tmp_path)

    status = waiver.main(["--repo-root", str(tmp_path)])

    output = capsys.readouterr().out
    assert status == 0
    assert "dependency security waiver audit: PASS" in output
    assert waiver.ADVISORY_ID in output
    assert "setuptools<83.0.0" in output


def test_script_entrypoint_executes_the_repository_audit(tmp_path: Path) -> None:
    """Direct script execution must use the same audited public entrypoint."""
    script = REPO_ROOT / "tools" / "audit_dependency_security_waiver.py"
    _write_replay_repository(tmp_path)

    original_argv = sys.argv
    try:
        sys.argv = [str(script), "--repo-root", str(tmp_path)]
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv

    assert exc_info.value.code == 0


def test_lock_parser_reads_exact_version_hashes_and_pin_owners() -> None:
    """The parser must preserve every security-relevant pip-compile field."""
    pin = waiver.parse_setuptools_lock_pin(_lock_text())

    assert pin.version == waiver.SETUPTOOLS_VERSION
    assert pin.hashes == waiver.SETUPTOOLS_HASHES
    assert pin.via == frozenset(waiver.BRAKET_DISTRIBUTIONS)


def test_lock_parser_rejects_missing_duplicate_and_malformed_pins() -> None:
    """Missing, duplicated, or non-SHA-256 stanzas must fail closed."""
    with pytest.raises(ValueError, match="found 0"):
        waiver.parse_setuptools_lock_pin("wheel==0.46.3\n")
    with pytest.raises(ValueError, match="found 2"):
        waiver.parse_setuptools_lock_pin(_lock_text() + _lock_text())
    with pytest.raises(ValueError, match="has no version"):
        waiver.parse_setuptools_lock_pin("setuptools==\n")
    malformed = _lock_text(hashes=("z" * 64,))
    with pytest.raises(ValueError, match="invalid sha256"):
        waiver.parse_setuptools_lock_pin(malformed)
    unhashed = waiver.parse_setuptools_lock_pin(
        "setuptools==81.0.0\n    # via\n    #   amazon-braket-schemas\n"
    )
    assert unhashed.hashes == frozenset()
    assert waiver.parse_setuptools_lock_pin("setuptools==81.0.0").via == frozenset()


def test_lock_audit_rejects_missing_version_hash_and_owner_drift() -> None:
    """Every Python CI lock must retain the same source-verified closure."""
    lock_texts = {
        waiver.LOCK_PATHS[0]: _lock_text(version="83.0.0"),
        waiver.LOCK_PATHS[1]: _lock_text(hashes=(next(iter(waiver.SETUPTOOLS_HASHES)),)),
        waiver.LOCK_PATHS[2]: _lock_text(via=(waiver.BRAKET_DISTRIBUTIONS[0],)),
    }
    errors = waiver.audit_lockfiles(lock_texts)

    assert any("pin is 83.0.0" in error for error in errors)
    assert any("distribution hashes drifted" in error for error in errors)
    assert any("pin owners drifted" in error for error in errors)
    assert waiver.audit_lockfiles({}) == tuple(
        f"missing CI lock: {path}" for path in waiver.LOCK_PATHS
    )
    malformed = _lock_mapping(**{waiver.LOCK_PATHS[0]: _lock_text(hashes=("z" * 64,))})
    assert "invalid sha256" in waiver.audit_lockfiles(malformed)[0]


def test_build_system_audit_rejects_setuptools_and_malformed_metadata() -> None:
    """The waiver is invalid if this project starts building with setuptools."""
    valid = '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
    assert waiver.audit_build_system(valid) == ()
    assert waiver.audit_build_system("[project]\nname='x'\n") == (
        "pyproject.toml has no [build-system] table",
    )

    invalid = (
        '[build-system]\nrequires=["setuptools>=81", "not a req @"]\n'
        'build-backend="setuptools.build_meta"\n'
    )
    errors = waiver.audit_build_system(invalid)
    assert f"project build backend must remain {waiver.BUILD_BACKEND}" in errors
    assert "project build requirements must not include setuptools" in errors
    assert any("invalid build requirement" in error for error in errors)
    assert waiver.audit_build_system('[build-system]\nrequires="hatchling"\n')[-1] == (
        "build-system.requires must be a string array"
    )


def test_distribution_audit_rejects_changed_or_invalid_braket_metadata() -> None:
    """Changed upstream metadata must invalidate the exception automatically."""
    assert waiver.audit_distribution_records(_valid_records()) == ()

    records = (
        _distribution_record(
            waiver.BRAKET_DISTRIBUTIONS[0],
            version="not-a-version",
            setuptools_requirement="setuptools>=83.0.0; python_version > '3.11'",
            extra_requirements=("bad requirement @",),
        ),
        waiver.DistributionRecord(
            name="unexpected-package",
            version="1.0.0",
            requirements=("numpy",),
        ),
    )
    errors = waiver.audit_distribution_records(records)

    assert "installed metadata must cover exactly both Braket pin owners" in errors
    assert any("invalid installed version" in error for error in errors)
    assert any("invalid Requires-Dist metadata" in error for error in errors)
    assert any("no longer hard-pins" in error for error in errors)
    assert any("found 0" in error for error in errors)


def test_installed_distribution_loader_reports_missing_packages() -> None:
    """Absent CI packages must produce an explicit policy error."""
    records, errors = waiver.installed_distribution_records(
        ("definitely-not-a-real-distribution-4184931",)
    )

    assert records == ()
    assert errors == (
        "required installed distribution is missing: definitely-not-a-real-distribution-4184931",
    )


def test_ci_workflow_audit_rejects_broader_or_unchecked_exceptions() -> None:
    """CI may neither omit the precheck nor ignore a second advisory."""
    assert waiver.audit_ci_workflow(_workflow()) == ()

    broader = (
        *waiver.EXPECTED_PIP_AUDIT_COMMAND,
        "--ignore-vuln",
        "PYSEC-2099-0001",
    )
    errors = waiver.audit_ci_workflow(_workflow(gate_command="true", pip_audit_command=broader))

    assert errors == (
        "CI must run the waiver audit exactly once",
        "CI pip-audit command must scan the full lock and ignore only PYSEC-2026-3447",
    )

    block_scalar = (
        "security:\n"
        "  steps:\n"
        "    - run: |\n"
        "        python tools/audit_dependency_security_waiver.py\n"
        "    - run: >-\n"
        f"        {_shlex_join(waiver.EXPECTED_PIP_AUDIT_COMMAND)}\n"
    )
    assert waiver.audit_ci_workflow(block_scalar) == ()

    hidden_second_exception = (
        _workflow()
        + "    - run: |\n"
        + "        pip-audit -r requirements-ci-py312-linux.txt --ignore-vuln OTHER\n"
    )
    assert waiver.audit_ci_workflow(hidden_second_exception) == (
        "CI pip-audit command must scan the full lock and ignore only PYSEC-2026-3447",
    )


def test_setuptools_import_audit_scans_all_project_python_roots(tmp_path: Path) -> None:
    """A project import of the waived build tool must invalidate the boundary."""
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "src" / "safe.py").write_text(
        '"import setuptools"\n# from setuptools import setup\nimport pathlib\n',
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "unsafe.py").write_text(
        "from setuptools.command import build\n", encoding="utf-8"
    )
    (tmp_path / "tools" / "unsafe.py").write_text(
        "import pathlib, setuptools.command\n", encoding="utf-8"
    )
    (tmp_path / "tools" / "invalid.py").write_text("if True print('x')\n", encoding="utf-8")

    errors = waiver.audit_setuptools_imports(tmp_path)

    assert errors == (
        "project code imports setuptools: scripts/unsafe.py",
        "cannot audit Python imports in tools/invalid.py: invalid syntax (<unknown>, line 1)",
        "project code imports setuptools: tools/unsafe.py",
    )
    assert waiver.audit_setuptools_imports(tmp_path / "absent") == ()


def test_operator_documentation_audit_requires_every_stable_marker() -> None:
    """Operators must retain the threat model and deterministic removal rule."""
    assert waiver.audit_operator_documentation(_documentation()) == ()

    errors = waiver.audit_operator_documentation("temporary exception\n")

    assert len(errors) == 5
    assert errors[0].endswith(waiver.WAIVER_DOC_HEADING)
    assert errors[-1].endswith("Remove the waiver")


def test_repository_and_cli_report_missing_surfaces_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An incomplete checkout must return findings through the normal CLI path."""
    result = waiver.audit_repository(tmp_path, _valid_records())

    assert not result.passed
    assert any("cannot read pyproject.toml" in error for error in result.errors)
    assert any("cannot read CI workflow" in error for error in result.errors)
    assert any("cannot read operator documentation" in error for error in result.errors)

    status = waiver.main(["--repo-root", str(tmp_path)])
    output = capsys.readouterr().out
    assert status == 1
    assert output.startswith("dependency security waiver audit: FAIL")
    assert "required installed distribution is missing" not in output
