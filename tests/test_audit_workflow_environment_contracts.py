# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — workflow environment contract gate tests
"""Tests for the gate that checks a CI job against the commands it runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import audit_workflow_environment_contracts as gate

_LOCK = """\
coverage[toml]==7.14.0 \\
    --hash=sha256:0000
pytest==9.1.1 \\
    --hash=sha256:1111
"""


def _workflow(root: Path, name: str, body: str) -> None:
    """Write one workflow file into a scratch repository."""
    directory = root / ".github" / "workflows"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    """Build a scratch repository carrying one pinned requirement lock."""
    (tmp_path / "requirements-ci.txt").write_text(_LOCK, encoding="utf-8")
    return tmp_path


def test_lock_parsing_keeps_distributions_that_carry_extras(tmp_path: Path) -> None:
    """Read ``name[extra]==version`` as the distribution it pins."""
    lock = tmp_path / "requirements-ci.txt"
    lock.write_text(_LOCK, encoding="utf-8")

    found = gate.distributions_in(lock)

    assert found == {"coverage", "pytest"}


def test_missing_lock_contributes_nothing(tmp_path: Path) -> None:
    """Return an empty set for a requirement file that does not exist."""
    assert gate.distributions_in(tmp_path / "absent.txt") == frozenset()


def test_job_running_a_module_its_install_omits_is_reported(tmp_path: Path) -> None:
    """Report a job that invokes a module no requirement file it installs pins."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "example.yml",
        """
jobs:
  narrow-tier:
    steps:
      - run: python -m pip install -r requirements-absent.txt
      - run: python -m pytest -q tests/test_thing.py
""",
    )

    findings = gate.audit(repo)

    assert any("pinning 'pytest'" in str(finding) for finding in findings)


def test_job_installing_the_module_is_accepted(tmp_path: Path) -> None:
    """Accept a job whose install provides every module it runs."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "example.yml",
        """
jobs:
  complete-tier:
    env:
      PYTHONPATH: src
    steps:
      - run: python -m pip install -r requirements-ci.txt
      - run: python -m pytest -q tests/test_thing.py
""",
    )

    assert gate.audit(repo) == []


@pytest.mark.parametrize(
    "importable_step",
    [
        "python -m pip install --no-deps -e .",
        "python -m pip install --no-deps dist/example-1.0-py3-none-any.whl",
        'echo "PYTHONPATH=$GITHUB_WORKSPACE/src" >> "$GITHUB_ENV"',
    ],
)
def test_every_way_a_job_exposes_the_package_counts(tmp_path: Path, importable_step: str) -> None:
    """Accept an editable install, a built wheel, and an exported path alike."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "example.yml",
        f"""
jobs:
  installs-the-project:
    steps:
      - run: python -m pip install -r requirements-ci.txt
      - run: {importable_step}
      - run: python -m pytest -q tests/test_thing.py
""",
    )

    assert gate.audit(repo) == []


def test_pytest_over_the_repository_tests_needs_the_package(tmp_path: Path) -> None:
    """Report a job that runs the repository tests without exposing the package."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "example.yml",
        """
jobs:
  pathless:
    steps:
      - run: python -m pip install -r requirements-ci.txt
      - run: python -m pytest -q tests/test_thing.py
""",
    )

    findings = gate.audit(repo)

    assert any("conftest.py cannot import the package" in str(finding) for finding in findings)


def test_a_tool_pinned_twice_to_different_versions_is_reported(tmp_path: Path) -> None:
    """Report one action pinned to conflicting versions across workflows."""
    repo = _repo(tmp_path)
    for name, version in (("one.yml", "1.0.0"), ("two.yml", "2.0.0")):
        _workflow(
            repo,
            name,
            f"""
jobs:
  uses-a-tool:
    steps:
      - uses: example/action-setup@sha
        with:
          version: {version}
""",
        )

    findings = gate.audit(repo)

    assert any("pinned to more than one version" in str(finding) for finding in findings)


def test_the_same_pin_everywhere_is_accepted(tmp_path: Path) -> None:
    """Accept one action pinned to the same version in every workflow."""
    repo = _repo(tmp_path)
    for name in ("one.yml", "two.yml"):
        _workflow(
            repo,
            name,
            """
jobs:
  uses-a-tool:
    steps:
      - uses: example/action-setup@sha
        with:
          version: 3.0.0
""",
        )

    assert gate.audit(repo) == []


def test_standard_library_modules_need_no_requirement_file(tmp_path: Path) -> None:
    """Accept ``python -m venv`` without a requirement file pinning it."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "example.yml",
        """
jobs:
  bootstraps:
    steps:
      - run: python -m venv .venv
""",
    )

    assert gate.audit(repo) == []


def test_the_live_repository_satisfies_its_own_workflow_contracts() -> None:
    """Every job in this repository installs what it runs."""
    repo = Path(__file__).resolve().parents[1]

    assert gate.audit(repo) == []


@pytest.mark.parametrize(
    "body",
    [
        "",
        "[]",
        "jobs: {}",
        "jobs: []",
        "jobs: {broken: null}",
        "jobs: {broken: {steps: []}}",
        "jobs: [",
    ],
)
def test_cli_refuses_invalid_workflow_inventory(tmp_path: Path, body: str) -> None:
    """Malformed or empty job definitions must not yield a clean audit."""
    _workflow(tmp_path, "invalid.yaml", body)
    result = subprocess.run(
        [sys.executable, str(Path(gate.__file__).resolve()), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1
    assert "invalid.yaml" in result.stdout
    assert gate.main(["--repo", str(tmp_path)]) == 1


def test_main_refuses_missing_or_empty_checkout(tmp_path: Path) -> None:
    """A missing checkout or absent workflow collection is an audit failure."""
    assert gate.main(["--repo", str(tmp_path / "absent")]) == 1
    assert gate.main(["--repo", str(tmp_path)]) == 1


@pytest.mark.parametrize("separator", ["\n      - run: ", " && ", "; "])
def test_install_after_use_does_not_satisfy_requirement(tmp_path: Path, separator: str) -> None:
    """A later pip installation cannot supply an earlier pytest invocation."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "ordered.yml",
        "jobs:\n  check:\n    env: {PYTHONPATH: src}\n    steps:\n      - run: python -m pytest tests/test_thing.py"
        + separator
        + "python -m pip install -r requirements-ci.txt\n",
    )
    assert any("pinning 'pytest'" in str(item) for item in gate.audit(repo))


def test_package_install_after_tests_is_not_import_evidence(tmp_path: Path) -> None:
    """Installing the project after collection cannot make earlier imports work."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "ordered.yml",
        """
jobs:
  check:
    steps:
      - run: python -m pip install -r requirements-ci.txt
      - run: python -m pytest tests/test_thing.py
      - run: python -m pip install -e .
""",
    )
    assert any("conftest.py" in str(item) for item in gate.audit(repo))


def test_yaml_extension_and_reusable_job_are_admitted(tmp_path: Path) -> None:
    """Both YAML suffixes and nonempty reusable workflow references are valid."""
    _workflow(
        tmp_path, "caller.yaml", "jobs:\n  delegate:\n    uses: ./.github/workflows/child.yml\n"
    )
    _workflow(
        tmp_path,
        "child.yml",
        "jobs:\n  run:\n    steps:\n      - run: python -m json.tool --help\n",
    )
    assert gate.main(["--repo", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "definition",
    [
        "{uses: ''}",
        "{uses: 123}",
        "{uses: child, steps: []}",
        "{steps: [null]}",
        "{steps: [{run: ''}]}",
        "{steps: [{run: echo ok, uses: child}]}",
        "{steps: [{run: echo ok, uses: null}]}",
    ],
)
def test_invalid_job_shapes_are_findings(tmp_path: Path, definition: str) -> None:
    """Invalid reusable references and ambiguous steps cannot disappear from an audit."""
    _workflow(tmp_path, "invalid.yml", f"jobs:\n  malformed: {definition}\n")
    assert gate.main(["--repo", str(tmp_path)]) == 1


def test_workflow_file_read_errors_are_findings(tmp_path: Path) -> None:
    """A directory with a YAML suffix is an input error, not a skipped workflow."""
    (tmp_path / ".github" / "workflows" / "directory.yml").mkdir(parents=True)
    assert gate.main(["--repo", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    "install",
    [
        "# pip install -r requirements-ci.txt",
        "echo pip install -r requirements-ci.txt",
        "pip freeze -r requirements-ci.txt",
    ],
)
def test_text_that_is_not_an_install_provides_no_modules(tmp_path: Path, install: str) -> None:
    """A comment, echoed command or requirement reader is not an installation."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "not-installed.yml",
        f"""
jobs:
  check:
    env: {{PYTHONPATH: src}}
    steps:
      - run: |
          {install}
          python -m pytest tests/test_thing.py
""",
    )
    assert any("pinning 'pytest'" in str(item) for item in gate.audit(repo))


@pytest.mark.parametrize("separator", ["\n          ", " && ", "; "])
def test_install_before_use_in_one_step_is_accepted(tmp_path: Path, separator: str) -> None:
    """Literal sequential shell commands expose dependencies before their consumer."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "installed.yml",
        f"""
jobs:
  check:
    steps:
      - run: |
          python -m pip install -r requirements-ci.txt{separator}pip install .{separator}python -m pytest tests/test_thing.py
""",
    )
    assert gate.main(["--repo", str(tmp_path)]) == 0


@pytest.mark.parametrize(
    "preparation",
    ["echo pip install .", 'echo "PYTHONPATH=src" >> "$GITHUB_ENV"'],
)
def test_same_step_requires_actual_project_visibility(tmp_path: Path, preparation: str) -> None:
    """Echoed installs and future-step GITHUB_ENV writes do not expose current imports."""
    repo = _repo(tmp_path)
    _workflow(
        repo,
        "visibility.yml",
        f"""
jobs:
  check:
    steps:
      - run: |
          python -m pip install -r requirements-ci.txt
          {preparation}
          python -m pytest tests/test_thing.py
""",
    )
    assert any("conftest.py" in str(item) for item in gate.audit(repo))
