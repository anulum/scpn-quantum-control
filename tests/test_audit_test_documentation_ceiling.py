# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the test documentation ceiling gate
"""Exercise the test documentation ceiling gate on real files, Git and Ruff."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tools import audit_test_documentation_ceiling as gate

DOCUMENTED = '"""Documented test module."""\n\n\ndef test_documented() -> None:\n    """Check that true is true."""\n    assert True\n'
LEGACY = (
    "def test_first() -> None:\n    assert True\n\n\ndef test_second() -> None:\n    assert True\n"
)
LEGACY_PATH = "tests/test_legacy.py"
DOCUMENTED_FUNCTION = (
    '\n\ndef test_third() -> None:\n    """Check that one is one."""\n    assert 1 == 1\n'
)
UNDOCUMENTED_FUNCTION = "\n\ndef test_third() -> None:\n    assert 1 == 1\n"


def _git(repository: Path, *arguments: str) -> str:
    """Run Git in ``repository`` and return its standard output."""
    completed = subprocess.run(  # noqa: S603 - fixed argument vector in a temporary repository
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _write(repository: Path, name: str, text: str) -> None:
    """Write ``text`` to ``name`` inside ``repository``, creating parents."""
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ceiling(repository: Path) -> dict[str, object]:
    """Return the recorded ceiling of ``repository`` as parsed JSON."""
    payload: dict[str, object] = json.loads(
        (repository / gate.DEFAULT_CEILING).read_text(encoding="utf-8")
    )
    return payload


def _store(repository: Path, payload: dict[str, object]) -> None:
    """Replace the recorded ceiling of ``repository`` with ``payload``."""
    _write(repository, str(gate.DEFAULT_CEILING), json.dumps(payload))


def _run(repository: Path, *arguments: str) -> int:
    """Run the gate's command line against ``repository``."""
    return gate.main(["--repo", str(repository), *arguments])


@pytest.fixture(autouse=True)
def _no_inherited_git_location() -> Iterator[None]:
    """Drop inherited ``GIT_*`` variables so Git targets the temporary repository.

    An exported ``GIT_DIR`` or ``GIT_WORK_TREE`` would redirect the fixture's
    ``git init`` and ``git commit`` to the checkout that runs the tests.
    """
    inherited = {
        name: os.environ.pop(name) for name in list(os.environ) if name.startswith("GIT_")
    }
    try:
        yield
    finally:
        os.environ.update(inherited)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a committed repository whose ceiling equals its measurement."""
    _write(tmp_path, "tests/test_documented.py", DOCUMENTED)
    _write(tmp_path, LEGACY_PATH, LEGACY)
    _write(tmp_path, "oscillatools/tests/test_other.py", DOCUMENTED)
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _git(tmp_path, "add", "--", "tests", "oscillatools")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    measured = gate.measure(tmp_path)
    assert measured == {LEGACY_PATH: 3}
    (tmp_path / gate.DEFAULT_CEILING).parent.mkdir()
    gate.write_ceiling(
        tmp_path / gate.DEFAULT_CEILING,
        gate.Ceiling(
            gate.ruff_version(tmp_path), _git(tmp_path, "rev-parse", "HEAD"), measured, ()
        ),
    )
    return tmp_path


def test_fixture_repository_is_separate_from_the_checkout(repository: Path) -> None:
    """Keep every fixture Git command inside the temporary repository."""
    assert not any(name.startswith("GIT_") for name in os.environ)
    assert (
        Path(_git(repository, "rev-parse", "--absolute-git-dir"))
        == (repository / ".git").resolve()
    )
    assert Path(_git(repository, "rev-parse", "--show-toplevel")) == repository.resolve()


def test_measurement_equal_to_the_ceiling_passes(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept a tree whose findings equal the recorded per-file counts."""
    assert _run(repository) == 0
    assert "3 findings in 1 files; ceiling 3 in 1; 0 problems" in capsys.readouterr().out


def test_new_undocumented_test_file_is_rejected(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a test file that is absent from the ceiling and has findings."""
    _write(repository, "oscillatools/tests/test_new.py", LEGACY)
    assert _run(repository) == 1
    captured = capsys.readouterr().err
    assert "undocumented test file outside the ceiling: oscillatools/tests/test_new.py" in captured


def test_growth_above_the_ceiling_is_rejected(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a recorded file whose finding count rose."""
    _write(repository, LEGACY_PATH, LEGACY + UNDOCUMENTED_FUNCTION)
    assert _run(repository) == 1
    assert f"documentation findings grew: {LEGACY_PATH}: 3 -> 4" in capsys.readouterr().err


def test_reduction_must_be_recorded_and_then_passes(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require the ceiling to follow a reduction, then accept the lowered figure."""
    _write(repository, LEGACY_PATH, '"""Legacy module, now described."""\n\n\n' + LEGACY)
    assert _run(repository) == 1
    assert f"ceiling is above the measurement: {LEGACY_PATH}: 3 -> 2" in capsys.readouterr().err
    assert _run(repository, "--lower") == 0
    assert "ceiling written: 2 findings in 1 files" in capsys.readouterr().out
    recorded = _ceiling(repository)
    assert recorded["files"] == {LEGACY_PATH: 2}
    assert recorded["total_findings"] == 2
    assert recorded["measured_commit"] == _git(repository, "rev-parse", "HEAD")
    assert _run(repository) == 0


@pytest.mark.parametrize("removed", [False, True])
def test_row_without_findings_must_be_removed(
    repository: Path, capsys: pytest.CaptureFixture[str], removed: bool
) -> None:
    """Refuse a ceiling row whose file became clean or disappeared."""
    if removed:
        (repository / LEGACY_PATH).unlink()
    else:
        _write(repository, LEGACY_PATH, DOCUMENTED)
    assert _run(repository) == 1
    assert f"ceiling row has no findings left: {LEGACY_PATH}" in capsys.readouterr().err
    assert _run(repository, "--lower") == 0
    assert _ceiling(repository)["files"] == {}
    assert _run(repository) == 0


def test_lowering_never_admits_new_debt(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse to lower the ceiling while a file grew or a new file has findings."""
    _write(repository, LEGACY_PATH, LEGACY + UNDOCUMENTED_FUNCTION)
    _write(repository, "tests/test_new.py", LEGACY)
    before = _ceiling(repository)
    assert _run(repository, "--lower") == 1
    captured = capsys.readouterr().err
    assert "cannot lower the ceiling while findings grew" in captured
    assert LEGACY_PATH in captured
    assert "tests/test_new.py" in captured
    assert _ceiling(repository) == before


def test_release_change_blocks_comparison_until_rebaselined(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse cross-release comparison and lowering; record a new measurement instead."""
    payload = _ceiling(repository)
    payload["ruff_version"] = "ruff 0.0.0"
    _store(repository, payload)
    assert _run(repository) == 1
    assert "measured with ruff 0.0.0" in capsys.readouterr().err
    assert _run(repository, "--lower") == 1
    assert "use --rebaseline, not --lower" in capsys.readouterr().err
    assert _run(repository, "--rebaseline") == 0
    capsys.readouterr()
    recorded = _ceiling(repository)
    assert recorded["ruff_version"] == gate.ruff_version(repository)
    assert recorded["history"] == [
        {
            "ruff_version": "ruff 0.0.0",
            "measured_commit": payload["measured_commit"],
            "total_findings": 3,
            "total_files": 1,
        }
    ]
    assert _run(repository) == 0


def test_rebaseline_is_refused_under_the_same_release(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep the ceiling falling: a new measurement needs a changed Ruff release."""
    _write(repository, LEGACY_PATH, LEGACY + UNDOCUMENTED_FUNCTION)
    assert _run(repository, "--rebaseline") == 1
    assert "the ceiling can only be lowered" in capsys.readouterr().err
    assert _ceiling(repository)["files"] == {LEGACY_PATH: 3}


def test_changed_test_file_must_be_fully_documented(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require a touched test file to have no finding, whatever its ceiling."""
    _write(repository, LEGACY_PATH, LEGACY + DOCUMENTED_FUNCTION)
    assert _run(repository) == 0
    capsys.readouterr()
    assert _run(repository, "--changed-against", "HEAD") == 1
    assert (
        f"changed test file must be fully documented: {LEGACY_PATH} (3 findings)"
        in capsys.readouterr().err
    )


def test_changed_documented_file_and_untouched_debt_pass(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept a documented change while untouched recorded debt stays under its ceiling."""
    _write(repository, "tests/test_documented.py", DOCUMENTED + DOCUMENTED_FUNCTION)
    _write(repository, "tests/notes.txt", "not a Python test file\n")
    _git(repository, "add", "--", "tests/notes.txt")
    assert gate.changed_test_files(repository, "HEAD") == ["tests/test_documented.py"]
    assert _run(repository, "--changed-against", "HEAD") == 0
    assert "0 problems" in capsys.readouterr().out


def test_unknown_revision_fails_closed(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a comparison revision that Git cannot resolve."""
    assert _run(repository, "--changed-against", "no-such-revision") == 1
    assert "test documentation ceiling failed: git diff" in capsys.readouterr().err


def test_absent_repository_and_ceiling_are_errors(
    repository: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a missing working directory and a missing ceiling file as failures."""
    with pytest.raises(ValueError, match="cannot run git"):
        gate.changed_test_files(tmp_path / "absent", "HEAD")
    (repository / gate.DEFAULT_CEILING).unlink()
    assert _run(repository) == 1
    assert "test documentation ceiling failed" in capsys.readouterr().err


def test_missing_test_scope_is_an_error(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail when one of the measured test scopes is not a directory."""
    (repository / "oscillatools/tests/test_other.py").unlink()
    (repository / "oscillatools/tests").rmdir()
    assert _run(repository) == 1
    assert "required documentation scope is not a directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "test_documentation_ceiling_v0"}, "unsupported test documentation ceiling"),
        ({"ruff_version": " "}, "must record the Ruff release"),
        ({"measured_commit": 7}, "must record the commit"),
        ({"scopes": ["tests"]}, "scopes differ"),
        ({"files": []}, "files must be an object"),
        ({"files": {"src/module.py": 1}}, "outside the test scopes"),
        ({"files": {"tests/../src/test_escape.py": 1}}, "outside the test scopes"),
        ({"files": {"tests/readme.txt": 1}}, "outside the test scopes"),
        ({"files": {LEGACY_PATH: 0}}, "must be a positive integer"),
        ({"files": {LEGACY_PATH: True}}, "must be a positive integer"),
        ({"files": {LEGACY_PATH: "3"}}, "must be a positive integer"),
        ({"history": {}}, "history must be a list of objects"),
        ({"history": ["ruff 0.0.0"]}, "history must be a list of objects"),
    ],
)
def test_invalid_ceiling_is_refused(
    repository: Path, change: dict[str, object], message: str
) -> None:
    """Reject a ceiling with a stale schema or malformed provenance, paths or counts."""
    payload = _ceiling(repository)
    payload.update(change)
    _store(repository, payload)
    with pytest.raises(ValueError, match=message):
        gate.load_ceiling(repository / gate.DEFAULT_CEILING)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[]", "unsupported test documentation ceiling"),
        ('{"schema": "a", "schema": "b"}', "duplicate key in ceiling: schema"),
        ("{", "Expecting property name"),
    ],
)
def test_malformed_ceiling_text_is_refused(repository: Path, text: str, message: str) -> None:
    """Reject a ceiling that is not one JSON object with unique keys."""
    _write(repository, str(gate.DEFAULT_CEILING), text)
    with pytest.raises(ValueError, match=message):
        gate.load_ceiling(repository / gate.DEFAULT_CEILING)


def test_script_entry_point_reports_the_same_result(repository: Path) -> None:
    """Run the gate as a script, the way the workflow and the hook invoke it."""
    script = Path(gate.__file__).resolve()
    completed = subprocess.run(  # noqa: S603 - fixed argument vector
        [sys.executable, str(script), "--repo", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "0 problems" in completed.stdout
