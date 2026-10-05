# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the tracked file size ceiling gate
"""Exercise the tracked file size ceiling gate on real files and a real Git index."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tools import audit_tracked_file_size_ceiling as gate

BIG: str = "data/big.bin"
SMALL: str = "notes/small.txt"
ONE_AND_A_HALF: int = gate.MEBIBYTE + gate.MEBIBYTE // 2
ABSENT_OBJECT: str = "0123456789abcdef0123456789abcdef01234567"


def _git(repository: Path, *arguments: str, stdin: str = "") -> str:
    """Run Git in ``repository`` and return its standard output."""
    completed = subprocess.run(  # noqa: S603 - fixed argument vector in a temporary repository
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *arguments],
        cwd=repository,
        capture_output=True,
        input=stdin,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _track(repository: Path, name: str, size: int) -> None:
    """Write ``size`` zero bytes to ``name`` inside ``repository`` and stage the file."""
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(size))
    _git(repository, "add", "--", name)


def _ceiling(repository: Path) -> dict[str, object]:
    """Return the recorded ceiling of ``repository`` as parsed JSON."""
    payload: dict[str, object] = json.loads(
        (repository / gate.DEFAULT_CEILING).read_text(encoding="utf-8")
    )
    return payload


def _store(repository: Path, payload: dict[str, object]) -> None:
    """Replace the recorded ceiling of ``repository`` with ``payload``."""
    (repository / gate.DEFAULT_CEILING).write_text(json.dumps(payload), encoding="utf-8")


def _run(repository: Path, *arguments: str) -> int:
    """Run the gate's command line against ``repository``."""
    return gate.main(["--repo", str(repository), *arguments])


@pytest.fixture(autouse=True)
def _no_inherited_git_location() -> Iterator[None]:
    """Drop inherited ``GIT_*`` variables so Git targets the temporary repository.

    An exported ``GIT_DIR`` or ``GIT_WORK_TREE`` would redirect the fixture's
    ``git init`` and ``git add`` to the checkout that runs the tests.
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
    """Create a repository with one small and one large staged file and a tight ceiling."""
    _git(tmp_path, "init", "--quiet", "--initial-branch=main")
    _track(tmp_path, SMALL, 10)
    _track(tmp_path, BIG, ONE_AND_A_HALF)
    (tmp_path / gate.DEFAULT_CEILING).parent.mkdir()
    gate.write_ceiling(tmp_path / gate.DEFAULT_CEILING, gate.Ceiling(gate.MEBIBYTE, {BIG: 2}))
    return tmp_path


def test_fixture_repository_is_separate_from_the_checkout(repository: Path) -> None:
    """Keep every fixture Git command inside the temporary repository."""
    assert not any(name.startswith("GIT_") for name in os.environ)
    assert (
        Path(_git(repository, "rev-parse", "--absolute-git-dir"))
        == (repository / ".git").resolve()
    )
    assert Path(_git(repository, "rev-parse", "--show-toplevel")) == repository.resolve()


def test_tight_ceiling_passes(repository: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Accept an index whose only large file has the smallest ceiling that holds it."""
    assert gate.tracked_sizes(repository) == {SMALL: 10, BIG: ONE_AND_A_HALF}
    assert _run(repository) == 0
    assert (
        f"2 files, 1 of at least {gate.MEBIBYTE} bytes ({ONE_AND_A_HALF} bytes); "
        "1 rows; 0 problems"
    ) in capsys.readouterr().out
    recorded = _ceiling(repository)
    assert recorded["total_files"] == 1
    assert recorded["total_mebibytes"] == 2


def test_new_file_at_the_threshold_is_rejected(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a staged file of exactly the threshold size that has no row."""
    _track(repository, "data/new.bin", gate.MEBIBYTE)
    assert _run(repository) == 1
    captured = capsys.readouterr()
    assert f"large file outside the ceiling: data/new.bin ({gate.MEBIBYTE} bytes)" in captured.err
    assert "1 problems" in captured.out


def test_new_file_one_byte_under_the_threshold_passes(repository: Path) -> None:
    """Accept a staged file that stays one byte under the threshold."""
    _track(repository, "data/new.bin", gate.MEBIBYTE - 1)
    assert _run(repository) == 0


def test_growth_inside_the_ceiling_passes(repository: Path) -> None:
    """Accept a recorded file that grows up to its whole-mebibyte ceiling."""
    _track(repository, BIG, 2 * gate.MEBIBYTE)
    assert _run(repository) == 0


def test_growth_above_the_ceiling_is_rejected(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a recorded file that grows one byte past its ceiling."""
    _track(repository, BIG, 2 * gate.MEBIBYTE + 1)
    assert _run(repository) == 1
    assert (
        f"tracked file grew above its ceiling: {BIG}: 2 MiB -> {2 * gate.MEBIBYTE + 1} bytes"
        in capsys.readouterr().err
    )


def test_ceiling_above_the_file_must_be_lowered(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require a tight row, then accept the lowered one."""
    gate.write_ceiling(repository / gate.DEFAULT_CEILING, gate.Ceiling(gate.MEBIBYTE, {BIG: 3}))
    assert _run(repository) == 1
    assert f"ceiling is above the file: {BIG}: 3 MiB -> 2 MiB" in capsys.readouterr().err
    assert _run(repository, "--lower") == 0
    assert _ceiling(repository)["files"] == {BIG: 2}
    assert _run(repository) == 0


def test_row_of_a_file_under_the_threshold_must_be_removed(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require the row to go when its file shrinks under the threshold."""
    _track(repository, BIG, 10)
    assert _run(repository) == 1
    assert (
        f"ceiling row is no longer needed: {BIG} is 10 bytes, under the threshold"
        in capsys.readouterr().err
    )
    assert _run(repository, "--lower") == 0
    recorded = _ceiling(repository)
    assert recorded["files"] == {}
    assert recorded["total_files"] == 0


def test_row_of_an_untracked_file_must_be_removed(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require the row to go when its file leaves the index."""
    _git(repository, "rm", "--quiet", "--cached", "--", BIG)
    assert _run(repository) == 1
    assert f"ceiling row names a file that is not tracked: {BIG}" in capsys.readouterr().err
    assert _run(repository, "--lower") == 0
    assert _ceiling(repository)["files"] == {}


def test_lowering_never_admits_or_raises(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep a new large file and a grown file rejected when the ceiling is lowered."""
    _track(repository, "data/new.bin", gate.MEBIBYTE)
    _track(repository, BIG, 3 * gate.MEBIBYTE)
    assert _run(repository, "--lower") == 1
    captured = capsys.readouterr().err
    assert "large file outside the ceiling: data/new.bin" in captured
    assert f"tracked file grew above its ceiling: {BIG}: 2 MiB" in captured
    assert _ceiling(repository)["files"] == {BIG: 2}


def test_sizes_come_from_the_index(repository: Path) -> None:
    """Ignore an unstaged edit and an untracked file: the gate judges what Git records."""
    (repository / BIG).write_bytes(bytes(5 * gate.MEBIBYTE))
    (repository / "data/untracked.bin").write_bytes(bytes(gate.MEBIBYTE))
    assert gate.tracked_sizes(repository) == {SMALL: 10, BIG: ONE_AND_A_HALF}
    assert _run(repository) == 0


def test_submodule_link_is_not_sized(repository: Path) -> None:
    """Leave a submodule link out: it names a commit of another repository, not a blob."""
    _git(repository, "update-index", "--add", "--cacheinfo", f"160000,{ABSENT_OBJECT},vendor/sub")
    assert "vendor/sub" in _git(repository, "ls-files")
    assert gate.tracked_sizes(repository) == {SMALL: 10, BIG: ONE_AND_A_HALF}
    assert _run(repository) == 0


def test_blob_absent_from_the_object_store_fails_closed(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail when the index names a blob Git cannot size instead of counting it as small."""
    _git(repository, "update-index", "--add", "--cacheinfo", f"100644,{ABSENT_OBJECT},ghost.bin")
    assert _run(repository) == 1
    assert (
        f"tracked file size ceiling failed: git cannot size the tracked file ghost.bin: "
        f"{ABSENT_OBJECT} missing"
    ) in capsys.readouterr().err


def test_unfinished_merge_carries_the_largest_stage(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Size a conflicted path by its largest stage, whichever stage Git lists last."""
    (repository / "large.tmp").write_bytes(bytes(gate.MEBIBYTE))
    large = _git(repository, "hash-object", "-w", "--", "large.tmp")
    small = _git(repository, "hash-object", "-w", "--", SMALL)
    _git(
        repository,
        "update-index",
        "--index-info",
        stdin=(
            f"100644 {small} 1\tdata/conflict.bin\n"
            f"100644 {large} 2\tdata/conflict.bin\n"
            f"100644 {small} 3\tdata/conflict.bin\n"
        ),
    )
    assert gate.tracked_sizes(repository)["data/conflict.bin"] == gate.MEBIBYTE
    assert _run(repository) == 1
    assert "large file outside the ceiling: data/conflict.bin" in capsys.readouterr().err


def test_absent_and_broken_repositories_and_absent_ceiling_are_errors(
    repository: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a missing directory, a directory Git refuses and a missing ceiling as failures."""
    with pytest.raises(ValueError, match="cannot run git"):
        gate.tracked_sizes(tmp_path / "absent")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: absent-git-directory\n", encoding="utf-8")
    with pytest.raises(ValueError, match="git ls-files --stage -z failed"):
        gate.tracked_sizes(broken)
    (repository / gate.DEFAULT_CEILING).unlink()
    assert _run(repository) == 1
    assert "tracked file size ceiling failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "tracked_file_size_ceiling_v0"}, "unsupported tracked file size ceiling"),
        ({"threshold_bytes": 0}, "threshold must be a positive integer"),
        ({"threshold_bytes": True}, "threshold must be a positive integer"),
        ({"threshold_bytes": "1048576"}, "threshold must be a positive integer"),
        ({"files": []}, "files must be an object"),
        ({"files": {"": 1}}, "not a path inside the repository"),
        ({"files": {"/data/big.bin": 2}}, "not a path inside the repository"),
        ({"files": {"data/../../big.bin": 2}}, "not a path inside the repository"),
        ({"files": {BIG: 0}}, "must be a positive integer number of mebibytes"),
        ({"files": {BIG: True}}, "must be a positive integer number of mebibytes"),
        ({"files": {BIG: "2"}}, "must be a positive integer number of mebibytes"),
    ],
)
def test_invalid_ceiling_is_refused(
    repository: Path, change: dict[str, object], message: str
) -> None:
    """Reject a ceiling with a stale schema or a malformed threshold, path or limit."""
    payload = _ceiling(repository)
    payload.update(change)
    _store(repository, payload)
    with pytest.raises(ValueError, match=message):
        gate.load_ceiling(repository / gate.DEFAULT_CEILING)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[]", "unsupported tracked file size ceiling"),
        ('{"schema": "a", "schema": "b"}', "duplicate key in ceiling: schema"),
        ("{", "Expecting property name"),
    ],
)
def test_malformed_ceiling_text_is_refused(repository: Path, text: str, message: str) -> None:
    """Reject a ceiling that is not one JSON object with unique keys."""
    (repository / gate.DEFAULT_CEILING).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        gate.load_ceiling(repository / gate.DEFAULT_CEILING)


@pytest.mark.parametrize(("size", "needed"), [(1, 1), (gate.MEBIBYTE, 1), (gate.MEBIBYTE + 1, 2)])
def test_needed_ceiling_rounds_up_to_whole_mebibytes(size: int, needed: int) -> None:
    """Round a size up to the next whole mebibyte and keep an exact multiple as it is."""
    assert gate.needed_mebibytes(size) == needed


def test_script_entry_point_reports_the_same_result(repository: Path) -> None:
    """Run the gate as a script, the way the workflow invokes it."""
    script = Path(gate.__file__).resolve()
    completed = subprocess.run(  # noqa: S603 - fixed argument vector
        [sys.executable, str(script), "--repo", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "0 problems" in completed.stdout
