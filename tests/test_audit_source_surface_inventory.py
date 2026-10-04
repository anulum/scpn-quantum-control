# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the source surface inventory gate
"""Exercise the source surface inventory gate on real Git repositories and workflows."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from tools import audit_source_surface_inventory as gate

WORKFLOW = """\
name: CI
on: push
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          ruff check \\
            src/
      - name: format
        run: ruff format --check src/
  notes: plain text instead of a job body
  empty:
    runs-on: ubuntu-latest
    steps: none
  mixed:
    runs-on: ubuntu-latest
    steps:
      - plain text instead of a step
      - run: echo done
"""
RUFF = {"workflow": "checks.yml", "job": "lint", "command": "ruff check src/"}
OUTSIDE = {".md": "document", ".yml": "configuration", ".json": "data and configuration"}


def _git(repository: Path, *arguments: str) -> str:
    """Run Git in ``repository`` and return its standard output."""
    completed = subprocess.run(
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


def _inventory(surfaces: Sequence[object], outside: dict[str, str] | None = None) -> str:
    """Return inventory JSON for ``surfaces`` and the ``outside`` kinds."""
    return json.dumps(
        {
            "schema": gate.SCHEMA,
            "surfaces": list(surfaces),
            "outside": OUTSIDE if outside is None else outside,
        }
    )


def _store(repository: Path, surfaces: list[dict[str, object]]) -> None:
    """Replace the inventory of ``repository`` and stage it."""
    _write(repository, str(gate.DEFAULT_POLICY), _inventory(surfaces))
    _git(repository, "add", "--all")


def _run(repository: Path, *arguments: str) -> int:
    """Run the gate's command line against ``repository``."""
    return gate.main(["--repo", str(repository), *arguments])


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Return a committed repository whose one Python root is gated."""
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--quiet")
    _write(root, "src/pkg/module.py", '"""Module."""\n')
    _write(root, "README.md", "# Title\n")
    _write(root, ".github/workflows/checks.yml", WORKFLOW)
    _store(root, [{"kind": ".py", "root": "src", "gates": [RUFF]}])
    _git(root, "commit", "--quiet", "--message", "initial")
    return root


def test_fixture_repository_is_separate_from_the_checkout(repository: Path) -> None:
    """The fixture repository's Git directory is its own."""
    assert (
        Path(_git(repository, "rev-parse", "--absolute-git-dir"))
        == (repository / ".git").resolve()
    )


def test_consistent_inventory_passes(repository: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A repository whose every file is classified and gated passes."""
    assert _run(repository) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "Source surface inventory: 4 tracked files; 1 surfaces "
        "(1 gated, 0 open, 0 evidence); 3 kinds outside; 0 problems\n"
    )


def test_new_language_fails(repository: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A tracked file of a kind the inventory does not know fails the gate."""
    _write(repository, "native/kernel.zig", "const a = 1;\n")
    _git(repository, "add", "--all")

    assert _run(repository) == 1

    assert (
        "unclassified file kind: .zig under native (1 files, e.g. native/kernel.zig)"
        in capsys.readouterr().err
    )


def test_new_root_of_a_known_language_fails(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A known source kind in a root without a row fails the gate."""
    _write(repository, "plugins/extra.py", '"""Extra."""\n')
    _write(repository, "plugins/more.PY", '"""More."""\n')
    _write(repository, "setup.py", '"""Setup."""\n')
    _git(repository, "add", "--all")

    assert _run(repository) == 1

    errors = capsys.readouterr().err
    assert (
        "source root without an owner: .py under plugins (2 files, e.g. plugins/extra.py)"
        in errors
    )
    assert "source root without an owner: .py under . (1 files, e.g. setup.py)" in errors


def test_file_without_suffix_is_classified_by_name(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file without a suffix is classified by its whole name."""
    _write(repository, "Dockerfile", "FROM scratch\n")
    _git(repository, "add", "--all")

    assert gate.classify("Dockerfile") == ("Dockerfile", ".")
    assert gate.classify("hooks/pre-push") == ("pre-push", "hooks")
    assert gate.classify("src/pkg/Module.PY") == (".py", "src")
    assert _run(repository) == 1
    assert "unclassified file kind: Dockerfile under ." in capsys.readouterr().err


def test_unused_rows_fail(repository: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A row or an outside kind that no tracked file uses must be removed."""
    _write(
        repository,
        str(gate.DEFAULT_POLICY),
        _inventory(
            [
                {"kind": ".py", "root": "src", "gates": [RUFF]},
                {"kind": ".rs", "root": "engine", "missing": "no lint"},
            ],
            {**OUTSIDE, ".csv": "data"},
        ),
    )

    assert _run(repository) == 1

    errors = capsys.readouterr().err
    assert "inventory row has no tracked file left: .rs under engine" in errors
    assert "outside kind has no tracked file left: .csv" in errors


def test_open_and_evidence_rows_are_reported(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Open rows are listed, and evidence rows under data need no gate."""
    _write(repository, "scripts/run.sh", "#!/bin/sh\n")
    _write(repository, "data/bench/source.go", "package main\n")
    _store(
        repository,
        [
            {"kind": ".py", "root": "src", "gates": [RUFF]},
            {"kind": ".sh", "root": "scripts", "gates": [RUFF], "missing": "no shell lint"},
            {"kind": ".go", "root": "data", "evidence": "recorded with its results"},
        ],
    )

    assert _run(repository) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "open: .sh under scripts: no shell lint\n" in captured.out
    assert "3 surfaces (1 gated, 1 open, 1 evidence)" in captured.out


def test_gate_references_are_checked_against_the_workflows(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate must name an existing workflow, job and command."""
    _write(repository, ".github/workflows/broken.yml", "jobs: [unclosed\n")
    _write(repository, ".github/workflows/listing.yml", "- one\n- two\n")
    _write(repository, ".github/workflows/nojobs.yml", "name: nothing\n")
    _store(
        repository,
        [
            {
                "kind": ".py",
                "root": "src",
                "gates": [
                    RUFF,
                    {
                        "workflow": "checks.yml",
                        "job": "lint",
                        "command": "ruff format --check src/",
                    },
                    {"workflow": "checks.yml", "job": "lint", "command": "mypy --strict src/"},
                    {"workflow": "checks.yml", "job": "types", "command": "mypy"},
                    {"workflow": "checks.yml", "job": "notes", "command": "plain"},
                    {"workflow": "checks.yml", "job": "empty", "command": "none"},
                    {"workflow": "checks.yml", "job": "mixed", "command": "echo done"},
                    {"workflow": "absent.yml", "job": "lint", "command": "ruff"},
                    {"workflow": "absent.yml", "job": "other", "command": "ruff"},
                    {"workflow": "broken.yml", "job": "lint", "command": "ruff"},
                    {"workflow": "listing.yml", "job": "lint", "command": "ruff"},
                    {"workflow": "nojobs.yml", "job": "lint", "command": "ruff"},
                ],
            }
        ],
    )

    assert _run(repository) == 1

    errors = capsys.readouterr().err.splitlines()
    assert ".py under src: job lint of checks.yml does not run: mypy --strict src/" in errors
    assert ".py under src: workflow checks.yml has no job types" in errors
    assert ".py under src: job notes of checks.yml does not run: plain" in errors
    assert ".py under src: job empty of checks.yml does not run: none" in errors
    assert sum("cannot read workflow absent.yml" in line for line in errors) == 2
    assert sum("cannot read workflow broken.yml" in line for line in errors) == 1
    assert ".py under src: workflow listing.yml has no jobs mapping" in errors
    assert ".py under src: workflow nojobs.yml has no jobs mapping" in errors
    assert len(errors) == 9


def test_job_commands_join_continuations(repository: Path) -> None:
    """Run steps are joined with line continuations and extra whitespace removed."""
    commands = gate.job_commands(repository, "checks.yml")

    assert commands == {
        "lint": "ruff check src/ ruff format --check src/",
        "notes": "",
        "empty": "",
        "mixed": "echo done",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "inventory is not valid JSON"),
        ("[]", "inventory schema must be source_surface_inventory_v1"),
        (json.dumps({"schema": "other"}), "inventory schema must be"),
        (
            json.dumps({"schema": gate.SCHEMA, "surfaces": {}, "outside": {}}),
            "inventory needs a surfaces list and an outside object",
        ),
        (
            json.dumps({"schema": gate.SCHEMA, "surfaces": [], "outside": []}),
            "inventory needs a surfaces list and an outside object",
        ),
        (_inventory(["row"]), "surface 0: a surface must be an object"),
        (_inventory([{"root": "src"}]), "surface 0: kind must be a non-empty string"),
        (_inventory([{"kind": ".py", "root": " "}]), "surface 0: root must be a non-empty string"),
        (_inventory([{"kind": 3, "root": "src"}]), "surface 0: kind must be a string"),
        (
            _inventory([{"kind": ".py", "root": "src", "gates": "lint"}]),
            ".py under src: gates must be a list",
        ),
        (
            _inventory([{"kind": ".py", "root": "src", "gates": ["lint"]}]),
            ".py under src: a gate must be an object",
        ),
        (
            _inventory([{"kind": ".py", "root": "src", "gates": [{"workflow": "checks.yml"}]}]),
            ".py under src: job must be a non-empty string",
        ),
        (
            _inventory([{"kind": ".py", "root": "src"}]),
            ".py under src: a row needs a gate, missing work or an evidence reason",
        ),
        (
            _inventory([{"kind": ".go", "root": "data", "evidence": "kept", "missing": "lint"}]),
            ".go under data: an evidence row records neither gates nor missing work",
        ),
        (
            _inventory([{"kind": ".go", "root": "data", "evidence": "kept", "gates": [RUFF]}]),
            ".go under data: an evidence row records neither gates nor missing work",
        ),
        (
            _inventory([{"kind": ".go", "root": "src", "evidence": "kept"}]),
            ".go under src: evidence rows are allowed under data only",
        ),
        (
            _inventory(
                [
                    {"kind": ".py", "root": "src", "gates": [RUFF]},
                    {"kind": ".py", "root": "src", "missing": "lint"},
                ]
            ),
            ".py under src: repeated row",
        ),
        (
            _inventory([{"kind": ".py", "root": "src", "gates": [RUFF]}], {".md": ""}),
            "outside kind .md: the class must be a non-empty string",
        ),
        (
            _inventory([{"kind": ".py", "root": "src", "gates": [RUFF]}], {".py": "source"}),
            "kind listed as a surface and as outside: .py",
        ),
    ],
)
def test_malformed_inventory_is_refused(text: str, message: str) -> None:
    """Every malformed inventory is refused with a message naming the fault."""
    with pytest.raises(ValueError, match=re.escape(message)):
        gate.parse_policy(text)


def test_surface_states() -> None:
    """A row is open when work is missing, evidence when recorded, gated otherwise."""
    policy = gate.parse_policy(
        _inventory(
            [
                {"kind": ".py", "root": "src", "gates": [RUFF]},
                {"kind": ".ts", "root": "web", "gates": [RUFF], "missing": " no lint "},
                {"kind": ".go", "root": "data", "evidence": "recorded"},
            ]
        )
    )

    assert policy.surfaces[".py", "src"].state == "gated"
    assert policy.surfaces[".ts", "web"].state == "open"
    assert policy.surfaces[".ts", "web"].missing == "no lint"
    assert policy.surfaces[".go", "data"].state == "evidence"
    assert policy.surfaces[".py", "src"].gates == (
        gate.Gate("checks.yml", "lint", "ruff check src/"),
    )
    assert policy.outside == OUTSIDE


def test_unreadable_inventory_fails(repository: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing inventory file fails the gate with one message."""
    assert _run(repository, "--policy", "tools/absent.json") == 1

    assert "source surface inventory failed: cannot read the inventory" in capsys.readouterr().err


def test_directory_without_a_repository_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Outside a Git repository the tracked files cannot be listed.

    The ceiling keeps Git from finding a repository above the temporary
    directory, wherever the test session places it.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    _write(
        tmp_path,
        str(gate.DEFAULT_POLICY),
        _inventory([{"kind": ".py", "root": "src", "gates": [RUFF]}]),
    )

    assert _run(tmp_path) == 1

    assert "source surface inventory failed: git ls-files failed" in capsys.readouterr().err


def test_missing_git_executable_fails(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a Git executable the gate reports that Git cannot run."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(ValueError, match="cannot run git"):
        gate.tracked_files(repository)


def test_first_introduction_has_no_base(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A comparison revision without an inventory file imposes no debt rule."""
    _git(repository, "rm", "--quiet", "--cached", str(gate.DEFAULT_POLICY))
    _git(repository, "commit", "--quiet", "--message", "before the gate")
    _write(repository, "scripts/run.sh", "#!/bin/sh\n")
    _store(
        repository,
        [
            {"kind": ".py", "root": "src", "gates": [RUFF]},
            {"kind": ".sh", "root": "scripts", "missing": "no shell lint"},
        ],
    )

    assert gate.policy_at(repository, "HEAD", gate.DEFAULT_POLICY) is None
    assert _run(repository, "--changed-against", "HEAD") == 0
    assert capsys.readouterr().err == ""


def test_new_open_row_is_refused_against_the_base(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row may not be open unless it was already open at the comparison revision."""
    _write(repository, "scripts/run.sh", "#!/bin/sh\n")
    _write(repository, "web/app.ts", "export {};\n")
    _store(
        repository,
        [
            {"kind": ".py", "root": "src", "gates": [RUFF]},
            {"kind": ".sh", "root": "scripts", "missing": "no shell lint"},
            {"kind": ".ts", "root": "web", "gates": [RUFF]},
        ],
    )
    _git(repository, "commit", "--quiet", "--message", "record the debt")
    _write(repository, "tools/helper.sh", "#!/bin/sh\n")
    _store(
        repository,
        [
            {"kind": ".py", "root": "src", "gates": [RUFF]},
            {"kind": ".sh", "root": "scripts", "missing": "no shell lint"},
            {"kind": ".sh", "root": "tools", "missing": "no shell lint"},
            {"kind": ".ts", "root": "web", "gates": [RUFF], "missing": "no format gate"},
        ],
    )

    assert _run(repository) == 0
    capsys.readouterr()
    assert _run(repository, "--changed-against", "HEAD") == 1

    errors = capsys.readouterr().err.splitlines()
    assert errors == [
        "surface admitted without full enforcement: .sh under tools (base: absent); "
        "missing: no shell lint",
        "surface admitted without full enforcement: .ts under web (base: gated); "
        "missing: no format gate",
    ]


def test_comparison_revision_must_be_a_commit(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown comparison revision fails the gate instead of skipping the rule."""
    assert _run(repository, "--changed-against", "no-such-revision") == 1

    assert (
        "source surface inventory failed: comparison revision is not a commit: no-such-revision"
        in capsys.readouterr().err
    )
