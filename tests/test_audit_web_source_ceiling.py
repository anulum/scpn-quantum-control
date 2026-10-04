# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the web source lint and format ceiling gate
"""Exercise the web source ceiling with the pinned Biome release and real Git repositories."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import audit_web_source_ceiling as gate

REPOSITORY = Path(__file__).resolve().parents[1]
INSTALLED = REPOSITORY / gate.WEB_ROOT / "node_modules" / ".bin" / "biome"
pytestmark = pytest.mark.skipif(
    not INSTALLED.is_file(), reason="the web workspace is not installed"
)

CLEAN = "export const answer: number = 42;\n"
ONE_FINDING = "export function first(values: number[]): number {\n  return values[0]!;\n}\n"
TWO_FINDINGS = (
    "export function ends(values: number[]): number {\n  return values[0]! + values[1]!;\n}\n"
)
UNFORMATTED = "export const  spaced = 1;\n"
STYLESHEET = "a {\n  color: red;\n}\n"
BROKEN = "export const = ;\n"


def _git(repo: Path, *arguments: str) -> str:
    """Run Git in ``repo`` with a fixed identity and return its output."""
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _executable(path: Path, script: str) -> Path:
    """Write a POSIX shell ``script`` to ``path`` and make it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _workspace(root: Path, sources: dict[str, str], pin: str | None = None) -> Path:
    """Create a committed repository with a web workspace that runs the real Biome.

    Parameters
    ----------
    root
        Empty directory that becomes the repository.
    sources
        Web sources by path relative to the workspace.
    pin
        Release written into the manifest; the installed release when omitted.

    Returns
    -------
    Path
        The repository root.

    """
    web = root / gate.WEB_ROOT
    web.mkdir(parents=True)
    release = gate.pinned_release(REPOSITORY) if pin is None else pin
    (web / "package.json").write_text(
        json.dumps({"devDependencies": {gate.DISTRIBUTION: release}}), encoding="utf-8"
    )
    shutil.copyfile(REPOSITORY / gate.WEB_ROOT / "biome.jsonc", web / "biome.jsonc")
    (web / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    _executable(web / "node_modules" / ".bin" / "biome", f'exec "{INSTALLED}" "$@"')
    for name, text in sources.items():
        path = web / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "--message", "fixture")
    return root


def _record(repo: Path, **changes: Any) -> Path:
    """Write the measurement of ``repo`` as its ceiling, with optional field changes."""
    executable = gate.workspace_executable(repo)
    measured = gate.measure(repo, executable)
    path = repo / gate.DEFAULT_CEILING
    path.parent.mkdir(parents=True, exist_ok=True)
    gate.write_ceiling(
        path,
        gate.Ceiling(
            gate.installed_release(repo, executable),
            _git(repo, "rev-parse", "HEAD"),
            measured.lint,
            measured.unformatted,
            (),
        ),
    )
    if changes:
        body = json.loads(path.read_text(encoding="utf-8"))
        body.update(changes)
        path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_measure_counts_findings_and_lists_unformatted_sources(tmp_path: Path) -> None:
    """Lint findings are counted per file; unformatted files are listed; other files are ignored."""
    repo = _workspace(
        tmp_path,
        {
            "src/clean.ts": CLEAN,
            "src/one.ts": ONE_FINDING,
            "src/two.tsx": TWO_FINDINGS,
            "src/spaced.ts": UNFORMATTED,
            "src/sheet.css": STYLESHEET,
            "src/note.md": "# not a web source\n",
            "index.html": "<p>not measured</p>\n",
        },
    )
    executable = gate.workspace_executable(repo)

    assert gate.installed_release(repo, executable) == gate.pinned_release(REPOSITORY)
    assert gate.tracked_sources(repo) == [
        "studio-web/src/clean.ts",
        "studio-web/src/one.ts",
        "studio-web/src/sheet.css",
        "studio-web/src/spaced.ts",
        "studio-web/src/two.tsx",
    ]
    assert gate.measure(repo, executable) == gate.Measurement(
        {"studio-web/src/one.ts": 1, "studio-web/src/two.tsx": 2},
        ("studio-web/src/spaced.ts",),
    )


def test_measure_of_a_workspace_without_sources_is_empty(tmp_path: Path) -> None:
    """A workspace with no tracked source yields an empty measurement without running Biome."""
    repo = _workspace(tmp_path, {"README.md": "no sources\n"})

    assert gate.measure(repo, tmp_path / "absent") == gate.Measurement({}, ())


def test_parse_error_fails_the_gate_instead_of_being_counted(tmp_path: Path) -> None:
    """A source Biome cannot parse stops the measurement with the diagnostic's category."""
    repo = _workspace(tmp_path, {"src/broken.ts": BROKEN})

    with pytest.raises(
        ValueError, match=r"biome lint reported parse in studio-web/src/broken\.ts"
    ):
        gate.measure(repo, gate.workspace_executable(repo))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("echo not-json", "gave no usable report"),
        ("echo '{}'", "gave no usable report"),
        (
            """echo '{"summary": {"diagnosticsNotPrinted": 3}, "diagnostics": []}'""",
            "the report omits diagnostics",
        ),
        (
            """echo '{"summary": {"diagnosticsNotPrinted": 0}, "diagnostics": [{"category": "lint/x"}]}'""",
            "gave no usable report",
        ),
        (
            """echo '{"summary": {"diagnosticsNotPrinted": 0}, "diagnostics": """
            """[{"category": "lint/x", "location": {"path": "src/other.ts"}}]}'""",
            r"reported lint/x outside the sources: src/other\.ts",
        ),
    ],
)
def test_unusable_reports_are_refused(tmp_path: Path, output: str, message: str) -> None:
    """A report that is not the expected document, or names a foreign file, is refused.

    Parameters
    ----------
    tmp_path
        Directory that becomes the repository.
    output
        Shell command of the stand-in executable that prints the report.
    message
        Pattern the refusal must match.

    """
    repo = _workspace(tmp_path, {"src/clean.ts": CLEAN})
    stand_in = _executable(tmp_path / "stand-in", output)

    with pytest.raises(ValueError, match=message):
        gate.measure(repo, stand_in)


def test_format_report_with_another_category_fails_the_gate(tmp_path: Path) -> None:
    """A format run that reports anything but a format difference stops the measurement."""
    repo = _workspace(tmp_path, {"src/clean.ts": CLEAN})
    stand_in = _executable(
        tmp_path / "stand-in",
        """if [ "$1" = lint ]; then
  echo '{"summary": {"diagnosticsNotPrinted": 0}, "diagnostics": []}'
else
  echo '{"summary": {"diagnosticsNotPrinted": 0}, "diagnostics": """
        """[{"category": "internalError/io", "location": {"path": "src/clean.ts"}}]}'
fi""",
    )

    with pytest.raises(ValueError, match=r"biome format reported internalError/io in studio-web"):
        gate.measure(repo, stand_in)


def test_executable_and_release_failures_are_reported(tmp_path: Path) -> None:
    """A missing installation, a silent or failing executable and a file that cannot run are refused."""
    repo = _workspace(tmp_path / "repo", {"src/clean.ts": CLEAN})
    silent = _executable(tmp_path / "silent", "exit 0")
    failing = _executable(tmp_path / "failing", "echo 'Version: 9.9.9'; exit 2")
    plain = tmp_path / "plain"
    plain.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    plain.chmod(0o644)
    (repo / gate.WEB_ROOT / "node_modules" / ".bin" / "biome").unlink()

    with pytest.raises(ValueError, match="is not installed in studio-web"):
        gate.workspace_executable(repo)
    for executable in (silent, failing):
        with pytest.raises(ValueError, match="--version reported no release"):
            gate.installed_release(repo, executable)
    with pytest.raises(ValueError, match="cannot run"):
        gate.installed_release(repo, plain)
    with pytest.raises(ValueError, match="git ls-files"):
        gate.tracked_sources(tmp_path)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (None, "cannot read the web workspace manifest"),
        ("{", "cannot read the web workspace manifest"),
        ('{"devDependencies": {}}', "does not pin one exact"),
        ('{"devDependencies": {"@biomejs/biome": "^2.5.15"}}', "does not pin one exact"),
        ('{"devDependencies": {"@biomejs/biome": 2}}', "does not pin one exact"),
    ],
)
def test_pinned_release_requires_one_exact_release(
    tmp_path: Path, manifest: str | None, message: str
) -> None:
    """A missing or unreadable manifest, a missing pin and a version range are refused.

    Parameters
    ----------
    tmp_path
        Directory that holds the workspace manifest.
    manifest
        Manifest text, or ``None`` for no manifest at all.
    message
        Pattern the refusal must match.

    """
    web = tmp_path / gate.WEB_ROOT
    web.mkdir()
    if manifest is not None:
        (web / "package.json").write_text(manifest, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        gate.pinned_release(tmp_path)


def test_compare_reports_every_departure_from_the_ceiling() -> None:
    """New, grown, fallen and vanished rows and both format departures are each reported."""
    ceiling = gate.Ceiling(
        "2.5.15",
        "0" * 40,
        {"studio-web/a.ts": 2, "studio-web/b.ts": 2, "studio-web/c.ts": 2, "studio-web/d.ts": 1},
        ("studio-web/a.ts", "studio-web/gone.ts"),
        (),
    )
    measured = gate.Measurement(
        {"studio-web/a.ts": 2, "studio-web/b.ts": 3, "studio-web/c.ts": 1, "studio-web/new.ts": 4},
        ("studio-web/a.ts", "studio-web/new.ts"),
    )

    assert gate.compare(measured, ceiling, "2.5.15") == [
        "lint findings grew: studio-web/b.ts: 2 -> 3",
        "ceiling is above the measurement: studio-web/c.ts: 2 -> 1; lower it with --lower",
        "web source outside the ceiling has lint findings: studio-web/new.ts (4)",
        "ceiling row has no lint findings left: studio-web/d.ts; remove it with --lower",
        "not formatted: studio-web/new.ts; run biome format --write on it",
        "ceiling lists a formatted or removed file: studio-web/gone.ts; remove it with --lower",
    ]
    assert (
        gate.compare(gate.Measurement(ceiling.lint, ceiling.unformatted), ceiling, "2.5.15") == []
    )
    assert gate.compare(measured, ceiling, "2.6.0") == [
        "ceiling was measured with @biomejs/biome 2.5.15, this is 2.6.0; findings are not "
        "comparable across releases, record a new measurement with --rebaseline"
    ]


def test_changed_sources_must_be_clean_and_formatted(tmp_path: Path) -> None:
    """Sources that differ from a revision are listed and each remaining defect is reported."""
    repo = _workspace(
        tmp_path, {"src/kept.ts": ONE_FINDING, "src/edited.ts": CLEAN, "src/note.md": "x\n"}
    )
    web = repo / gate.WEB_ROOT
    (web / "src" / "edited.ts").write_text(ONE_FINDING.replace("{\n", "{\n\n"), encoding="utf-8")
    (web / "src" / "added.css").write_text(STYLESHEET, encoding="utf-8")
    (web / "src" / "note.md").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "--all")
    measured = gate.measure(repo, gate.workspace_executable(repo))
    changed = gate.changed_sources(repo, "HEAD")

    assert changed == ["studio-web/src/added.css", "studio-web/src/edited.ts"]
    assert gate.check_changed(measured, changed) == [
        "changed web source must have no lint finding: studio-web/src/edited.ts (1)",
        "changed web source must be formatted: studio-web/src/edited.ts",
    ]
    with pytest.raises(ValueError, match="git diff"):
        gate.changed_sources(repo, "no-such-revision")


def test_lower_follows_the_measurement_and_never_admits_debt() -> None:
    """Lowering adopts a smaller measurement and refuses a grown or newly unformatted file."""
    ceiling = gate.Ceiling(
        "2.5.15", "old", {"studio-web/a.ts": 3}, ("studio-web/a.ts", "studio-web/b.ts"), ()
    )
    smaller = gate.Measurement({"studio-web/a.ts": 1}, ("studio-web/b.ts",))

    assert gate.lower(smaller, ceiling, "new") == gate.Ceiling(
        "2.5.15", "new", {"studio-web/a.ts": 1}, ("studio-web/b.ts",), ()
    )
    with pytest.raises(ValueError, match=r"debt grew: studio-web/a\.ts, studio-web/c\.ts"):
        gate.lower(gate.Measurement({"studio-web/a.ts": 4}, ("studio-web/c.ts",)), ceiling, "new")


def test_rebaseline_needs_a_new_release_and_keeps_history() -> None:
    """A new release records the measurement and the previous totals; the same release is refused."""
    ceiling = gate.Ceiling(
        "2.5.15", "old", {"studio-web/a.ts": 3}, ("studio-web/a.ts",), ({"release": "2.4.0"},)
    )
    measured = gate.Measurement({"studio-web/a.ts": 5}, ())

    assert gate.rebaseline(measured, ceiling, "2.6.0", "new") == gate.Ceiling(
        "2.6.0",
        "new",
        {"studio-web/a.ts": 5},
        (),
        (
            {"release": "2.4.0"},
            {
                "release": "2.5.15",
                "measured_commit": "old",
                "total_lint_findings": 3,
                "total_lint_files": 1,
                "total_unformatted_files": 1,
            },
        ),
    )
    with pytest.raises(ValueError, match="did not change"):
        gate.rebaseline(measured, ceiling, "2.5.15", "new")


def test_written_ceiling_reads_back_unchanged(tmp_path: Path) -> None:
    """A ceiling survives a write and a read, with sorted rows and totals in the file."""
    ceiling = gate.Ceiling(
        "2.5.15",
        "a" * 40,
        {"studio-web/b.ts": 2, "studio-web/a.tsx": 1},
        ("studio-web/a.tsx", "studio-web/c.css"),
        ({"release": "2.4.0"},),
    )
    path = tmp_path / "ceiling.json"

    gate.write_ceiling(path, ceiling)
    body = json.loads(path.read_text(encoding="utf-8"))

    assert gate.load_ceiling(path) == ceiling
    assert list(body["lint"]) == ["studio-web/a.tsx", "studio-web/b.ts"]
    assert (body["total_lint_findings"], body["total_lint_files"]) == (3, 2)
    assert body["total_unformatted_files"] == 2
    assert path.read_text(encoding="utf-8").endswith("}\n")


_VALID: dict[str, Any] = {
    "schema": gate.SCHEMA,
    "release": "2.5.15",
    "measured_commit": "a" * 40,
    "lint": {"studio-web/a.ts": 1},
    "unformatted": ["studio-web/a.ts"],
    "history": [],
}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema": "other"}, "unsupported web source ceiling schema"),
        ({"release": "2.5"}, "must record the Biome release"),
        ({"release": 2}, "must record the Biome release"),
        ({"measured_commit": " "}, "must record the commit"),
        ({"lint": []}, "lint counts must be an object"),
        ({"lint": {"src/a.ts": 1}}, "not a web source: src/a.ts"),
        ({"lint": {"studio-web/../a.ts": 1}}, "not a web source"),
        ({"lint": {"studio-web/a.md": 1}}, "not a web source"),
        ({"lint": {"studio-web/a.ts": 0}}, "must be a positive integer"),
        ({"lint": {"studio-web/a.ts": True}}, "must be a positive integer"),
        ({"unformatted": "studio-web/a.ts"}, "must be a list of paths"),
        ({"unformatted": [1]}, "must be a list of paths"),
        ({"unformatted": ["docs/a.css"]}, "not a web source: docs/a.css"),
        ({"unformatted": ["studio-web/b.ts", "studio-web/a.ts"]}, "sorted and unique"),
        ({"unformatted": ["studio-web/a.ts", "studio-web/a.ts"]}, "sorted and unique"),
        ({"history": {}}, "history must be a list of objects"),
        ({"history": [1]}, "history must be a list of objects"),
    ],
)
def test_invalid_ceilings_are_refused(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    """Every malformed field of a ceiling document is refused with its own message.

    Parameters
    ----------
    tmp_path
        Directory that receives the document.
    changes
        Fields replaced in an otherwise valid document.
    message
        Pattern the refusal must match.

    """
    path = tmp_path / "ceiling.json"
    path.write_text(json.dumps({**_VALID, **changes}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        gate.load_ceiling(path)


def test_ceiling_with_a_repeated_key_or_another_document_is_refused(tmp_path: Path) -> None:
    """A repeated key and a document that is not an object are both refused."""
    path = tmp_path / "ceiling.json"
    path.write_text(
        '{"schema": "web_source_ceiling_v1", "lint": {"studio-web/a.ts": 1, "studio-web/a.ts": 2}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate key in ceiling: studio-web/a\.ts"):
        gate.load_ceiling(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported web source ceiling schema"):
        gate.load_ceiling(path)


def test_main_passes_on_an_exact_ceiling_and_reports_departures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate passes when the tree equals its ceiling and fails with one line per departure."""
    repo = _workspace(tmp_path, {"src/one.ts": ONE_FINDING, "src/spaced.ts": UNFORMATTED})
    _record(repo)

    assert gate.main(["--repo", str(repo)]) == 0
    assert capsys.readouterr().out == (
        "Web source ceiling: 1 lint findings in 1 files, 1 unformatted files; ceiling 1 in 1, "
        "1 unformatted; 0 problems\n"
    )

    web = repo / gate.WEB_ROOT / "src"
    (web / "one.ts").write_text(TWO_FINDINGS, encoding="utf-8")
    (web / "fresh.ts").write_text(UNFORMATTED, encoding="utf-8")
    _git(repo, "add", "--all")

    assert gate.main(["--repo", str(repo), "--changed-against", "HEAD"]) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "lint findings grew: studio-web/src/one.ts: 1 -> 2",
        "not formatted: studio-web/src/fresh.ts; run biome format --write on it",
        "changed web source must be formatted: studio-web/src/fresh.ts",
        "changed web source must have no lint finding: studio-web/src/one.ts (2)",
    ]
    assert captured.out.endswith("4 problems\n")


def test_main_lowers_the_ceiling_to_a_smaller_measurement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cleaning a file makes the gate ask for a lower ceiling, and ``--lower`` writes it."""
    repo = _workspace(tmp_path, {"src/one.ts": ONE_FINDING, "src/spaced.ts": UNFORMATTED})
    path = _record(repo)
    web = repo / gate.WEB_ROOT / "src"
    (web / "one.ts").write_text(CLEAN, encoding="utf-8")
    (web / "spaced.ts").write_text(CLEAN, encoding="utf-8")

    assert gate.main(["--repo", str(repo)]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "ceiling row has no lint findings left: studio-web/src/one.ts; remove it with --lower",
        "ceiling lists a formatted or removed file: studio-web/src/spaced.ts; remove it with --lower",
    ]
    assert gate.main(["--repo", str(repo), "--lower"]) == 0
    assert capsys.readouterr().out == (
        "Web source ceiling written: 0 lint findings in 0 files; 0 unformatted files\n"
    )
    lowered = gate.load_ceiling(path)
    assert (lowered.lint, lowered.unformatted) == ({}, ())
    assert lowered.measured_commit == _git(repo, "rev-parse", "HEAD")
    assert gate.main(["--repo", str(repo)]) == 0


def test_main_rebaselines_only_after_a_release_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ceiling from another release refuses comparison and lowering, and accepts a rebaseline."""
    repo = _workspace(tmp_path, {"src/one.ts": ONE_FINDING})
    path = _record(repo, release="2.4.0")

    assert gate.main(["--repo", str(repo)]) == 1
    assert "record a new measurement with --rebaseline" in capsys.readouterr().err
    assert gate.main(["--repo", str(repo), "--lower"]) == 1
    assert "use --rebaseline, not --lower" in capsys.readouterr().err
    assert gate.main(["--repo", str(repo), "--rebaseline"]) == 0
    capsys.readouterr()

    rebased = gate.load_ceiling(path)
    assert rebased.release == gate.pinned_release(REPOSITORY)
    assert rebased.history == (
        {
            "release": "2.4.0",
            "measured_commit": _git(repo, "rev-parse", "HEAD"),
            "total_lint_findings": 1,
            "total_lint_files": 1,
            "total_unformatted_files": 0,
        },
    )
    assert gate.main(["--repo", str(repo), "--rebaseline"]) == 1
    assert "can only be lowered" in capsys.readouterr().err


def test_main_refuses_another_installed_release_and_a_missing_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An installation that is not the pinned release, and an absent ceiling, fail the gate."""
    repo = _workspace(tmp_path / "pinned", {"src/clean.ts": CLEAN}, pin="2.4.0")
    _record(repo)

    assert gate.main(["--repo", str(repo)]) == 1
    assert "is not the pinned release 2.4.0; reinstall studio-web" in capsys.readouterr().err

    bare = _workspace(tmp_path / "bare", {"src/clean.ts": CLEAN})
    assert gate.main(["--repo", str(bare)]) == 1
    assert capsys.readouterr().err.startswith("web source ceiling failed: ")
