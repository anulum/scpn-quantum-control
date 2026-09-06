# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Documentation-debt baseline tests
"""A baseline is its provenance, so the provenance is what is tested here.

The recorded 2026-09-04 figure could not be reproduced from what was kept beside
it: it listed ``D413`` among the rules, and the same scope measured with Ruff
0.16.4 reports ``D421`` and no ``D413``. The count was fine; the missing tool
version is what made it unrepeatable.

These tests therefore pin the command, the scopes and the shape of the record,
and exercise the parsing against injected Ruff output rather than by scanning
the repository, so they stay fast and deterministic. One test does run the real
scan, because a generator that has never been run end to end is not evidence
either; it asserts internal consistency rather than a fixed number, since the
number moves whenever a docstring is written.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.documentation_debt_baseline import (
    DOCUMENTATION_DEBT_SCOPES,
    PYDOCSTYLE_CONVENTION,
    baseline_command,
    main,
    measure_documentation_debt,
)


class _FakeCompleted:
    """Stand-in for :class:`subprocess.CompletedProcess`."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        """Store the fields the module reads.

        Parameters
        ----------
        stdout, stderr, returncode
            Values the fake process reports.

        """
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestTheRecordedCommand:
    """What is recorded must be exactly what is run."""

    def test_the_command_is_isolated_preview_numpy_json(self) -> None:
        """Every flag here changes the number, so every flag is pinned."""
        command = baseline_command("/usr/bin/python3")

        assert command[:8] == (
            "/usr/bin/python3",
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--preview",
            "--select",
            "D",
        )
        assert "--config" in command
        assert PYDOCSTYLE_CONVENTION in command
        assert command[command.index("--output-format") + 1] == "json"

    def test_the_scopes_are_the_ones_the_review_named(self) -> None:
        """The scope list is part of the measurement, not a convenience."""
        assert command_scopes(baseline_command("python")) == list(DOCUMENTATION_DEBT_SCOPES)
        assert DOCUMENTATION_DEBT_SCOPES[:3] == ("src", "tests", "oscillatools")

    def test_isolation_is_deliberate(self) -> None:
        """A repository-configured run answers a different question.

        Per-file ignores hide tests and oscillatools, which is exactly the debt a
        full-scope baseline exists to show, so the two runs must never be quoted
        as one measurement.
        """
        assert "--isolated" in baseline_command("python")


def command_scopes(command: tuple[str, ...]) -> list[str]:
    """Return the trailing path arguments of a baseline command.

    Parameters
    ----------
    command
        A command from :func:`baseline_command`.

    Returns
    -------
    list
        The scope arguments, in order.

    """
    return list(command[command.index("json") + 1 :])


class TestParsingInjectedOutput:
    """Counting, bucketing and failure handling, without scanning anything."""

    @staticmethod
    def _install(
        monkeypatch: pytest.MonkeyPatch, root: Path, findings: list[dict[str, Any]], code: int = 1
    ) -> None:
        """Replace the subprocess calls the module makes.

        Parameters
        ----------
        monkeypatch
            Fixture used to install the fakes.
        root
            Repository root the fake findings live under.
        findings
            Ruff-shaped finding records.
        code
            Exit status the fake Ruff reports.

        """

        def fake_run(command: list[str], **kwargs: Any) -> _FakeCompleted:
            if "--version" in command:
                return _FakeCompleted(stdout="ruff 9.9.9\n")
            if command[:2] == ["git", "-C"]:
                return _FakeCompleted(stdout="0" * 40 + "\n")
            return _FakeCompleted(stdout=json.dumps(findings), returncode=code)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.chdir(root)

    def test_counts_findings_files_scopes_and_rules(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two findings in one file count as two findings and one file.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """
        findings = [
            {"filename": str(tmp_path / "src" / "a.py"), "code": "D103"},
            {"filename": str(tmp_path / "src" / "a.py"), "code": "D103"},
            {"filename": str(tmp_path / "tests" / "b.py"), "code": "D102"},
        ]
        self._install(monkeypatch, tmp_path, findings)

        baseline = measure_documentation_debt(tmp_path)

        assert baseline.total_findings == 3
        assert baseline.total_files == 2
        assert baseline.by_scope == {"src": 2, "tests": 1}
        assert baseline.by_rule == {"D103": 2, "D102": 1}

    def test_provenance_is_carried(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A count without its tool version is what this card is about.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """
        self._install(monkeypatch, tmp_path, [])

        baseline = measure_documentation_debt(tmp_path)

        assert baseline.ruff_version == "ruff 9.9.9"
        assert baseline.python_version.startswith("Python ")
        assert baseline.source_sha == "0" * 40
        assert baseline.scopes == DOCUMENTATION_DEBT_SCOPES
        assert "--isolated" in baseline.command

    def test_a_file_outside_the_root_is_bucketed_visibly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An unexpected path must be visible, not silently attributed.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """
        self._install(monkeypatch, tmp_path, [{"filename": "/elsewhere/x.py", "code": "D100"}])

        assert measure_documentation_debt(tmp_path).by_scope == {"<outside>": 1}

    def test_a_broken_invocation_is_not_recorded_as_a_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Exit 2 is a broken command; zero findings would be a lie.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """
        self._install(monkeypatch, tmp_path, [], code=2)

        with pytest.raises(RuntimeError, match="not a findings result"):
            measure_documentation_debt(tmp_path)

    def test_an_unavailable_git_leaves_the_sha_unknown(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing SHA is stated, never invented.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """

        def fake_run(command: list[str], **kwargs: Any) -> _FakeCompleted:
            if "--version" in command:
                return _FakeCompleted(stdout="ruff 9.9.9\n")
            if command[:2] == ["git", "-C"]:
                raise OSError("no git here")
            return _FakeCompleted(stdout="[]", returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert measure_documentation_debt(tmp_path).source_sha == "unknown"

    def test_json_round_trips_and_is_sorted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The artefact must diff cleanly between refreshes.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """
        self._install(
            monkeypatch, tmp_path, [{"filename": str(tmp_path / "src" / "a.py"), "code": "D103"}]
        )
        text = measure_documentation_debt(tmp_path).to_json()

        restored = json.loads(text)

        assert text.endswith("\n")
        assert list(restored) == sorted(restored)
        assert restored["total_findings"] == 1

    def test_main_writes_the_artefact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The entry point is what a refresh actually runs.

        Parameters
        ----------
        monkeypatch, tmp_path
            Fixtures.

        """
        self._install(monkeypatch, tmp_path, [])
        destination = tmp_path / "baseline.json"

        assert main(["--output", str(destination)]) == 0
        assert json.loads(destination.read_text())["total_findings"] == 0

    def test_main_prints_the_artefact_without_an_output_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``--output`` the baseline goes to standard output.

        Parameters
        ----------
        monkeypatch, tmp_path, capsys
            Fixtures.

        """
        self._install(monkeypatch, tmp_path, [])

        assert main([]) == 0

        assert json.loads(capsys.readouterr().out)["total_findings"] == 0


class TestTheRealScan:
    """One end-to-end run, asserting consistency rather than a fixed number."""

    def test_the_repository_scan_is_internally_consistent(self) -> None:
        """The count moves as docstrings are written; the invariants do not."""
        baseline = measure_documentation_debt(Path.cwd())

        assert baseline.total_findings == sum(baseline.by_scope.values())
        assert baseline.total_findings == sum(baseline.by_rule.values())
        assert baseline.total_files <= baseline.total_findings
        assert baseline.ruff_version.startswith("ruff ")
        assert set(baseline.by_scope) <= {*DOCUMENTATION_DEBT_SCOPES, "<outside>"}
        assert all(rule.startswith("D") for rule in baseline.by_rule)
