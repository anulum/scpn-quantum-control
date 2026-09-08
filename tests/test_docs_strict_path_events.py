# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Strict docs workflow path-event contract tests
"""The strict docs build must trigger on everything it actually consumes.

`docs-strict.yml` installs `oscillatools/` and a pinned
`requirements-ci-*.txt`, then runs `mkdocs build --strict`. Importing any
rendered `scpn_quantum_control` module pulls the package `__init__`, which
imports `oscillatools`, so a change to that package or to the pinned
environment can break this gate. Its trigger filters omitted all three, and the
pull-request filter also omitted the workflow's own path, so an edit to the
gate could not run the gate.

Path-event tests check the repository's configured filters, including the root
Rust toolchain consumed by Rustdoc. Gate tests execute the actual shell with
resolved job results; they do not emulate GitHub event delivery. Cross-language
job bodies are also covered by `tests/test_cross_language_api_docs.py`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(".github/workflows/docs-strict.yml")
"""The workflow whose trigger contract is under test."""


def _workflow() -> dict[Any, Any]:
    """Return the parsed workflow document.

    The mapping is keyed by ``Any`` rather than ``str`` because PyYAML reads the
    unquoted key ``on`` as the boolean ``True``.

    Returns
    -------
    dict
        The whole workflow, including its trigger block.

    """
    loaded: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text())
    return loaded


def _triggers() -> dict[str, Any]:
    """Return the workflow's trigger block.

    PyYAML reads the unquoted key ``on`` as the boolean ``True``, so both spellings
    are accepted rather than assuming which one this file uses.

    Returns
    -------
    dict
        The event-to-configuration mapping.

    """
    document = _workflow()
    block: dict[str, Any] = document[True] if True in document else document["on"]
    return block


def _matches(pattern: str, path: str) -> bool:
    """Return whether a GitHub Actions path filter matches a changed file.

    GitHub's filter glob differs from :mod:`fnmatch` in the way that matters
    here: ``*`` does not cross a directory separator while ``**`` does. A
    root-level pattern such as ``pyproject.toml`` therefore matches only the
    root file, never a nested one, which is the behaviour these tests rely on.

    Parameters
    ----------
    pattern
        One entry from a workflow ``paths`` list.
    path
        Repository-relative path of a changed file.

    Returns
    -------
    bool
        True when the filter selects the file.

    """
    expression = ""
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if pattern.startswith("**", index):
            expression += ".*"
            index += 2
        elif character == "*":
            expression += "[^/]*"
            index += 1
        elif character == "?":
            expression += "[^/]"
            index += 1
        else:
            expression += re.escape(character)
            index += 1
    return re.fullmatch(expression, path) is not None


def _runs(event: str, changed: str) -> bool:
    """Return whether ``event`` triggers the workflow for one changed file.

    Parameters
    ----------
    event
        Either ``"push"`` or ``"pull_request"``.
    changed
        Repository-relative path of a changed file.

    Returns
    -------
    bool
        True when at least one path filter selects the file.

    """
    return any(_matches(pattern, changed) for pattern in _triggers()[event]["paths"])


class TestTheMatcher:
    """The matcher itself, so the contract below rests on something checked."""

    @pytest.mark.parametrize(
        ("pattern", "path", "expected"),
        [
            ("pyproject.toml", "pyproject.toml", True),
            ("pyproject.toml", "oscillatools/pyproject.toml", False),
            ("requirements-ci-*.txt", "requirements-ci-py312-linux.txt", True),
            ("requirements-ci-*.txt", "sub/requirements-ci-py312-linux.txt", False),
            ("requirements-ci-*.txt", "requirements-dev.txt", False),
            ("src/**", "src/scpn_quantum_control/fep/variational_free_energy.py", True),
            ("src/**", "srcx/a.py", False),
            ("docs/**", "docs/api/module_catalog.md", True),
        ],
    )
    def test_glob_semantics(self, pattern: str, path: str, expected: bool) -> None:
        """A single star must not cross a separator; a double star must.

        Parameters
        ----------
        pattern
            Filter under test.
        path
            Candidate changed file.
        expected
            Whether the filter should select it.

        """
        assert _matches(pattern, path) is expected


class TestTheThreeRecordedGaps:
    """The edit shapes a docs build must react to."""

    @pytest.mark.parametrize("event", ["push", "pull_request"])
    def test_rust_toolchain_changes_trigger_reference_validation(self, event: str) -> None:
        """Require the compiler configuration to trigger either event.

        Parameters
        ----------
        event
            Push or pull-request trigger under test.

        """
        assert Path("rust-toolchain.toml").is_file()
        assert _runs(event, "rust-toolchain.toml")

    @pytest.mark.parametrize("event", ["push", "pull_request"])
    def test_a_package_only_edit_triggers(self, event: str) -> None:
        """The job installs oscillatools and the strict build imports it.

        Parameters
        ----------
        event
            Trigger event under test.

        """
        assert _runs(event, "oscillatools/src/oscillatools/accel/__init__.py")

    @pytest.mark.parametrize("event", ["push", "pull_request"])
    def test_a_lock_only_edit_triggers(self, event: str) -> None:
        """The pinned requirements file is the environment this gate runs in.

        Parameters
        ----------
        event
            Trigger event under test.

        """
        assert _runs(event, "requirements-ci-py312-linux.txt")
        assert _runs(event, "pyproject.toml")

    @pytest.mark.parametrize("event", ["push", "pull_request"])
    def test_a_workflow_only_edit_triggers(self, event: str) -> None:
        """An edit to the gate must be able to run the gate.

        The pull-request filter omitted this path, so a pull request that only
        changed this workflow could not validate it.

        Parameters
        ----------
        event
            Trigger event under test.

        """
        assert _runs(event, ".github/workflows/docs-strict.yml")


class TestTriggerContract:
    """Properties that keep the two filters honest as the workflow changes."""

    def test_push_and_pull_request_select_the_same_paths(self) -> None:
        """A divergence is how the workflow-only gap appeared."""
        triggers = _triggers()
        assert triggers["push"]["paths"] == triggers["pull_request"]["paths"]

    @pytest.mark.parametrize("event", ["push", "pull_request"])
    def test_unrelated_edits_do_not_trigger(self, event: str) -> None:
        """Widening the filters must not turn this into an always-on job.

        ``docs.yml`` is the non-strict deploy sibling: editing it must not run
        the strict gate, and conflating the two is the mistake this guards.

        Parameters
        ----------
        event
            Trigger event under test.

        """
        for path in (
            "README.md",
            "tests/test_fep.py",
            "tools/capability_manifest.py",
            ".github/workflows/docs.yml",
        ):
            assert not _runs(event, path)

    @pytest.mark.parametrize(
        "installed",
        ["oscillatools/", "requirements-ci-py312-linux.txt"],
    )
    def test_every_environment_input_the_job_installs_is_a_trigger(self, installed: str) -> None:
        """What the job installs and what triggers it must not diverge again.

        Parameters
        ----------
        installed
            A path the workflow body installs from.

        """
        body = WORKFLOW.read_text()
        assert installed in body
        probe = (
            "oscillatools/src/oscillatools/__init__.py" if installed.endswith("/") else installed
        )
        assert _runs("push", probe)
        assert _runs("pull_request", probe)

    def test_the_aggregate_gate_is_retained(self) -> None:
        """Every reference job contributes to the gate when this workflow runs."""
        document = _workflow()
        gate = document["jobs"]["docs-gate"]
        assert gate["if"] == "always()"
        assert gate["needs"] == ["build-strict", "rust-reference", "typescript-reference"]

    @pytest.mark.parametrize(
        "failed_job", ["build-strict", "rust-reference", "typescript-reference"]
    )
    @pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
    def test_actual_gate_shell_rejects_unsuccessful_reference(
        self, failed_job: str, result: str
    ) -> None:
        """Execute the configured gate without running reference builds.

        Parameters
        ----------
        failed_job
            Reference job whose terminal result varies.
        result
            Success must pass; failure, cancellation and skipping must fail.

        """
        gate = _workflow()["jobs"]["docs-gate"]
        script = gate["steps"][0]["run"]
        for job in gate["needs"]:
            status = result if job == failed_job else "success"
            script = script.replace("${{ needs." + job + ".result }}", status)
        assert "${{" not in script
        completed = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
            capture_output=True,
            check=False,
            timeout=5,
        )
        assert (completed.returncode == 0) is (result == "success")

    def test_manual_dispatch_remains_available(self) -> None:
        """A skipped path set must stay runnable on demand."""
        assert "workflow_dispatch" in _triggers()
