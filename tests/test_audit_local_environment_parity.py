# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — local environment parity report tests
"""Tests for the report that measures workstation drift from the CI pins.

These cover the report's logic, not the machine it runs on. A live-parity
assertion belongs to the instrument rather than the suite: a CI lint job has no
pnpm and no native engine by design, so a test asserting that this environment
matches every axis would fail there for a reason that is not a defect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools import audit_local_environment_parity as report


def _scratch(tmp_path: Path, *, lock: str = "", toolchain: str = "", workflow: str = "") -> Path:
    """Build a scratch repository carrying only the pieces a case needs."""
    if lock:
        (tmp_path / report.BASE_LOCK).write_text(lock, encoding="utf-8")
    if toolchain:
        (tmp_path / "rust-toolchain.toml").write_text(toolchain, encoding="utf-8")
    if workflow:
        directory = tmp_path / ".github" / "workflows"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "a.yml").write_text(workflow, encoding="utf-8")
    return tmp_path


def test_interpreter_pin_is_read_from_the_most_common_declaration(tmp_path: Path) -> None:
    """Take the interpreter version the workflows pin most often."""
    repo = _scratch(
        tmp_path,
        workflow="""
jobs:
  one:
    steps:
      - uses: actions/setup-python@sha
        with:
          python-version: "3.12"
  two:
    steps:
      - uses: actions/setup-python@sha
        with:
          python-version: "3.12"
  three:
    steps:
      - uses: actions/setup-python@sha
        with:
          python-version: "3.13"
""",
    )

    interpreter, _ = report.workflow_pins(repo)

    assert interpreter == "3.12"


def test_a_templated_interpreter_pin_is_not_taken_as_a_version(tmp_path: Path) -> None:
    """Ignore a matrix expression rather than comparing against its text."""
    repo = _scratch(
        tmp_path,
        workflow="""
jobs:
  matrixed:
    steps:
      - uses: actions/setup-python@sha
        with:
          python-version: ${{ matrix.python-version }}
""",
    )

    interpreter, _ = report.workflow_pins(repo)

    assert interpreter is None


def test_a_different_interpreter_is_reported(tmp_path: Path) -> None:
    """Report a running interpreter that is not the one CI pins."""
    divergences = report.interpreter_divergences("3.9")

    assert [str(item) for item in divergences] == [
        f"python: CI 3.9, local {sys.version_info.major}.{sys.version_info.minor}"
    ]


def test_the_running_interpreter_is_accepted(tmp_path: Path) -> None:
    """Accept the interpreter this process is running under."""
    pinned = f"{sys.version_info.major}.{sys.version_info.minor}"

    assert report.interpreter_divergences(pinned) == []


def test_a_pinned_version_we_do_not_have_is_reported(tmp_path: Path) -> None:
    """Report a distribution installed at a version other than the pin."""
    repo = _scratch(tmp_path, lock="pytest==0.0.1 \\\n    --hash=sha256:0\n")

    divergences = report.distribution_divergences(repo)

    assert any(item.axis == "package pytest" and item.observed != "0.0.1" for item in divergences)


def test_an_absent_distribution_is_reported(tmp_path: Path) -> None:
    """Report a pinned distribution that is not installed at all."""
    repo = _scratch(tmp_path, lock="not-a-real-distribution==9.9.9 \\\n    --hash=sha256:0\n")

    divergences = report.distribution_divergences(repo)

    assert [item.observed for item in divergences] == ["absent"]


def test_extras_in_the_lock_are_compared_by_distribution(tmp_path: Path) -> None:
    """Compare ``name[extra]==version`` against the distribution it names."""
    repo = _scratch(tmp_path, lock="coverage[toml]==0.0.2 \\\n    --hash=sha256:0\n")

    divergences = report.distribution_divergences(repo)

    assert [item.axis for item in divergences] == ["package coverage"]


def test_a_missing_lock_is_reported_rather_than_passing(tmp_path: Path) -> None:
    """Report an absent base lock instead of finding nothing to compare."""
    divergences = report.distribution_divergences(tmp_path)

    assert [item.observed for item in divergences] == ["missing"]


def test_a_toolchain_component_that_is_not_installed_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a declared Rust component the toolchain does not carry.

    Both rustup lookups are stubbed so the case measures the comparison rather
    than whether this machine happens to have a Rust toolchain.
    """
    repo = _scratch(
        tmp_path,
        toolchain=(
            '[toolchain]\nchannel = "stable"\n'
            'components = ["rustfmt", "no-such-component"]\n'
            'targets = ["wasm32-unknown-unknown"]\n'
        ),
    )
    monkeypatch.setattr(report, "_command_version", lambda *_args: "rustc 1.98.1")
    monkeypatch.setattr(
        report,
        "_rustup_listing",
        lambda kind: "rustfmt\n" if kind == "component" else "wasm32-unknown-unknown\n",
    )

    divergences = report.toolchain_divergences(repo)

    assert [item.axis for item in divergences] == ["rust component no-such-component"]


def test_an_absent_rust_toolchain_is_reported_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report the missing compiler rather than enumerating what it would carry."""
    repo = _scratch(
        tmp_path,
        toolchain='[toolchain]\nchannel = "stable"\ncomponents = ["rustfmt", "clippy"]\n',
    )
    monkeypatch.setattr(report, "_command_version", lambda *_args: None)

    divergences = report.toolchain_divergences(repo)

    assert [item.observed for item in divergences] == ["not on PATH"]


def test_a_missing_toolchain_declaration_is_reported(tmp_path: Path) -> None:
    """Report the absence of the toolchain contract rather than skipping it."""
    divergences = report.toolchain_divergences(tmp_path)

    assert [item.observed for item in divergences] == ["missing"]
