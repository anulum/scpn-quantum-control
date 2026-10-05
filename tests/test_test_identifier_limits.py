# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — length limit for collected test identifiers tests
"""Prove the identifier limit with a nested session under the shared configuration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import _test_identifier_limits as limits
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PROBE = "tests/test_test_identifier_limits.py::test_probe_case"
PROBE_VARIABLE = "SCPN_QC_IDENTIFIER_PROBE_CHARACTERS"
_PROBE_VALUE = "x" * int(os.environ.get(PROBE_VARIABLE, "8"))


def _run_probe(characters: int) -> subprocess.CompletedProcess[str]:
    """Run the probe case verbosely in a nested session with a value of ``characters``."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", PROBE],
        cwd=REPOSITORY,
        env={**os.environ, PROBE_VARIABLE: str(characters)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("payload", [_PROBE_VALUE])
def test_probe_case(payload: str) -> None:
    """Carry a parameter whose length the nested sessions choose."""
    assert set(payload) == {"x"}


def test_identifiers_are_measured_against_the_limit() -> None:
    """Only identifiers beyond the limit are returned, shortened, with their length."""
    short = "tests/test_a.py::test_b[case]"
    exact = "y" * limits.MAX_TEST_IDENTIFIER_CHARACTERS
    long = "z" * (limits.MAX_TEST_IDENTIFIER_CHARACTERS + 1)

    assert limits.overlong_identifiers([short, exact]) == ()
    assert limits.overlong_identifiers([short, long, exact]) == (
        ("z" * 120, limits.MAX_TEST_IDENTIFIER_CHARACTERS + 1),
    )
    assert limits.overlong_identifiers([short], limit=5) == ((short, len(short)),)


def test_shared_configuration_admits_a_short_identifier() -> None:
    """A nested session under the shared configuration runs the short probe."""
    completed = _run_probe(8)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"{PROBE}[xxxxxxxx] PASSED" in completed.stdout


def test_shared_configuration_refuses_an_overlong_identifier() -> None:
    """A nested session runs no test, names the overlong case and prints no long line."""
    limit = limits.MAX_TEST_IDENTIFIER_CHARACTERS
    refused = _run_probe(limit + 1)
    report = refused.stdout + refused.stderr

    assert refused.returncode == pytest.ExitCode.USAGE_ERROR, report
    assert f"1 test identifier(s) exceed {limit} characters" in report
    assert "pytest.param(..., id=...)" in report
    assert f"{PROBE}[" in report
    assert "no tests ran" in report
    assert max(len(line) for line in report.splitlines()) < 400


def test_the_running_session_has_no_overlong_identifier(request: pytest.FixtureRequest) -> None:
    """Every test collected for this session is within the limit."""
    assert limits.overlong_identifiers(item.nodeid for item in request.session.items) == ()
