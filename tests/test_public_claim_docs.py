# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Public claim-doc consistency guards
"""Guard hand-written public quality claims against enforced reality.

``VALIDATION.md``, ``CAPABILITIES_AND_USECASES.md``, and ``RESULTS_SUMMARY.md``
carried quality numbers that had drifted years of releases behind the enforced
gates (679 tests / Python 3.9 / Qiskit 1.0 / a 95 or 100 percent coverage gate
against the real 90). These guards pin every such claim to its enforcing
source — ``pyproject.toml``, the CI workflow, and the generated capability
snapshot — so the docs fail closed instead of rotting silently.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VALIDATION = _REPO_ROOT / "VALIDATION.md"
_CAPABILITIES = _REPO_ROOT / "CAPABILITIES_AND_USECASES.md"
_RESULTS = _REPO_ROOT / "RESULTS_SUMMARY.md"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_SNAPSHOT = _REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md"

# Claim tokens that may only reappear if the enforced gates regress with them.
_FORBIDDEN_STALE_TOKENS = (
    "Python 3.9",
    "Qiskit 1.0",
    "100% line coverage",
    "cov-fail-under=95",
    "98% test coverage",
    "2,813 tests across 155 modules",
)


def _ci_coverage_gate() -> int:
    """Return the ``--cov-fail-under`` percentage enforced by the CI workflow.

    Returns
    -------
    int
        The aggregate coverage gate, parsed from ``.github/workflows/ci.yml``.
    """
    match = re.search(r"--cov-fail-under=(\d+)", _CI_WORKFLOW.read_text(encoding="utf-8"))
    assert match is not None, "ci.yml no longer declares --cov-fail-under"
    return int(match.group(1))


def _python_floor() -> str:
    """Return the minimum Python version declared in ``pyproject.toml``.

    Returns
    -------
    str
        The ``X.Y`` floor from ``requires-python = ">=X.Y"``.
    """
    match = re.search(
        r'requires-python\s*=\s*">=(\d+\.\d+)"', _PYPROJECT.read_text(encoding="utf-8")
    )
    assert match is not None, "pyproject.toml no longer declares requires-python"
    return match.group(1)


def _snapshot_reports_test_files() -> bool:
    """Return whether the generated capability snapshot lists test files.

    Returns
    -------
    bool
        ``True`` when the ``Python test files`` row exists in
        ``docs/_generated/capability_snapshot.md``.
    """
    return (
        re.search(r"\|\s*Python test files\s*\|\s*\d+\s*\|", _SNAPSHOT.read_text(encoding="utf-8"))
        is not None
    )


def test_claim_docs_carry_no_stale_quality_tokens() -> None:
    """None of the public claim docs repeats a known-rotten quality claim."""
    offenders: list[str] = []
    for path in (_VALIDATION, _CAPABILITIES, _RESULTS):
        text = path.read_text(encoding="utf-8")
        offenders.extend(
            f"{path.name}: '{token}'" for token in _FORBIDDEN_STALE_TOKENS if token in text
        )
    assert offenders == [], f"stale quality claims resurfaced: {offenders}"


def test_validation_quotes_the_enforced_coverage_gate() -> None:
    """VALIDATION.md states exactly the coverage gate that CI enforces."""
    text = _VALIDATION.read_text(encoding="utf-8")
    assert f"--cov-fail-under={_ci_coverage_gate()}" in text, (
        f"VALIDATION.md does not quote the CI coverage gate (enforced: {_ci_coverage_gate()})"
    )


def test_validation_quotes_the_declared_python_floor() -> None:
    """VALIDATION.md names the same Python floor that packaging declares."""
    assert f"Python {_python_floor()}" in _VALIDATION.read_text(encoding="utf-8"), (
        f"VALIDATION.md does not mention the packaging floor Python {_python_floor()}"
    )


def test_validation_embeds_no_volatile_test_file_count() -> None:
    """VALIDATION.md defers volatile counts to the generated inventory.

    Hand-written counts rot within hours here (829 drifted to 831 while this
    guard was being written), so the volatile number may live only in the
    generated snapshot, which VALIDATION.md must reference instead.
    """
    text = _VALIDATION.read_text(encoding="utf-8")
    assert re.search(r"\d+ test files", text) is None, (
        "VALIDATION.md hardcodes a test-file count; defer to the generated snapshot"
    )
    assert "docs/_generated/capability_snapshot.md" in text, (
        "VALIDATION.md must reference the generated capability snapshot"
    )
    assert _snapshot_reports_test_files(), "the generated snapshot lost its Python-test-files row"


def test_readme_external_speedups_carry_provenance_qualifier() -> None:
    """README external speedup ratios stay bound to their provenance caveat.

    The 1,665x / 44x pulse-shaping figures are v0.9.5-era workstation
    measurements documented on an external page; they may appear only inside
    a sentence that says so, until a committed benchmark artefact reproduces
    them.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for ratio in ("1,665×", "44×"):
        for match in re.finditer(re.escape(ratio), readme):
            window = readme[max(0, match.start() - 300) : match.end() + 300]
            assert "v0.9.5-era workstation measurements" in window, (
                f"README quotes {ratio} without its provenance qualifier"
            )


def test_results_summary_is_marked_as_historical_record() -> None:
    """RESULTS_SUMMARY.md frames its campaign-era numbers as historical."""
    text = _RESULTS.read_text(encoding="utf-8")
    assert "Historical campaign record" in text, (
        "RESULTS_SUMMARY.md lost its historical-record banner"
    )
    assert re.search(r"^\*\*Version:\*\* \d", text, flags=re.MULTILINE) is None, (
        "RESULTS_SUMMARY.md reintroduced a bare current-version header"
    )
