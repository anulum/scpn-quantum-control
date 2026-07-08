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

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VALIDATION = _REPO_ROOT / "VALIDATION.md"
_CAPABILITIES = _REPO_ROOT / "CAPABILITIES_AND_USECASES.md"
_RESULTS = _REPO_ROOT / "RESULTS_SUMMARY.md"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_SNAPSHOT = _REPO_ROOT / "docs" / "_generated" / "capability_snapshot.md"
_CONTROL_SCOPE = _REPO_ROOT / "docs" / "control_scope.md"
_ROADMAP = _REPO_ROOT / "ROADMAP.md"
_DOCUMENTATION_ALLOWLIST = _REPO_ROOT / "tools" / "documentation_surface_allowlist.json"
_CONTROL_BOUNDARY_LINK_SURFACES = (
    _REPO_ROOT / "docs" / "api.md",
    _REPO_ROOT / "docs" / "architecture_map.md",
    _REPO_ROOT / "docs" / "closed_loop_control.md",
    _REPO_ROOT / "docs" / "frc_pulsed_qaoa.md",
    _REPO_ROOT / "docs" / "realtime_feedback.md",
    _REPO_ROOT / "docs" / "tutorials.md",
)
_CONTROL_PUBLIC_SURFACES = (
    _CONTROL_SCOPE,
    *_CONTROL_BOUNDARY_LINK_SURFACES,
    _REPO_ROOT / "src" / "scpn_quantum_control" / "control" / "__init__.py",
)

# Claim tokens that may only reappear if the enforced gates regress with them.
_FORBIDDEN_STALE_TOKENS = (
    "Python 3.9",
    "Qiskit 1.0",
    "100% line coverage",
    "cov-fail-under=95",
    "98% test coverage",
    "2,813 tests across 155 modules",
)
_CONTROL_SCOPE_REQUIRED_PHRASES = (
    "Kuramoto-XY feedback and set-point tracking",
    "FRC pulsed-shot schedule scoring and QAOA sampling",
    "Software-in-the-loop closed-loop response analysis",
    "does not provide generic pulse-shape optimisation",
    "provider-native pulse calibration",
    "hardware drift compensation",
    "lab-instrument control",
)
_CONTROL_OVERCLAIM_TOKENS = (
    "provides generic pulse-control",
    "ships generic pulse-control",
    "automates pulse drift compensation",
    "compensates hardware drift",
    "provider-native pulse calibration engine",
    "controls arbitrary lab instruments",
    "executes closed-loop hardware pulse control",
)
_REMOVED_PAPER0_PATH_TOKENS = (
    "scripts/run_paper0",
    "scripts/compare_paper0",
    "scripts/export_paper0",
    "docs/paper0/",
    "data/paper0_",
    "tests/test_paper0",
    "paper0-knm-preregistered-replay-gate",
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


def test_readme_defers_source_file_count_to_generated_inventory() -> None:
    """The README never hardcodes the tracked source-file count.

    The hand-written 526 drifted 52 files behind the generated inventory before
    it was caught; the count lives only in the generated capability snapshot.
    """
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"\*\*\d+\*\* tracked Python source files", readme) is None, (
        "README.md hardcodes the source-file count; defer to the capability snapshot"
    )


def test_readme_has_single_quickstart_and_value_sections() -> None:
    """The deduplicated landing sections do not regrow their duplicates."""
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quickstarts = re.findall(r"^## Quick Start\s*$", readme, flags=re.MULTILINE)
    value_sections = re.findall(
        r"^## Application and [Cc]ommercial [Vv]alue\s*$", readme, flags=re.MULTILINE
    )
    assert len(quickstarts) == 1, f"README has {len(quickstarts)} Quick Start sections"
    assert len(value_sections) == 1, (
        f"README has {len(value_sections)} application/commercial-value sections"
    )


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


# CEO directive 2026-07-07 (BROADCAST_2026-07-07_no_superlatives_outward): outward
# surfaces carry no self-applied superlatives — they are internal quality targets.
# Tokens here are the unambiguous marketing tier; factual baseline comparisons
# ("not yet category-leading", scorecards) do not use these words.
_BOAST_TOKENS = (
    "world-class",
    "world class",
    "best-in-class",
    "best in class",
    "category of one",
    "cutting-edge",
    "cutting edge",
    "unrivalled",
    "unrivaled",
    "revolutionary",
    "groundbreaking",
)


def _public_markdown_surfaces() -> list[Path]:
    """Every hand-written or generated public Markdown page.

    Root-level ``*.md`` plus ``docs/**/*.md`` outside ``docs/internal`` — the
    same outward boundary the rendered-docs header guard uses.
    """
    docs = [path for path in (_REPO_ROOT / "docs").rglob("*.md") if "internal" not in path.parts]
    return sorted(docs) + sorted(_REPO_ROOT.glob("*.md"))


def test_public_markdown_carries_no_self_applied_superlatives() -> None:
    """No outward Markdown page uses marketing superlatives about this project."""
    offenders: list[str] = []
    for path in _public_markdown_surfaces():
        text = path.read_text(encoding="utf-8").lower()
        for token in _BOAST_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {token!r}")
    assert not offenders, (
        "self-applied superlatives are internal quality targets, not public "
        f"claims (BROADCAST_2026-07-07): {offenders}"
    )


def test_roadmap_does_not_link_removed_paper0_artifacts() -> None:
    """The current roadmap should not point at removed Paper 0 package paths."""
    roadmap = _ROADMAP.read_text(encoding="utf-8")
    offenders = [token for token in _REMOVED_PAPER0_PATH_TOKENS if token in roadmap]
    assert offenders == [], (
        "ROADMAP.md still names removed Paper 0 package artefacts as current "
        f"surfaces: {offenders}"
    )


def test_documentation_allowlist_has_no_removed_paper0_patterns() -> None:
    """Documentation-audit waivers must not reference removed Paper 0 generators."""
    payload = json.loads(_DOCUMENTATION_ALLOWLIST.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    offenders = [
        entry
        for entry in entries
        if "paper0" in str(entry.get("path_pattern", "")).lower()
        or "paper 0" in str(entry.get("reason", "")).lower()
    ]
    assert offenders == [], f"removed Paper 0 allow-list entries remain: {offenders}"


def test_control_scope_boundary_is_public_and_linked() -> None:
    """The control scope boundary stays explicit on public control surfaces."""
    scope_text = _CONTROL_SCOPE.read_text(encoding="utf-8")
    collapsed_scope = re.sub(r"\s+", " ", scope_text)
    missing = [
        phrase for phrase in _CONTROL_SCOPE_REQUIRED_PHRASES if phrase not in collapsed_scope
    ]
    assert not missing, f"control scope boundary lost required phrases: {missing}"
    link_missing = [
        str(path.relative_to(_REPO_ROOT))
        for path in _CONTROL_BOUNDARY_LINK_SURFACES
        if "control_scope.md" not in path.read_text(encoding="utf-8")
        and "Control Scope Boundary" not in path.read_text(encoding="utf-8")
    ]
    assert not link_missing, f"control scope boundary is not linked from: {link_missing}"


def test_control_public_surfaces_do_not_promote_generic_pulse_control() -> None:
    """Public control surfaces cannot drift into generic pulse-control claims."""
    offenders: list[str] = []
    for path in _CONTROL_PUBLIC_SURFACES:
        text = path.read_text(encoding="utf-8").lower()
        offenders.extend(
            f"{path.relative_to(_REPO_ROOT)}: {token!r}"
            for token in _CONTROL_OVERCLAIM_TOKENS
            if token in text
        )
    assert not offenders, f"generic pulse-control overclaims resurfaced: {offenders}"
