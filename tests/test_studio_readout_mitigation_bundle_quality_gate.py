# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio readout-mitigation bundle quality-gate tests
"""Lock the Studio readout-mitigation quality gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import studio_readout_mitigation_bundle_quality_gates as quality_gates


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-studio-readout-mitigation-bundle-quality"][5:]
        == quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_TYPING_RATCHET
    )
    ruff = gates["ruff D Studio readout-mitigation bundle quality ratchet"]
    assert (
        ruff[-len(quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_DOCSTRING_RATCHET) :]
        == quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_public_and_exact() -> None:
    """Require real public entrypoint execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["Studio readout-mitigation bundle focused coverage"]
    report = gates["Studio readout-mitigation bundle exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_COHORT) :]
        == quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/studio/readout_mitigation_bundle.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_GATES)" in Path(
        "tools/preflight.py"
    ).read_text(encoding="utf-8")


def test_ci_runs_and_aggregates_studio_readout_mitigation_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  studio-readout-mitigation-bundle-quality:")
    end = workflow.index("\n\n  studio-scorecard-bundle-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.STUDIO_READOUT_MITIGATION_BUNDLE_SOURCE in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "studio-readout-mitigation-bundle-quality" in aggregate
