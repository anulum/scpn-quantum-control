# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ML-DSA honesty-seal quality-gate tests
"""Lock the signer and result-pack seal gate into preflight and CI."""

from pathlib import Path

from tools import ml_dsa_seal_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete connected NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert gates["mypy-strict-ml-dsa-seal-quality"][5:] == quality_gates.ML_DSA_SEAL_TYPING_RATCHET
    ruff = gates["ruff D ml-dsa-seal quality ratchet"]
    assert (
        ruff[-len(quality_gates.ML_DSA_SEAL_DOCSTRING_RATCHET) :]
        == quality_gates.ML_DSA_SEAL_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact signer source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["ml-dsa-seal focused coverage"]
    report = gates["ml-dsa-seal exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.ML_DSA_SEAL_COVERAGE_COHORT) :]
        == quality_gates.ML_DSA_SEAL_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/crypto/ml_dsa_seal.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.ML_DSA_SEAL_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(ML_DSA_SEAL_COVERAGE_GATES)" in Path("tools/preflight.py").read_text()


def test_ci_runs_and_aggregates_ml_dsa_seal_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  ml-dsa-seal-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.ML_DSA_SEAL_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.ML_DSA_SEAL_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "crypto/ml_dsa_seal.py" in block
    assert "ml-dsa-seal-quality" in workflow[workflow.index("  ci-gate:") :]
