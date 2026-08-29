# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — cloud-native deployment quality-gate tests
"""Lock the cloud-native deployment gate into preflight and CI."""

from __future__ import annotations

from pathlib import Path

from tools import cloud_native_deployment_product_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings for the cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    mypy = gates["mypy-strict-cloud-native-deployment-quality"]
    ruff = gates["ruff D cloud-native-deployment quality ratchet"]
    assert mypy[5:] == quality_gates.CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET
    assert (
        ruff[-len(quality_gates.CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET) :]
        == quality_gates.CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and a 100 percent source-only report."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["cloud-native-deployment focused coverage"]
    report = gates["cloud-native-deployment exact coverage threshold"]
    assert f"--data-file={quality_gates.CLOUD_NATIVE_DEPLOYMENT_COVERAGE_DATA_FILE}" in run
    assert "--branch" in run
    assert run[-len(quality_gates.CLOUD_NATIVE_DEPLOYMENT_COVERAGE_COHORT) :] == (
        quality_gates.CLOUD_NATIVE_DEPLOYMENT_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    include = next(argument for argument in report if argument.startswith("--include="))
    assert "cloud_native_deployment_product.py" in include
    assert "deployment/cloud_native.py" in include


def test_preflight_uses_the_helper_defined_gates() -> None:
    """Keep helper-defined static and coverage commands verbatim in preflight."""
    static = dict(preflight.STATIC_GATES)
    coverage = dict(preflight.CLOUD_NATIVE_DEPLOYMENT_COVERAGE_GATES)
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert static[name] == command
    assert coverage == dict(quality_gates.build_coverage_gates(preflight._PY))


def test_ci_runs_and_aggregates_the_product_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  cloud-native-deployment-quality:")
    end = workflow.index("\n\n  control-stack-compose-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.CLOUD_NATIVE_DEPLOYMENT_QUALITY_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert "cloud_native_deployment_product.py" in block
    assert "deployment/cloud_native.py" in block
    assert "tests/test_cloud_native.py" in block
    assert "tests/test_cloud_native_branches.py" in block
    assert "cloud-native-deployment-quality" in workflow[workflow.index("  ci-gate:") :]
