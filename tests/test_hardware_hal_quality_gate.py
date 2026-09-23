# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL quality-gate tests
"""Lock the hardware HAL gate into preflight and CI."""

from pathlib import Path

import pytest
import yaml

from tools import hardware_hal_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete owner NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-hardware-hal-quality"][5:] == quality_gates.HARDWARE_HAL_TYPING_RATCHET
    )
    ruff = gates["ruff D hardware-hal quality ratchet"]
    assert (
        ruff[-len(quality_gates.HARDWARE_HAL_DOCSTRING_RATCHET) :]
        == quality_gates.HARDWARE_HAL_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["hardware-hal focused coverage"]
    report = gates["hardware-hal exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.HARDWARE_HAL_COVERAGE_COHORT) :]
        == quality_gates.HARDWARE_HAL_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.HARDWARE_HAL_COVERAGE_INCLUDE}" in report
    assert quality_gates.HARDWARE_AGGREGATOR_SOURCE in quality_gates.HARDWARE_HAL_TYPING_RATCHET
    assert all(
        path in quality_gates.HARDWARE_HAL_COVERAGE_COHORT
        for path in quality_gates.HARDWARE_AGGREGATOR_TESTS
    )
    assert (
        quality_gates.PROVIDER_CAPABILITY_CLOUD_ADAPTERS_SOURCE
        in quality_gates.HARDWARE_HAL_TYPING_RATCHET
    )
    assert (
        quality_gates.PROVIDER_CAPABILITY_CLOUD_ADAPTERS_TEST
        in quality_gates.HARDWARE_HAL_COVERAGE_COHORT
    )
    assert (
        quality_gates.PROVIDER_CAPABILITY_GATE_ADAPTERS_SOURCE
        in quality_gates.HARDWARE_HAL_TYPING_RATCHET
    )
    assert (
        quality_gates.PROVIDER_CAPABILITY_GATE_ADAPTERS_TEST
        in quality_gates.HARDWARE_HAL_COVERAGE_COHORT
    )
    assert (
        quality_gates.PROVIDER_SUBMISSION_GATE_SOURCE in quality_gates.HARDWARE_HAL_TYPING_RATCHET
    )
    assert (
        quality_gates.PROVIDER_SUBMISSION_GATE_TEST in quality_gates.HARDWARE_HAL_COVERAGE_COHORT
    )
    assert "*/hardware/provider_submission_gate.py" in quality_gates.HARDWARE_HAL_COVERAGE_INCLUDE
    assert (
        quality_gates.PROVIDER_CAPABILITY_SPECIALIZED_ADAPTERS_SOURCE
        in quality_gates.HARDWARE_HAL_TYPING_RATCHET
    )
    assert (
        quality_gates.PROVIDER_CAPABILITY_SPECIALIZED_ADAPTERS_TEST
        in quality_gates.HARDWARE_HAL_COVERAGE_COHORT
    )


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.HARDWARE_HAL_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(HARDWARE_HAL_COVERAGE_GATES)" in Path("tools/preflight.py").read_text()


def test_ci_runs_and_aggregates_hardware_hal_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  hardware-hal-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.HARDWARE_HAL_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.HARDWARE_HAL_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.HARDWARE_HAL_COVERAGE_INCLUDE in block
    assert "hardware-hal-quality" in workflow[workflow.index("  ci-gate:") :]


@pytest.mark.parametrize(
    "owner",
    [
        "tests/test_provider_route_catalogue.py",
        "tests/test_provider_route_configuration.py",
    ],
)
def test_route_inventory_is_in_each_executable_quality_step(owner: str) -> None:
    """Catalogue execution, typing and docs cannot be satisfied by another step."""
    for cohort in (
        quality_gates.HARDWARE_HAL_COVERAGE_COHORT,
        quality_gates.HARDWARE_HAL_TYPING_RATCHET,
        quality_gates.HARDWARE_HAL_DOCSTRING_RATCHET,
    ):
        assert owner in cohort
    workflow = yaml.safe_load(Path(".github/workflows/ci-control-provider.yml").read_text())
    steps = {
        step.get("name"): step.get("run", "")
        for step in workflow["jobs"]["hardware-hal-quality"]["steps"]
    }
    for name in (
        "Type-check hardware HAL quality cohort",
        "Ruff NumPy docstrings for hardware HAL quality cohort",
        "Run hardware HAL focused coverage",
    ):
        assert owner in steps[name].split()
