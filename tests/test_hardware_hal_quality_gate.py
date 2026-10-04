# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL quality-gate tests
"""Lock the hardware HAL gate into preflight and CI."""

import ast
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import gettempdir

import pytest
import yaml
from coverage.exceptions import DataError

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
    assert f"--data-file={Path(gettempdir()) / 'scpn-qc-hardware-hal-quality.coverage'}" in run
    assert f"--source={quality_gates.HARDWARE_HAL_COVERAGE_MODULES}" in run
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


def test_native_owner_enrollment_matches_each_existing_ci_command() -> None:
    """Every new owner has typing/docs/runtime scope, and CI executes the original gate vectors."""
    sources = [
        "_count_integrity",
        "provider_semantics",
        "provider_measurement",
        "provider_modalities",
        "hal_qiskit",
        "hal_braket",
        "hal_iqm",
        "hal_quandela",
        "hal_dwave",
        "hal_pasqal",
        "hal_quera_bloqade",
    ]
    for name in sources:
        source = f"src/scpn_quantum_control/hardware/{name}.py"
        assert source in quality_gates.HARDWARE_HAL_TYPING_RATCHET
        assert source in quality_gates.HARDWARE_HAL_DOCSTRING_RATCHET
        assert source in quality_gates.HARDWARE_HAL_COVERAGE_SOURCES
        assert (
            f"scpn_quantum_control.hardware.{name}"
            in quality_gates.HARDWARE_HAL_COVERAGE_MODULES.split(",")
        )
    for cohort in (
        quality_gates.HARDWARE_HAL_COVERAGE_COHORT,
        quality_gates.HARDWARE_HAL_TYPING_RATCHET,
        quality_gates.HARDWARE_HAL_DOCSTRING_RATCHET,
    ):
        assert len(cohort) == len(set(cohort))
    workflow = yaml.safe_load(Path(".github/workflows/ci-control-provider.yml").read_text())
    job = workflow["jobs"]["hardware-hal-quality"]
    assert "TMPDIR" not in job["env"]
    coverage_step_names = {
        "Run hardware HAL focused coverage",
        "Verify hardware HAL coverage owner enrollment",
        "Enforce hardware HAL exact coverage",
    }
    for step in job["steps"]:
        if step.get("name") in coverage_step_names:
            assert step["env"]["TMPDIR"] == "${{ runner.temp }}"
    steps = {step.get("name"): step.get("run", "") for step in job["steps"]}
    titles = {
        "mypy-strict-hardware-hal-quality": "Type-check hardware HAL quality cohort",
        "ruff D hardware-hal quality ratchet": "Ruff NumPy docstrings for hardware HAL quality cohort",
        "hardware-hal focused coverage": "Run hardware HAL focused coverage",
        "hardware-hal complete coverage enrollment": "Verify hardware HAL coverage owner enrollment",
        "hardware-hal exact coverage threshold": "Enforce hardware HAL exact coverage",
    }
    for name, command in quality_gates.build_static_quality_gates(
        "python"
    ) + quality_gates.build_coverage_gates("python"):
        actual = [
            arg.replace("${TMPDIR}", gettempdir()) for arg in shlex.split(steps[titles[name]])
        ]
        assert actual == command


@pytest.mark.parametrize(
    "gate,marker",
    [
        ("mypy-strict-hardware-hal-quality", "return-value"),
        ("ruff D hardware-hal quality ratchet", "D103"),
    ],
)
def test_native_static_gate_accepts_current_owners_and_propagates_known_invalid_file(
    tmp_path: Path,
    gate: str,
    marker: str,
) -> None:
    """Actual native tools accept all registered owners and fail a deliberate type or documentation defect."""
    command = dict(quality_gates.build_static_quality_gates(sys.executable))[gate]
    valid = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    original = Path("tools/hardware_hal_quality_gates.py")
    content = original.read_text()
    candidate = tmp_path / original.name
    if marker == "return-value":
        content += '\ndef candidate_type_refusal() -> int:\n    """Require the native checker to reject this deliberate return mismatch."""\n    return "invalid type"\n'
    else:
        function = next(
            node
            for node in ast.parse(content).body
            if isinstance(node, ast.FunctionDef) and node.name == "build_static_quality_gates"
        )
        doc = function.body[0]
        assert isinstance(doc, ast.Expr) and doc.end_lineno is not None
        lines = content.splitlines(keepends=True)
        del lines[doc.lineno - 1 : doc.end_lineno]
        content = "".join(lines)
    candidate.write_text(content)
    invalid = list(command)
    invalid[invalid.index(str(original))] = str(candidate)
    with pytest.raises(subprocess.CalledProcessError) as refused:
        subprocess.run(invalid, capture_output=True, text=True, timeout=90, check=True)
    assert refused.value.returncode != 0 and refused.value.cmd == invalid
    assert marker in (refused.value.stdout or "") + (refused.value.stderr or "")
    assert original.read_text() != candidate.read_text()


@pytest.mark.parametrize("omit_owner", [False, True])
def test_native_coverage_enrollment_accepts_real_measurements_and_refuses_an_omitted_owner(
    tmp_path: Path,
    omit_owner: bool,
) -> None:
    """Actual coverage databases must contain every declared file before percentage acceptance."""
    commands = dict(quality_gates.build_coverage_gates(sys.executable))
    data_file = tmp_path / "native_enrollment.coverage"
    script = tmp_path / "measure_declared_owners.py"
    modules = quality_gates.HARDWARE_HAL_COVERAGE_MODULES.split(",")
    if omit_owner:
        modules.remove("tools.hardware_hal_quality_gates")
    script.write_text(
        "import importlib\n"
        + "\n".join(f"importlib.import_module({name!r})" for name in modules)
        + "\n"
    )
    run = commands["hardware-hal focused coverage"]
    run = run[: run.index("pytest") - 1] + [str(script)]
    run = [f"--data-file={data_file}" if arg.startswith("--data-file=") else arg for arg in run]
    measured = subprocess.run(run, capture_output=True, text=True, timeout=90, check=False)
    assert measured.returncode == 0, measured.stdout + measured.stderr
    validate = [
        str(data_file) if arg == quality_gates.HARDWARE_HAL_COVERAGE_DATA_FILE else arg
        for arg in commands["hardware-hal complete coverage enrollment"]
    ]
    result = subprocess.run(validate, capture_output=True, text=True, timeout=90, check=False)
    if omit_owner:
        assert result.returncode != 0 and "tools/hardware_hal_quality_gates.py" in result.stderr
        with pytest.raises(ValueError, match="did not measure declared owners"):
            quality_gates.validate_coverage_sources(str(data_file))
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        quality_gates.validate_coverage_sources(str(data_file))
    # Import coverage proves enrollment only; actual domain execution still has its separate 100% gate.
    report = [
        f"--data-file={data_file}" if arg.startswith("--data-file=") else arg
        for arg in commands["hardware-hal exact coverage threshold"]
    ]
    partial = subprocess.run(report, capture_output=True, text=True, timeout=90, check=False)
    assert partial.returncode != 0 and "Coverage failure" in partial.stdout


def test_native_coverage_enrollment_refuses_missing_or_unreadable_databases(
    tmp_path: Path,
) -> None:
    """No database or invalid native storage can become accepted measurement evidence."""
    absent = tmp_path / "absent.coverage"
    with pytest.raises(ValueError, match="did not measure declared owners"):
        quality_gates.validate_coverage_sources(str(absent))
    corrupt = tmp_path / "corrupt.coverage"
    corrupt.write_text("invalid coverage database")
    with pytest.raises(DataError):
        quality_gates.validate_coverage_sources(str(corrupt))
