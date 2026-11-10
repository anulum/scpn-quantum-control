# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL quality-gate tests
"""Lock the hardware HAL gate into preflight and CI."""

import ast
import re
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
        "operator_policy",
        "operator_policy_contracts",
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


def test_provider_sdk_locks_have_complete_blocking_advisory_audits() -> None:
    """Every SDK lock keeps a blocking audit with only named, expiring vendor dispositions."""
    workflow = yaml.safe_load(Path(".github/workflows/ci-control-provider.yml").read_text())
    job = workflow["jobs"]["provider-sdk-lock-audit"]
    lanes = job["strategy"]["matrix"]["include"]
    cohorts = [lane["cohort"] for lane in lanes]
    upstream_gated = {"qbraid": "3.13", "iqm": "3.13", "quandela": "3.13", "quera": "3.13"}
    expected = {
        name: upstream_gated.get(name, "3.15")
        for name in (
            "ibm",
            "braket",
            "azure",
            "qbraid",
            "strangeworks",
            "cirq",
            "dwave",
            "iqm",
            "oqc",
            "pasqal",
            "quandela",
            "quantinuum",
            "quera",
            "pennylane",
        )
    }
    expected["rigetti"] = "3.12"
    assert len(cohorts) == len(expected)
    assert {lane["cohort"]: lane["python"] for lane in lanes} == expected
    assert all(lane["lock"] == "py" + lane["python"].replace(".", "") for lane in lanes)
    assert all(
        lane["minimal_lock"] == f"requirements-ci-minimal-install-{lane['lock']}-linux.txt"
        for lane in lanes
    )
    for suffix in ("in", "txt"):
        assert {
            f"requirements-ci-sdk-{lane['cohort']}-{lane['lock']}-linux.{suffix}" for lane in lanes
        } == {path.name for path in Path(".").glob(f"requirements-ci-sdk-*-linux.{suffix}")}
    minimal_inputs = {
        lock: Path(f"requirements-ci-minimal-install-{lock}-linux.in").read_text(encoding="utf-8")
        for lock in {lane["lock"] for lane in lanes}
    }
    pins = {
        lock: [line for line in text.splitlines() if line and not line.startswith("#")]
        for lock, text in minimal_inputs.items()
    }
    assert len({tuple(lines) for lines in pins.values()}) == 1
    for lock in pins:
        lock_text = Path(f"requirements-ci-minimal-install-{lock}-linux.txt").read_text(
            encoding="utf-8"
        )
        assert f"--python-version 3.{lock[3:]} " in lock_text
    assert not job.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in job["steps"])
    steps = {step.get("name"): step.get("run", "") for step in job["steps"]}
    assert steps["Install the pinned audit tool"] == (
        "python -m pip install --require-hashes -r requirements-ci-${{ matrix.lock }}-linux.txt"
    )
    audit = steps["Audit the complete provider SDK lock"]
    command = audit[audit.index("pip-audit --strict") :].replace("\\\n", " ")
    assert " ".join(command.split()) == (
        'pip-audit --strict --no-deps --disable-pip --desc on "${advisory_args[@]}" '
        "-r requirements-ci-sdk-${{ matrix.cohort }}-${{ matrix.lock }}-linux.txt"
    )
    assert re.findall(r"advisory_args\+=\(--ignore-vuln (CVE-[0-9-]+)\)", audit) == [
        "CVE-2025-69872",
        "CVE-2026-85394",
        "CVE-2024-23342",
    ]
    assert audit.count("Review-by 2026-11-24.") == 3
    assert 'if ! [[ "$(date -u +%F)" < "2026-11-24" ]]; then' in audit
    assert "exit 1" in audit
    assert "|| true" not in audit
    assert "--ignore-vuln" not in audit[audit.index("esac") :]
    native = workflow["jobs"]["provider-sdk-conformance"]
    assert native["strategy"]["matrix"]["include"] == lanes
    assert native["env"]["SCPN_PROVIDER_SDK_COHORT"] == "${{ matrix.cohort }}"
    native_steps = {step.get("name"): step.get("run", "") for step in native["steps"]}
    assert " ".join(native_steps["Install the locked common codec test environment"].split()) == (
        "python -m pip install --require-hashes -r ${{ matrix.minimal_lock }}"
    )
    assert " ".join(native_steps["Install the complete isolated SDK cohort"].split()) == (
        "python -m pip install --require-hashes "
        "-r requirements-ci-sdk-${{ matrix.cohort }}-${{ matrix.lock }}-linux.txt"
    )
    assert native_steps["Check installed dependency compatibility"] == "python -m pip check"
    runtime = next(
        step
        for step in native["steps"]
        if step.get("name") == "Execute the required native SDK codec and public workload path"
    )
    assert runtime["shell"] == "bash"
    assert " ".join(runtime["run"].split()) == (
        "python -m pytest -q "
        "tests/test_provider_sdk_conformance.py::test_native_sdk_cohort_roundtrip "
        '| tee "${RUNNER_TEMP}/provider-sdk-conformance.txt"'
    )
    assert not native.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in native["steps"])
    minimal = workflow["jobs"]["provider-sdk-minimal-install"]
    assert minimal["env"]["SCPN_PROVIDER_SDK_COHORT"] == "minimal"
    assert minimal["env"]["PYTHONPATH"] == ""
    minimal_steps = {step.get("name"): step.get("run", "") for step in minimal["steps"]}
    wheel_build = minimal_steps["Build and install the actual base wheels without extras"]
    assert "python -m build --wheel --no-isolation" in wheel_build
    assert "python -m pip install --no-deps --require-hashes" in wheel_build
    assert "-e " not in wheel_build
    assert minimal_steps["Check installed base dependency compatibility"] == "python -m pip check"
    assert (
        "test_native_sdk_cohort_roundtrip"
        in minimal_steps["Execute installed base HAL and actual optional SDK refusal"]
    )
    assert not minimal.get("continue-on-error", False)


@pytest.mark.parametrize("cohort", ["dwave", "strangeworks"])
def test_provider_advisory_disposition_expires_before_audit_execution(cohort: str) -> None:
    """Refuse an expired vendor exception before invoking the actual auditor.

    Parameters
    ----------
    cohort
        Isolated native SDK codec whose named disposition must expire.

    """
    workflow = yaml.safe_load(Path(".github/workflows/ci-control-provider.yml").read_text())
    job = workflow["jobs"]["provider-sdk-lock-audit"]
    script = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Audit the complete provider SDK lock"
    )
    script = script.replace("${{ matrix.cohort }}", cohort).replace("${{ matrix.lock }}", "py315")
    script = script.replace("$(date -u +%F)", "2026-11-24")
    syntax = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, text=True, timeout=10, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    expired = subprocess.run(
        ["bash", "-e", "-o", "pipefail"],
        input=script,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert expired.returncode == 1
    assert "refusing an expired exception" in expired.stdout
    assert expired.stderr == ""


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
