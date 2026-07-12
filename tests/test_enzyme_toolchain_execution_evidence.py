# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — enzyme toolchain execution evidence tests
# scpn-quantum-control -- real Enzyme/LLVM toolchain AD execution evidence tests
"""Tests for the real Enzyme/LLVM reverse-mode AD execution evidence surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.compiler.mlir_enzyme_execution_runner as runner
from scpn_quantum_control.compiler import (
    EnzymeToolchainADCase,
    EnzymeToolchainADExecutionEvidence,
    run_enzyme_toolchain_execution_evidence,
)
from scpn_quantum_control.compiler.mlir_enzyme_execution_runner import resolve_enzyme_toolchain

_EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "differentiable_phase_qnode"


def _rebuild(payload: dict[str, object]) -> EnzymeToolchainADExecutionEvidence:
    """Rebuild Enzyme toolchain evidence from a JSON-like payload."""
    rows = tuple(cast("list[dict[str, object]]", payload["cases"]))
    cases = tuple(
        EnzymeToolchainADCase(
            case_id=cast("str", row["case_id"]),
            operation_family=cast("str", row["operation_family"]),
            operand_dimension=cast("int", row["operand_dimension"]),
            status=cast("str", row["status"]),
            gradient_error=cast("float | None", row["gradient_error"]),
            runtime_seconds=cast("float | None", row["runtime_seconds"]),
            failure_class=cast("str | None", row["failure_class"]),
            claim_boundary=cast("str", row["claim_boundary"]),
        )
        for row in rows
    )
    return EnzymeToolchainADExecutionEvidence(
        artifact_id=cast("str", payload["artifact_id"]),
        toolchain_available=cast("bool", payload["toolchain_available"]),
        toolchain=cast("dict[str, str]", payload["toolchain"]),
        cases=cases,
        beyond_scalar_executed=cast("bool", payload["beyond_scalar_executed"]),
        executed_operation_families=tuple(
            cast("list[str]", payload["executed_operation_families"])
        ),
        max_gradient_error=cast("float", payload["max_gradient_error"]),
        gradient_parity_tolerance=cast("float", payload["gradient_parity_tolerance"]),
        claim_boundary=cast("str", payload["claim_boundary"]),
    )


def _write_executable(path: Path, body: str = "printf 'fake LLVM 18.1.3\\n'\n") -> None:
    """Create a tiny executable script for resolver tests."""
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o700)


def _patch_which(
    monkeypatch: pytest.MonkeyPatch,
    commands: dict[str, str],
) -> None:
    """Patch the runner's PATH resolver with a typed command map."""

    def which(command: str) -> str | None:
        return commands.get(command)

    monkeypatch.setattr(cast("Any", runner).shutil, "which", which)


def test_runner_is_gated_or_executes_beyond_scalar() -> None:
    """The runner either executes the battery beyond scalar or fails closed as gated."""
    evidence = run_enzyme_toolchain_execution_evidence()
    if evidence.toolchain_available:
        assert evidence.beyond_scalar_executed is True
        assert set(evidence.executed_operation_families) >= {"scalar", "vector", "matrix"}
        assert evidence.max_gradient_error <= evidence.gradient_parity_tolerance
        for case in evidence.cases:
            if case.status == "executed":
                assert case.gradient_error is not None
                assert case.gradient_error <= evidence.gradient_parity_tolerance
    else:
        assert evidence.beyond_scalar_executed is False
        assert evidence.executed_operation_families == ()
        assert evidence.max_gradient_error == 0.0
        assert all(case.status == "hard_gap" for case in evidence.cases)
        assert all(case.failure_class for case in evidence.cases)


def test_runner_matches_toolchain_detection() -> None:
    """The evidence toolchain flag agrees with the toolchain resolver."""
    evidence = run_enzyme_toolchain_execution_evidence()
    assert evidence.toolchain_available is (resolve_enzyme_toolchain() is not None)


def test_runner_evidence_round_trips_through_json() -> None:
    """The captured evidence serialises and rebuilds into an equivalent record."""
    evidence = run_enzyme_toolchain_execution_evidence()
    payload = json.loads(json.dumps(evidence.to_dict(), sort_keys=True))
    rebuilt = _rebuild(payload)
    assert rebuilt.toolchain_available == evidence.toolchain_available
    assert rebuilt.beyond_scalar_executed == evidence.beyond_scalar_executed
    assert len(rebuilt.cases) == len(evidence.cases)


def test_executed_case_requires_finite_metrics() -> None:
    """An executed case must carry finite non-negative metrics and no failure_class."""
    with pytest.raises(ValueError, match="finite non-negative"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="executed",
            gradient_error=None,
            runtime_seconds=1e-3,
            failure_class=None,
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="must not carry a failure_class"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="executed",
            gradient_error=0.0,
            runtime_seconds=1e-3,
            failure_class="should-not-be-here",
            claim_boundary="bounded",
        )


def test_hard_gap_case_requires_reason_and_no_metrics() -> None:
    """A hard-gap case must carry a reason and no execution metrics."""
    with pytest.raises(ValueError, match="failure_class"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="hard_gap",
            gradient_error=None,
            runtime_seconds=None,
            failure_class=None,
            claim_boundary="bounded",
        )
    with pytest.raises(ValueError, match="must not carry execution metrics"):
        EnzymeToolchainADCase(
            case_id="x",
            operation_family="vector",
            operand_dimension=4,
            status="hard_gap",
            gradient_error=0.0,
            runtime_seconds=None,
            failure_class="declined",
            claim_boundary="bounded",
        )


def test_unavailable_toolchain_cannot_record_executed_cases() -> None:
    """An evidence marked toolchain-unavailable cannot hold executed rows."""
    executed = EnzymeToolchainADCase(
        case_id="vec",
        operation_family="vector",
        operand_dimension=4,
        status="executed",
        gradient_error=0.0,
        runtime_seconds=1e-3,
        failure_class=None,
        claim_boundary="bounded",
    )
    with pytest.raises(ValueError, match="unavailable toolchain cannot record executed"):
        EnzymeToolchainADExecutionEvidence(
            artifact_id="probe",
            toolchain_available=False,
            toolchain={"status": "unavailable"},
            cases=(executed,),
            beyond_scalar_executed=True,
            executed_operation_families=("vector",),
            max_gradient_error=0.0,
            gradient_parity_tolerance=1e-9,
            claim_boundary="bounded",
        )


def test_aggregate_rejects_gradient_error_over_tolerance() -> None:
    """Evidence cannot be built when an executed gradient error exceeds tolerance."""
    over = EnzymeToolchainADCase(
        case_id="vec",
        operation_family="vector",
        operand_dimension=4,
        status="executed",
        gradient_error=1e-3,
        runtime_seconds=1e-3,
        failure_class=None,
        claim_boundary="bounded",
    )
    with pytest.raises(ValueError, match="exceeds the declared parity tolerance"):
        EnzymeToolchainADExecutionEvidence(
            artifact_id="probe",
            toolchain_available=True,
            toolchain={"clang": "x"},
            cases=(over,),
            beyond_scalar_executed=True,
            executed_operation_families=("vector",),
            max_gradient_error=1e-3,
            gradient_parity_tolerance=1e-9,
            claim_boundary="bounded",
        )


def test_toolchain_resolver_rejects_relative_plugin_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Relative Enzyme plugin overrides are rejected before subprocess probing."""
    clang = tmp_path / "clang"
    opt = tmp_path / "opt"
    _write_executable(clang)
    _write_executable(opt)
    commands = {"clang": str(clang), "opt": str(opt)}

    monkeypatch.setenv("SCPN_ENZYME_PLUGIN", "relative/LLVMEnzyme-18.so")
    _patch_which(monkeypatch, commands)

    assert resolve_enzyme_toolchain() is None


def test_toolchain_resolver_rejects_non_executable_compiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Compiler command paths must point at executable files."""
    clang = tmp_path / "clang"
    opt = tmp_path / "opt"
    plugin = tmp_path / "LLVMEnzyme-18.so"
    clang.write_text("#!/bin/sh\n", encoding="utf-8")
    opt.write_text("#!/bin/sh\n", encoding="utf-8")
    opt.chmod(0o700)
    plugin.write_text("plugin", encoding="utf-8")
    commands = {"clang": str(clang), "opt": str(opt)}

    monkeypatch.setenv("SCPN_ENZYME_PLUGIN", str(plugin))
    _patch_which(monkeypatch, commands)

    assert resolve_enzyme_toolchain() is None


def test_toolchain_resolver_accepts_absolute_admitted_toolchain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Absolute executable compiler paths and absolute plugins are admitted."""
    clang = tmp_path / "clang"
    opt = tmp_path / "opt"
    plugin = tmp_path / "LLVMEnzyme-18.so"
    _write_executable(clang, "printf 'fake clang 18.1.3\\n'\n")
    _write_executable(opt, "printf 'fake opt 18.1.3\\n'\n")
    plugin.write_text("plugin", encoding="utf-8")
    commands = {"clang": str(clang), "opt": str(opt)}

    monkeypatch.setenv("SCPN_ENZYME_PLUGIN", str(plugin))
    _patch_which(monkeypatch, commands)

    toolchain = resolve_enzyme_toolchain()

    assert toolchain is not None
    assert toolchain.clang == str(clang.resolve())
    assert toolchain.opt == str(opt.resolve())
    assert toolchain.plugin == str(plugin.resolve())
    assert toolchain.metadata["clang"] == "fake clang 18.1.3"
    assert toolchain.metadata["opt"] == "fake opt 18.1.3"


def test_committed_evidence_artifacts_are_valid() -> None:
    """Every committed Enzyme-execution artefact reloads into valid evidence."""
    artifacts = sorted(_EVIDENCE_DIR.glob("enzyme_toolchain_ad_execution_evidence_*.json"))
    assert artifacts, "expected at least one committed Enzyme-execution evidence artefact"
    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rebuilt = _rebuild(payload)
        assert isinstance(rebuilt, EnzymeToolchainADExecutionEvidence)
        if rebuilt.toolchain_available:
            assert rebuilt.beyond_scalar_executed is True
            assert rebuilt.max_gradient_error <= rebuilt.gradient_parity_tolerance


_FAKE_CLANG_SCRIPT = '''#!/usr/bin/env python3
"""Admitted fake clang: copies C source to IR, links marker-driven kernels."""
import sys
from pathlib import Path

args = sys.argv[1:]
if "--version" in args:
    print("fake clang 18.1.3")
    sys.exit(0)
out = Path(args[args.index("-o") + 1])
if "-emit-llvm" in args:
    source = next(argument for argument in args if argument.endswith(".c"))
    out.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    sys.exit(0)
source = next(argument for argument in args if argument.endswith(".ll"))
text = Path(source).read_text(encoding="utf-8")
if "EXITCODE7" in text:
    out.write_text("#!/bin/sh\\nexit 7\\n", encoding="utf-8")
elif "BADOUT" in text:
    out.write_text('#!/bin/sh\\nprintf "not-a-number\\\\n"\\n', encoding="utf-8")
else:
    marker = next(line for line in text.splitlines() if "GRADIENT:" in line)
    values = marker.split("GRADIENT:", 1)[1].split("*", 1)[0].split()
    body = "".join(f'printf "%s\\\\n" "{value}"\\n' for value in values)
    out.write_text("#!/bin/sh\\n" + body, encoding="utf-8")
out.chmod(0o700)
sys.exit(0)
'''

_FAKE_OPT_SCRIPT = '''#!/usr/bin/env python3
"""Admitted fake opt: fails on the OPTFAIL marker, otherwise passes IR through."""
import sys
from pathlib import Path

args = sys.argv[1:]
if "--version" in args:
    print("fake opt 18.1.3")
    sys.exit(0)
out = Path(args[args.index("-o") + 1])
source = next(
    argument
    for argument in args
    if argument.endswith(".ll") and argument != str(out)
)
text = Path(source).read_text(encoding="utf-8")
if "OPTFAIL" in text:
    sys.stderr.write("enzyme pass rejected the kernel: OPTFAIL marker\\n")
    sys.exit(1)
out.write_text(text, encoding="utf-8")
sys.exit(0)
'''


def _install_fake_toolchain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Install an admitted marker-driven fake clang/opt/plugin toolchain."""
    clang = tmp_path / "clang"
    opt = tmp_path / "opt"
    plugin = tmp_path / "LLVMEnzyme-18.so"
    clang.write_text(_FAKE_CLANG_SCRIPT, encoding="utf-8")
    clang.chmod(0o700)
    opt.write_text(_FAKE_OPT_SCRIPT, encoding="utf-8")
    opt.chmod(0o700)
    plugin.write_text("plugin", encoding="utf-8")
    monkeypatch.setenv("SCPN_ENZYME_PLUGIN", str(plugin))
    _patch_which(monkeypatch, {"clang": str(clang), "opt": str(opt)})


def _double_gradient(x: runner.FloatArray) -> runner.FloatArray:
    """Return the analytic gradient of the sum of squares kernel."""
    return cast("runner.FloatArray", np.asarray(2.0 * x, dtype=np.float64))


def _identity_matrix_gradient(x: runner.FloatArray) -> runner.FloatArray:
    """Return the analytic gradient of the 2x2 trace kernel."""
    return np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _fake_case(case_id: str, family: str, body_marker: str, size: int) -> Any:
    """Build a private battery case whose body carries a fake-toolchain marker."""
    references: dict[str, Any] = {
        "fake_scalar_square": _double_gradient,
        "fake_matrix_trace": _identity_matrix_gradient,
    }
    return runner._EnzymeCase(
        case_id=case_id,
        operation_family=family,
        body=body_marker,
        inputs=np.arange(1.0, float(size) + 1.0, dtype=np.float64),
        reference_gradient=references.get(case_id, _double_gradient),
    )


def _patch_battery(monkeypatch: pytest.MonkeyPatch, cases: tuple[Any, ...]) -> None:
    """Replace the runner battery with the injected cases."""
    monkeypatch.setattr(runner, "_battery", lambda: cases)


def test_runner_executes_battery_through_admitted_fake_toolchain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public runner executes real admitted subprocesses end to end."""
    _install_fake_toolchain(monkeypatch, tmp_path)
    scalar = _fake_case("fake_scalar_square", "scalar", "/* GRADIENT:2 4 */ return 0;", 2)
    matrix = _fake_case("fake_matrix_trace", "matrix", "/* GRADIENT:1 0 0 1 */ return 0;", 4)
    _patch_battery(monkeypatch, (scalar, matrix))

    evidence = run_enzyme_toolchain_execution_evidence(artifact_id="fake-toolchain-probe")

    assert evidence.toolchain_available is True
    assert evidence.toolchain["clang"] == "fake clang 18.1.3"
    assert evidence.toolchain["opt"] == "fake opt 18.1.3"
    assert [case.status for case in evidence.cases] == ["executed", "executed"]
    assert evidence.beyond_scalar_executed is True
    assert evidence.executed_operation_families == ("scalar", "matrix")
    assert evidence.max_gradient_error == 0.0


def test_runner_classifies_differentiation_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failing Enzyme pass yields a hard gap naming the stage and reason."""
    _install_fake_toolchain(monkeypatch, tmp_path)
    scalar = _fake_case("fake_scalar_square", "scalar", "/* GRADIENT:2 4 */ return 0;", 2)
    failing = _fake_case("fake_optfail", "vector", "/* OPTFAIL */ return 0;", 3)
    _patch_battery(monkeypatch, (scalar, failing))

    evidence = run_enzyme_toolchain_execution_evidence(artifact_id="fake-optfail-probe")

    gap = evidence.cases[1]
    assert gap.status == "hard_gap"
    assert gap.failure_class is not None
    assert gap.failure_class.startswith("opt:")
    assert "OPTFAIL" in gap.failure_class
    assert evidence.executed_operation_families == ("scalar",)


def test_runner_classifies_nonzero_kernel_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A differentiated kernel exiting non-zero is recorded as an execution hard gap."""
    _install_fake_toolchain(monkeypatch, tmp_path)
    scalar = _fake_case("fake_scalar_square", "scalar", "/* GRADIENT:2 4 */ return 0;", 2)
    crashing = _fake_case("fake_exitcode", "vector", "/* EXITCODE7 */ return 0;", 3)
    _patch_battery(monkeypatch, (scalar, crashing))

    evidence = run_enzyme_toolchain_execution_evidence(artifact_id="fake-exit-probe")

    gap = evidence.cases[1]
    assert gap.status == "hard_gap"
    assert gap.failure_class == "execution returned 7"


def test_runner_classifies_unparseable_kernel_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-numeric kernel output fails closed as a typed hard gap."""
    _install_fake_toolchain(monkeypatch, tmp_path)
    scalar = _fake_case("fake_scalar_square", "scalar", "/* GRADIENT:2 4 */ return 0;", 2)
    garbled = _fake_case("fake_badout", "vector", "/* BADOUT */ return 0;", 3)
    _patch_battery(monkeypatch, (scalar, garbled))

    evidence = run_enzyme_toolchain_execution_evidence(artifact_id="fake-badout-probe")

    gap = evidence.cases[1]
    assert gap.status == "hard_gap"
    assert gap.failure_class is not None
    assert gap.failure_class.startswith("ValueError:")


def test_runner_fails_closed_when_available_toolchain_executes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An available toolchain that executes no case cannot fabricate evidence."""
    _install_fake_toolchain(monkeypatch, tmp_path)
    failing = _fake_case("fake_optfail", "vector", "/* OPTFAIL */ return 0;", 3)
    _patch_battery(monkeypatch, (failing,))

    with pytest.raises(ValueError, match="available toolchain must execute at least one case"):
        run_enzyme_toolchain_execution_evidence(artifact_id="fake-all-gap-probe")


def test_runner_records_gated_battery_when_toolchain_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing toolchain gates every battery case with setup instructions."""
    monkeypatch.delenv("SCPN_ENZYME_PLUGIN", raising=False)
    monkeypatch.setattr(runner, "glob", lambda pattern: [])
    _patch_which(monkeypatch, {})

    evidence = run_enzyme_toolchain_execution_evidence(artifact_id="absent-toolchain-probe")

    assert evidence.toolchain_available is False
    assert evidence.toolchain["status"] == "unavailable"
    assert "SCPN_ENZYME_PLUGIN" in evidence.toolchain["setup"]
    assert all(case.status == "hard_gap" for case in evidence.cases)
    assert all(
        case.failure_class is not None and "toolchain unavailable" in case.failure_class
        for case in evidence.cases
    )
    assert evidence.executed_operation_families == ()
    assert evidence.beyond_scalar_executed is False
    assert evidence.max_gradient_error == 0.0


def test_plugin_discovery_skips_non_file_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prefix discovery skips missing or non-file plugin candidates and fails closed."""
    clang = tmp_path / "clang"
    opt = tmp_path / "opt"
    _write_executable(clang)
    _write_executable(opt)
    directory_candidate = tmp_path / "LLVMEnzyme-19.so"
    directory_candidate.mkdir()
    missing_candidate = tmp_path / "LLVMEnzyme-18.so"
    monkeypatch.delenv("SCPN_ENZYME_PLUGIN", raising=False)
    monkeypatch.setattr(
        runner,
        "glob",
        lambda pattern: [str(missing_candidate), str(directory_candidate)],
    )
    _patch_which(monkeypatch, {"clang": str(clang), "opt": str(opt)})

    assert resolve_enzyme_toolchain() is None


def test_toolchain_resolver_fails_closed_without_path_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing PATH commands gate the resolver before any subprocess runs."""
    plugin = tmp_path / "LLVMEnzyme-18.so"
    plugin.write_text("plugin", encoding="utf-8")
    monkeypatch.setenv("SCPN_ENZYME_PLUGIN", str(plugin))
    _patch_which(monkeypatch, {})

    assert resolve_enzyme_toolchain() is None


def test_probe_version_reports_unknown_for_unrunnable_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Admitted but unrunnable tools degrade to an unknown version, not a crash."""
    clang = tmp_path / "clang"
    opt = tmp_path / "opt"
    plugin = tmp_path / "LLVMEnzyme-18.so"
    clang.write_bytes(b"\x00\x01\x02")
    clang.chmod(0o700)
    opt.write_bytes(b"\x00\x01\x02")
    opt.chmod(0o700)
    plugin.write_text("plugin", encoding="utf-8")
    monkeypatch.setenv("SCPN_ENZYME_PLUGIN", str(plugin))
    _patch_which(monkeypatch, {"clang": str(clang), "opt": str(opt)})

    toolchain = resolve_enzyme_toolchain()

    assert toolchain is not None
    assert toolchain.metadata["clang"] == "unknown"
    assert toolchain.metadata["opt"] == "unknown"


def test_admitted_subprocess_rejects_empty_and_non_executable_commands(
    tmp_path: Path,
) -> None:
    """The subprocess admission boundary rejects empty and non-executable commands."""
    with pytest.raises(ValueError, match="must be non-empty"):
        runner._run_admitted_subprocess((), timeout_seconds=5)
    plain_file = tmp_path / "not-executable"
    plain_file.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="not executable"):
        runner._run_admitted_subprocess(
            (str(plain_file),),
            timeout_seconds=5,
        )


def test_case_contract_rejects_malformed_identity_fields() -> None:
    """The execution-case contract rejects malformed identity metadata."""
    valid: dict[str, Any] = {
        "case_id": "row",
        "operation_family": "vector",
        "operand_dimension": 4,
        "status": "hard_gap",
        "gradient_error": None,
        "runtime_seconds": None,
        "failure_class": "declined",
        "claim_boundary": "bounded",
    }
    for field, value, message in (
        ("case_id", " ", "case_id must be non-empty"),
        ("operation_family", "", "operation_family must be non-empty"),
        ("operand_dimension", 0, "operand_dimension must be positive"),
        ("status", "pending", "status must be executed or hard_gap"),
        ("claim_boundary", "", "claim_boundary must be non-empty"),
    ):
        broken = dict(valid)
        broken[field] = value
        with pytest.raises(ValueError, match=message):
            EnzymeToolchainADCase(**broken)


def test_evidence_contract_rejects_malformed_aggregate_fields() -> None:
    """The aggregate evidence contract rejects malformed metadata and summaries."""
    gap = EnzymeToolchainADCase(
        case_id="gap",
        operation_family="vector",
        operand_dimension=4,
        status="hard_gap",
        gradient_error=None,
        runtime_seconds=None,
        failure_class="declined",
        claim_boundary="bounded",
    )
    executed = EnzymeToolchainADCase(
        case_id="ok",
        operation_family="matrix",
        operand_dimension=4,
        status="executed",
        gradient_error=1e-12,
        runtime_seconds=1e-3,
        failure_class=None,
        claim_boundary="bounded",
    )
    valid: dict[str, Any] = {
        "artifact_id": "probe",
        "toolchain_available": True,
        "toolchain": {"clang": "x"},
        "cases": (executed,),
        "beyond_scalar_executed": True,
        "executed_operation_families": ("matrix",),
        "max_gradient_error": 1e-12,
        "gradient_parity_tolerance": 1e-9,
        "claim_boundary": "bounded",
    }
    for overrides, message in (
        ({"artifact_id": " "}, "artifact_id must be non-empty"),
        ({"cases": ()}, "evidence requires at least one case"),
        ({"claim_boundary": ""}, "claim_boundary must be non-empty"),
        (
            {"gradient_parity_tolerance": -1e-9},
            "gradient_parity_tolerance must be finite and non-negative",
        ),
        (
            {"gradient_parity_tolerance": float("nan")},
            "gradient_parity_tolerance must be finite and non-negative",
        ),
        ({"toolchain": {"clang": ""}}, "toolchain metadata must map non-empty strings"),
        (
            {"cases": (gap,)},
            "available toolchain must execute at least one case",
        ),
        (
            {"executed_operation_families": ("vector",)},
            "executed_operation_families must list executed families in order",
        ),
        (
            {"beyond_scalar_executed": False},
            "beyond_scalar_executed must reflect an executed non-scalar family",
        ),
        (
            {"max_gradient_error": 5e-10},
            "max_gradient_error must equal the worst executed gradient_error",
        ),
    ):
        broken = dict(valid)
        broken.update(overrides)
        with pytest.raises(ValueError, match=message):
            EnzymeToolchainADExecutionEvidence(**broken)
