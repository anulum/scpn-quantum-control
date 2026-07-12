# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR enzyme execution runner module
# scpn-quantum-control -- real Enzyme/LLVM toolchain reverse-mode AD execution runner
"""Capture real Enzyme/LLVM reverse-mode AD execution evidence beyond scalar replay.

This runner drives the installed Enzyme/LLVM toolchain end to end rather than recording
metadata: for each battery case it emits a C source with an ``__enzyme_autodiff`` call,
lowers it to LLVM IR with ``clang``, applies the Enzyme differentiation pass with ``opt``
and the ``LLVMEnzyme`` plugin, links the differentiated IR back to a native executable
with ``clang``, runs it, and checks the toolchain-produced gradient against the analytic
reference. Scalar, vector and 2x2/3x3 matrix families are covered, so the captured
evidence demonstrates compiler-native AD beyond a scalar derivative probe.

When the toolchain is absent (for example in a minimal CI image) the runner fails closed:
every case is recorded as a ``hard_gap`` with setup instructions and
``toolchain_available`` is ``False``, so the evidence is gated rather than fabricated.
"""

from __future__ import annotations

import os
import shutil

# Subprocess is used only through _run_admitted_subprocess.
import subprocess  # nosec B404
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_Command = tuple[str, ...]

ENZYME_TOOLCHAIN_AD_STATUSES = frozenset({"executed", "hard_gap"})

_ENZYME_EXECUTION_CLAIM_BOUNDARY = (
    "Bounded real Enzyme/LLVM reverse-mode AD execution: scalar, vector and 2x2/3x3 "
    "matrix C kernels differentiated by the installed Enzyme pass and run natively, with "
    "the toolchain gradient checked against the analytic reference within float64 "
    "tolerance; no Enzyme-JAX, arbitrary-program, provider, hardware or performance claim."
)

_ENZYME_SETUP_INSTRUCTIONS = (
    "Install LLVM 18 (clang, opt) and the Enzyme LLVMEnzyme-18 plugin; expose the plugin "
    "path via the SCPN_ENZYME_PLUGIN environment variable or under ~/.local/opt/enzyme-*/lib."
)


@dataclass(frozen=True)
class EnzymeToolchainADCase:
    """One real Enzyme/LLVM reverse-mode AD execution row."""

    case_id: str
    operation_family: str
    operand_dimension: int
    status: str
    gradient_error: float | None
    runtime_seconds: float | None
    failure_class: str | None
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate the execution-case evidence invariants."""
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.operation_family.strip():
            raise ValueError("operation_family must be non-empty")
        if self.operand_dimension <= 0:
            raise ValueError("operand_dimension must be positive")
        if self.status not in ENZYME_TOOLCHAIN_AD_STATUSES:
            raise ValueError("status must be executed or hard_gap")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        if self.status == "executed":
            if self.failure_class is not None:
                raise ValueError("executed cases must not carry a failure_class")
            for name, value in (
                ("gradient_error", self.gradient_error),
                ("runtime_seconds", self.runtime_seconds),
            ):
                if value is None or value < 0.0 or not np.isfinite(value):
                    raise ValueError(f"executed cases require a finite non-negative {name}")
        else:
            if not self.failure_class or not self.failure_class.strip():
                raise ValueError("hard_gap cases require a failure_class")
            if self.gradient_error is not None or self.runtime_seconds is not None:
                raise ValueError("hard_gap cases must not carry execution metrics")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready execution-case metadata."""
        return {
            "case_id": self.case_id,
            "operation_family": self.operation_family,
            "operand_dimension": self.operand_dimension,
            "status": self.status,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "failure_class": self.failure_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class EnzymeToolchainADExecutionEvidence:
    """Aggregate real Enzyme/LLVM reverse-mode AD execution evidence."""

    artifact_id: str
    toolchain_available: bool
    toolchain: Mapping[str, str]
    cases: tuple[EnzymeToolchainADCase, ...]
    beyond_scalar_executed: bool
    executed_operation_families: tuple[str, ...]
    max_gradient_error: float
    gradient_parity_tolerance: float
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate aggregate Enzyme execution evidence invariants."""
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if not self.cases:
            raise ValueError("evidence requires at least one case")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        if self.gradient_parity_tolerance < 0.0 or not np.isfinite(self.gradient_parity_tolerance):
            raise ValueError("gradient_parity_tolerance must be finite and non-negative")
        executed = tuple(case for case in self.cases if case.status == "executed")
        toolchain = dict(self.toolchain)
        if any(not key or not value for key, value in toolchain.items()):
            raise ValueError("toolchain metadata must map non-empty strings")
        if self.toolchain_available:
            if not executed:
                raise ValueError("available toolchain must execute at least one case")
        else:
            if executed:
                raise ValueError("unavailable toolchain cannot record executed cases")
        families = tuple(dict.fromkeys(case.operation_family for case in executed))
        if tuple(self.executed_operation_families) != families:
            raise ValueError("executed_operation_families must list executed families in order")
        beyond = any(case.operation_family != "scalar" for case in executed)
        if self.beyond_scalar_executed != beyond:
            raise ValueError("beyond_scalar_executed must reflect an executed non-scalar family")
        observed = max((case.gradient_error or 0.0) for case in executed) if executed else 0.0
        if not np.isclose(self.max_gradient_error, observed, rtol=0.0, atol=1e-18):
            raise ValueError("max_gradient_error must equal the worst executed gradient_error")
        if executed and observed > self.gradient_parity_tolerance:
            raise ValueError("executed gradient_error exceeds the declared parity tolerance")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready aggregate Enzyme execution evidence."""
        return {
            "artifact_id": self.artifact_id,
            "toolchain_available": self.toolchain_available,
            "toolchain": dict(self.toolchain),
            "cases": [case.to_dict() for case in self.cases],
            "beyond_scalar_executed": self.beyond_scalar_executed,
            "executed_operation_families": list(self.executed_operation_families),
            "max_gradient_error": self.max_gradient_error,
            "gradient_parity_tolerance": self.gradient_parity_tolerance,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class _EnzymeToolchain:
    """Resolved Enzyme/LLVM toolchain locations and versions."""

    clang: str
    opt: str
    plugin: str
    metadata: Mapping[str, str]


def _resolve_enzyme_plugin() -> str | None:
    """Return the LLVMEnzyme plugin path from the environment or a known prefix."""
    override = os.environ.get("SCPN_ENZYME_PLUGIN")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_absolute() and candidate.is_file():
            return str(candidate.resolve(strict=True))
        return None
    candidates = sorted(
        glob(str(Path.home() / ".local" / "opt" / "enzyme-*" / "lib" / "LLVMEnzyme-*.so"))
    )
    for plugin_candidate in reversed(candidates):
        path = Path(plugin_candidate)
        if path.is_absolute() and path.is_file():
            return str(path.resolve(strict=True))
    return None


def _resolve_executable(command: str) -> str | None:
    """Return an absolute executable path for a PATH-resolved command."""
    resolved = shutil.which(command)
    if not resolved:
        return None
    path = Path(resolved).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        return None
    return str(path)


def _run_admitted_subprocess(
    command: _Command,
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run a prevalidated no-shell subprocess and capture text output."""
    if not command:
        raise ValueError("subprocess command must be non-empty")
    executable = Path(command[0]).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"subprocess executable is not executable: {command[0]}")
    admitted = (str(executable), *command[1:])
    # The executable is absolute, validated, and always run with shell=False.
    return subprocess.run(  # nosec B603
        admitted,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )


def _probe_version(executable: str) -> str:
    """Return the first --version line for an executable, or 'unknown'."""
    try:
        completed = _run_admitted_subprocess((executable, "--version"), timeout_seconds=30)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return "unknown"
    line = completed.stdout.strip().splitlines()
    return line[0].strip() if line else "unknown"


def resolve_enzyme_toolchain() -> _EnzymeToolchain | None:
    """Resolve clang, opt and the Enzyme plugin, or return None when any is absent."""
    clang = _resolve_executable("clang")
    opt = _resolve_executable("opt")
    plugin = _resolve_enzyme_plugin()
    if not clang or not opt or not plugin:
        return None
    metadata = {
        "clang": _probe_version(clang),
        "opt": _probe_version(opt),
        "enzyme_plugin": Path(plugin).name,
    }
    return _EnzymeToolchain(clang=clang, opt=opt, plugin=plugin, metadata=metadata)


_C_PROGRAM_TEMPLATE = """#include <stdio.h>
__attribute__((noinline)) double scpn_kernel(const double* x, int n) {{
{body}
}}
extern double __enzyme_autodiff(void*, const double*, double*, int);
int main(void) {{
    double x[{n}] = {{{inputs}}};
    double g[{n}] = {{{zeros}}};
    __enzyme_autodiff((void*)scpn_kernel, x, g, {n});
    for (int i = 0; i < {n}; i++) printf("%.17g\\n", g[i]);
    return 0;
}}
"""


@dataclass(frozen=True)
class _EnzymeCase:
    """One battery specification: a C kernel body and its analytic reference gradient."""

    case_id: str
    operation_family: str
    body: str
    inputs: FloatArray
    reference_gradient: Callable[[FloatArray], FloatArray]


def _battery() -> tuple[_EnzymeCase, ...]:
    """Return the scalar, vector and matrix Enzyme execution battery."""
    return (
        _EnzymeCase(
            "scalar_square",
            "scalar",
            "    return x[0] * x[0];",
            np.array([3.0]),
            lambda x: 2.0 * x,
        ),
        _EnzymeCase(
            "vector_sum_squares_4",
            "vector",
            "    double s = 0.0;\n    for (int i = 0; i < n; i++) s += x[i] * x[i];\n    return s;",
            np.array([1.0, 2.0, 3.0, 4.0]),
            lambda x: 2.0 * x,
        ),
        _EnzymeCase(
            "vector_weighted_sum_4",
            "vector",
            (
                "    static const double w[4] = {0.5, -1.0, 2.0, 3.0};\n"
                "    double s = 0.0;\n    for (int i = 0; i < n; i++) s += w[i] * x[i];\n"
                "    return s;"
            ),
            np.array([1.0, 1.0, 1.0, 1.0]),
            lambda x: np.array([0.5, -1.0, 2.0, 3.0]),
        ),
        _EnzymeCase(
            "matrix_trace_2x2",
            "matrix",
            "    return x[0] + x[3];",
            np.array([2.0, 1.0, 1.0, 3.0]),
            lambda x: np.array([1.0, 0.0, 0.0, 1.0]),
        ),
        _EnzymeCase(
            "matrix_frobenius_3x3",
            "matrix",
            "    double s = 0.0;\n    for (int i = 0; i < n; i++) s += x[i] * x[i];\n    return s;",
            np.arange(1.0, 10.0, dtype=np.float64),
            lambda x: 2.0 * x,
        ),
    )


def _run_enzyme_case(case: _EnzymeCase, toolchain: _EnzymeToolchain) -> EnzymeToolchainADCase:
    """Compile, differentiate, link and run one case through the real toolchain."""
    dimension = int(case.inputs.size)
    inputs = ", ".join(f"{value:.17g}" for value in case.inputs.tolist())
    zeros = ", ".join("0.0" for _ in range(dimension))
    source = _C_PROGRAM_TEMPLATE.format(body=case.body, n=dimension, inputs=inputs, zeros=zeros)
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="scpn-enzyme-") as work:
            base = Path(work)
            (base / "kernel.c").write_text(source, encoding="utf-8")
            ir = base / "kernel.ll"
            ad = base / "kernel_ad.ll"
            exe = base / "kernel.exe"
            steps: tuple[_Command, ...] = (
                (
                    toolchain.clang,
                    "-S",
                    "-emit-llvm",
                    "-O2",
                    str(base / "kernel.c"),
                    "-o",
                    str(ir),
                ),
                (
                    toolchain.opt,
                    f"-load-pass-plugin={toolchain.plugin}",
                    "-passes=enzyme",
                    str(ir),
                    "-S",
                    "-o",
                    str(ad),
                ),
                (toolchain.clang, str(ad), "-o", str(exe), "-lm"),
            )
            for command in steps:
                completed = _run_admitted_subprocess(command, timeout_seconds=120)
                if completed.returncode != 0:
                    stage = Path(command[0]).name
                    detail = completed.stderr.strip().splitlines()
                    reason = detail[-1] if detail else f"{stage} returned {completed.returncode}"
                    return EnzymeToolchainADCase(
                        case_id=case.case_id,
                        operation_family=case.operation_family,
                        operand_dimension=dimension,
                        status="hard_gap",
                        gradient_error=None,
                        runtime_seconds=None,
                        failure_class=f"{stage}: {reason}"[:200],
                        claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
                    )
            run = _run_admitted_subprocess((str(exe),), timeout_seconds=60)
            runtime_seconds = time.perf_counter() - start
            if run.returncode != 0:
                return EnzymeToolchainADCase(
                    case_id=case.case_id,
                    operation_family=case.operation_family,
                    operand_dimension=dimension,
                    status="hard_gap",
                    gradient_error=None,
                    runtime_seconds=None,
                    failure_class=f"execution returned {run.returncode}",
                    claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
                )
            produced = np.array([float(token) for token in run.stdout.split()], dtype=np.float64)
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return EnzymeToolchainADCase(
            case_id=case.case_id,
            operation_family=case.operation_family,
            operand_dimension=dimension,
            status="hard_gap",
            gradient_error=None,
            runtime_seconds=None,
            failure_class=f"{type(error).__name__}: {str(error)[:160]}",
            claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
        )
    reference = np.asarray(case.reference_gradient(case.inputs), dtype=np.float64)
    gradient_error = float(np.max(np.abs(produced - reference)))
    return EnzymeToolchainADCase(
        case_id=case.case_id,
        operation_family=case.operation_family,
        operand_dimension=dimension,
        status="executed",
        gradient_error=gradient_error,
        runtime_seconds=float(runtime_seconds),
        failure_class=None,
        claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
    )


def run_enzyme_toolchain_execution_evidence(
    *,
    artifact_id: str = "enzyme-toolchain-ad-execution",
    gradient_parity_tolerance: float = 1e-9,
) -> EnzymeToolchainADExecutionEvidence:
    """Capture real Enzyme/LLVM reverse-mode AD execution evidence beyond scalar replay.

    Resolves the installed Enzyme/LLVM toolchain and runs the scalar, vector and matrix
    battery through it. When the toolchain is absent every case is recorded as a gated
    hard gap with setup instructions instead of being fabricated.
    """
    battery = _battery()
    toolchain = resolve_enzyme_toolchain()
    if toolchain is None:
        cases = tuple(
            EnzymeToolchainADCase(
                case_id=case.case_id,
                operation_family=case.operation_family,
                operand_dimension=int(case.inputs.size),
                status="hard_gap",
                gradient_error=None,
                runtime_seconds=None,
                failure_class="enzyme toolchain unavailable (clang, opt or LLVMEnzyme plugin missing)",
                claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
            )
            for case in battery
        )
        return EnzymeToolchainADExecutionEvidence(
            artifact_id=artifact_id,
            toolchain_available=False,
            toolchain={"status": "unavailable", "setup": _ENZYME_SETUP_INSTRUCTIONS},
            cases=cases,
            beyond_scalar_executed=False,
            executed_operation_families=(),
            max_gradient_error=0.0,
            gradient_parity_tolerance=gradient_parity_tolerance,
            claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
        )
    cases = tuple(_run_enzyme_case(case, toolchain) for case in battery)
    executed = tuple(case for case in cases if case.status == "executed")
    families = tuple(dict.fromkeys(case.operation_family for case in executed))
    max_gradient_error = (
        max((case.gradient_error or 0.0) for case in executed) if executed else 0.0
    )
    return EnzymeToolchainADExecutionEvidence(
        artifact_id=artifact_id,
        toolchain_available=True,
        toolchain=MappingProxyType(dict(toolchain.metadata)),
        cases=cases,
        beyond_scalar_executed=any(case.operation_family != "scalar" for case in executed),
        executed_operation_families=families,
        max_gradient_error=float(max_gradient_error),
        gradient_parity_tolerance=gradient_parity_tolerance,
        claim_boundary=_ENZYME_EXECUTION_CLAIM_BOUNDARY,
    )
