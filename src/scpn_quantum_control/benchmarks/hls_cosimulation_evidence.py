# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HLS software co-simulation evidence + handoff (RC-3)
"""Hash-bound software co-simulation evidence for the pulse→HLS lane.

Elevates the host-compiler co-simulation of the generated Vivado/Vitis HLS
pulse-player bundle from a test-only check to a first-class, reproducible
evidence artifact:

* the generated header and testbench are compiled under a **host compiler**
  (``g++``) against the packaged non-synthesis shim
  (``codegen/hls_host_shim``) and executed — the testbench asserts the
  streamed words reproduce the quantised envelope bit-for-bit and prints
  ``PASS <n>``;
* every input to that verdict is **hash-bound**: the header, testbench, XDC,
  and each shim header carry SHA-256 content digests, alongside the exact
  compile command and the compiler identity;
* the artifact states its boundary explicitly: this is *codegen + software
  co-simulation only* — **no synthesis, no timing closure, no board
  execution**. The RTL path and the sub-50 ns latency work are owned by
  SC-NEUROCORE, which ingests the emitted directory through the
  ``sc-neurocore.hdl_gen.hls_ingest.v1`` file-system contract.

Fail-closed: a missing host compiler raises (no fabricated evidence); a
compile or run failure is recorded as ``passed=False`` with the captured
output — failure evidence is still evidence.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..codegen.ultrascale_hls import HLSBundle, TargetSku, pulse_to_vivado_hls
from .decisive_run_harness import command_line, dependency_versions, git_commit
from .isolated_host_readiness import HostReadiness, capture_host_readiness

SCHEMA_VERSION = "1.0"

#: Packaged non-synthesis shim backing the AMD Xilinx HLS API under host C++.
HOST_SHIM_DIR = Path(__file__).resolve().parent.parent / "codegen" / "hls_host_shim"

SC_NEUROCORE_CONTRACT = "sc-neurocore.hdl_gen.hls_ingest.v1"

_BOUNDARY_NOTE = (
    "software co-simulation under a host compiler with a non-synthesis shim; "
    "no synthesis, no timing closure, no board execution; the RTL path and "
    f"sub-50 ns latency work are owned by SC-NEUROCORE ({SC_NEUROCORE_CONTRACT})"
)


class CosimulationRunner(Protocol):
    """Callable producing the co-simulation evidence for a bundle."""

    def __call__(self, bundle: HLSBundle, *, compiler: str) -> CosimulationEvidence:
        """Compile and run the bundle testbench and return the evidence."""
        ...


@dataclass(frozen=True)
class CosimulationEvidence:
    """Hash-bound verdict of one host-compiler co-simulation run."""

    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    samples_streamed: int
    compile_command: tuple[str, ...]
    compiler_path: str
    compiler_version: str
    duration_s: float
    sources: tuple[dict[str, Any], ...]
    boundary: str = _BOUNDARY_NOTE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the evidence."""
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "samples_streamed": self.samples_streamed,
            "compile_command": list(self.compile_command),
            "compiler_path": self.compiler_path,
            "compiler_version": self.compiler_version,
            "duration_s": self.duration_s,
            "sources": [dict(source) for source in self.sources],
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class HLSCosimulationConfig:
    """Configuration of the co-simulation evidence run.

    Parameters
    ----------
    n_samples
        Length of the canonical half-sine envelope streamed by the bundle.
    amplitude
        Peak envelope amplitude (must fit the fixed-point range).
    sample_rate_hz
        Replay sample rate of the generated pulse player.
    target_sku
        Target device SKU (``"zu3eg"`` or ``"zu9eg"``).
    fifo_depth, fixed_point_width, fixed_point_frac_bits
        Bundle generation parameters (see
        :func:`~scpn_quantum_control.codegen.ultrascale_hls.pulse_to_vivado_hls`).
    compiler
        Host compiler executable name.
    reserved_core
        CPU core whose isolation state grades the measured wall-clock times.
    """

    n_samples: int = 256
    amplitude: float = 0.8
    sample_rate_hz: float = 250e6
    target_sku: TargetSku = "zu3eg"
    fifo_depth: int = 1024
    fixed_point_width: int = 16
    fixed_point_frac_bits: int = 8
    compiler: str = "g++"
    reserved_core: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises
        ------
        ValueError
            If the sample count is not positive, or the amplitude is not
            finite and positive, or the compiler name is empty.
        """
        if self.n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isfinite(self.amplitude) or self.amplitude <= 0.0:
            raise ValueError("amplitude must be finite and positive")
        if not self.compiler:
            raise ValueError("compiler must be a non-empty executable name")

    def waveform(self) -> Any:
        """Return the canonical deterministic half-sine envelope."""
        return self.amplitude * np.sin(np.linspace(0.0, np.pi, self.n_samples))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the configuration."""
        return {
            "n_samples": self.n_samples,
            "amplitude": self.amplitude,
            "sample_rate_hz": self.sample_rate_hz,
            "target_sku": self.target_sku,
            "fifo_depth": self.fifo_depth,
            "fixed_point_width": self.fixed_point_width,
            "fixed_point_frac_bits": self.fixed_point_frac_bits,
            "compiler": self.compiler,
            "reserved_core": self.reserved_core,
        }


@dataclass(frozen=True)
class HLSCosimulationHandoff:
    """Reproducible co-sim evidence + handoff artifact for SC-NEUROCORE."""

    evidence: dict[str, Any]
    bundle_meta: dict[str, Any]
    consumer_contract: str
    timing_grade: str
    host: dict[str, Any]
    config: dict[str, Any]
    provenance: dict[str, Any]
    notes: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the handoff artifact."""
        return {
            "schema_version": self.schema_version,
            "evidence": self.evidence,
            "bundle_meta": self.bundle_meta,
            "consumer_contract": self.consumer_contract,
            "timing_grade": self.timing_grade,
            "host": self.host,
            "config": self.config,
            "provenance": self.provenance,
            "notes": list(self.notes),
        }


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_records(bundle: HLSBundle, shim_dir: Path) -> tuple[dict[str, Any], ...]:
    """Hash-bind every source that feeds the co-simulation verdict."""
    records: list[dict[str, Any]] = [
        {
            "role": "hls_header",
            "name": "pulse_axi_stream.hpp",
            "sha256": _sha256(bundle.cpp_source),
        },
        {
            "role": "testbench",
            "name": "pulse_axi_stream_tb.cpp",
            "sha256": _sha256(bundle.cpp_testbench),
        },
        {
            "role": "constraints",
            "name": "pulse_player.xdc",
            "sha256": _sha256(bundle.constraints_xdc),
        },
    ]
    for shim_path in sorted(shim_dir.glob("*.h")):
        records.append(
            {
                "role": "host_shim",
                "name": shim_path.name,
                "sha256": _sha256(shim_path.read_text(encoding="utf-8")),
            }
        )
    return tuple(records)


def host_compiler_identity(compiler: str = "g++") -> tuple[str, str]:
    """Resolve the host compiler path and version line (fail-closed).

    Parameters
    ----------
    compiler
        Compiler executable name.

    Returns
    -------
    tuple of (str, str)
        Absolute compiler path and its ``--version`` first line.

    Raises
    ------
    RuntimeError
        If the compiler is not on ``PATH`` — no co-simulation evidence can be
        produced, and none is fabricated.
    """
    path = shutil.which(compiler)
    if path is None:
        raise RuntimeError(
            f"host compiler {compiler!r} not found; refusing to fabricate co-simulation evidence"
        )
    completed = subprocess.run(  # nosec B603 - resolved compiler path, no shell
        [path, "--version"],
        capture_output=True,
        check=True,
        text=True,
        shell=False,
    )
    return path, completed.stdout.splitlines()[0]


def run_hls_cosimulation(bundle: HLSBundle, *, compiler: str = "g++") -> CosimulationEvidence:
    """Compile and execute the bundle testbench under the host compiler.

    Writes the header and testbench to a temporary build directory, compiles
    against the packaged non-synthesis shim, runs the binary, and hash-binds
    every source into the evidence. A compile or run failure is recorded as
    ``passed=False`` with the captured output.

    Parameters
    ----------
    bundle
        Generated HLS bundle from
        :func:`~scpn_quantum_control.codegen.ultrascale_hls.pulse_to_vivado_hls`.
    compiler
        Host compiler executable name.

    Returns
    -------
    CosimulationEvidence
        The hash-bound co-simulation verdict.

    Raises
    ------
    RuntimeError
        If the host compiler is missing (fail-closed).
    """
    compiler_path, compiler_version = host_compiler_identity(compiler)
    sources = _source_records(bundle, HOST_SHIM_DIR)

    with tempfile.TemporaryDirectory(prefix="hls_cosim_") as workdir_name:
        workdir = Path(workdir_name)
        (workdir / "pulse_axi_stream.hpp").write_text(bundle.cpp_source, encoding="utf-8")
        testbench = workdir / "pulse_axi_stream_tb.cpp"
        testbench.write_text(bundle.cpp_testbench, encoding="utf-8")
        binary = workdir / "pulse_tb"

        compile_command = (
            compiler_path,
            "-std=c++17",
            "-Wno-unknown-pragmas",
            f"-I{HOST_SHIM_DIR}",
            f"-I{workdir}",
            str(testbench),
            "-o",
            str(binary),
        )
        started = time.perf_counter()
        compiled = subprocess.run(  # nosec B603 - resolved compiler path, no shell
            list(compile_command),
            capture_output=True,
            check=False,
            text=True,
            shell=False,
        )
        if compiled.returncode != 0:
            return CosimulationEvidence(
                passed=False,
                exit_code=int(compiled.returncode),
                stdout=compiled.stdout,
                stderr=compiled.stderr,
                samples_streamed=0,
                compile_command=compile_command,
                compiler_path=compiler_path,
                compiler_version=compiler_version,
                duration_s=time.perf_counter() - started,
                sources=sources,
            )

        executed = subprocess.run(  # nosec B603 - freshly built local binary, no shell
            [str(binary)],
            capture_output=True,
            check=False,
            text=True,
            shell=False,
        )
        duration = time.perf_counter() - started

    stdout = executed.stdout.strip()
    passed = executed.returncode == 0 and stdout.startswith("PASS ")
    samples_streamed = int(stdout.split()[1]) if passed else 0
    return CosimulationEvidence(
        passed=passed,
        exit_code=int(executed.returncode),
        stdout=executed.stdout,
        stderr=executed.stderr,
        samples_streamed=samples_streamed,
        compile_command=compile_command,
        compiler_path=compiler_path,
        compiler_version=compiler_version,
        duration_s=duration,
        sources=sources,
    )


def run_hls_cosimulation_handoff(
    config: HLSCosimulationConfig | None = None,
    *,
    host_readiness: HostReadiness | None = None,
    cosim_runner: CosimulationRunner = run_hls_cosimulation,
) -> HLSCosimulationHandoff:
    """Produce the RC-3 co-simulation evidence + SC-NEUROCORE handoff artifact.

    Parameters
    ----------
    config
        Run configuration; ``None`` selects :class:`HLSCosimulationConfig`
        defaults.
    host_readiness
        Pre-captured host-isolation verdict; when ``None`` the live host is
        assessed via :func:`~.isolated_host_readiness.capture_host_readiness`.
    cosim_runner
        Co-simulation callable (injectable for tests); defaults to
        :func:`run_hls_cosimulation`.

    Returns
    -------
    HLSCosimulationHandoff
        The hash-bound evidence, bundle metadata, provenance, and the
        explicit no-synthesis boundary notes.
    """
    config = config or HLSCosimulationConfig()

    bundle = pulse_to_vivado_hls(
        config.waveform(),
        config.sample_rate_hz,
        config.target_sku,
        fifo_depth=config.fifo_depth,
        fixed_point_width=config.fixed_point_width,
        fixed_point_frac_bits=config.fixed_point_frac_bits,
    )
    evidence = cosim_runner(bundle, compiler=config.compiler)

    readiness = host_readiness or capture_host_readiness(config.reserved_core)
    timing_grade = "isolated_measured" if readiness.ready else "advisory_shared_host"
    notes = [
        _BOUNDARY_NOTE,
        "the testbench verdict is bit-true against the quantised envelope, "
        "not a hardware timing or fidelity claim",
    ]
    if not readiness.ready:
        notes.append("co-simulation wall-clock measured on a shared host: advisory only")

    return HLSCosimulationHandoff(
        evidence=evidence.to_dict(),
        bundle_meta={
            "target_sku": bundle.target_sku,
            "sample_rate_hz": bundle.sample_rate_hz,
            "fifo_depth": bundle.fifo_depth,
            "n_samples": config.n_samples,
        },
        consumer_contract=SC_NEUROCORE_CONTRACT,
        timing_grade=timing_grade,
        host=asdict(readiness),
        config=config.to_dict(),
        provenance={
            "git_commit": git_commit(),
            "command": command_line(),
            "dependencies": dependency_versions(),
        },
        notes=tuple(notes),
    )
