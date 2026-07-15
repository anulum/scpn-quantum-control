# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Decisive-advantage classical-baseline run harness
"""Measured classical-baseline harness feeding the decisive-advantage gate.

The decisive-advantage protocol (:mod:`.decisive_advantage_protocol`) declares
*which* comparison must be decided; this harness *runs* the classical side of it
at the preregistered decision size and emits schema-valid rows that
:func:`~.decisive_advantage_protocol.evaluate_decision` can score. It never
fabricates a quantum row: with no QPU credits the ``qpu_hardware`` column is
absent, so the honest outcome of the default protocol is ``inconclusive`` — the
promotion gate stays closed, exactly as the quantum-advantage gap contract
requires.

Every decision baseline produces the same dynamical observable — the
synchronisation order parameter ``R`` at the final evolution time — so the
comparison is like-for-like:

* ``dense_statevector_evolution`` reuses
  :func:`~scpn_quantum_control.hardware.classical.classical_exact_evolution`,
  the exact reference at the decision size; its ``reference_error`` is zero.
* ``classical_ode`` reuses
  :func:`~.classical_baselines.scipy_ode_baseline`, the classical phase model.
* ``mps_tensor_network`` reuses
  :func:`~.classical_baselines.mps_tebd_baseline`; when the tensor-network extra
  is unavailable the row degrades to an explicit size-gated skip rather than a
  fabricated number.

Measurement grade
-----------------
Wall-clock timing is *measured*; ``memory_bytes`` is the documented analytic
memory model shared with the other benchmark surfaces (statevector ``2^n·16 B``,
MPS ``n·2·χ²·16 B``, ODE trajectory ``n_times·n·8 B``). The artifact records the
host-isolation verdict from
:func:`~.isolated_host_readiness.capture_host_readiness`; on a shared host the
timings are labelled advisory, and because the decision is ``inconclusive``
without a QPU row, no advantage is ever claimed on advisory timings.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.bridge import build_knm_paper27, omega_for_oscillators
from scpn_quantum_control.hardware.classical import (
    classical_exact_diag,
    classical_exact_evolution,
)

from .classical_baselines import (
    ClassicalBaselineRun,
    mps_tebd_baseline,
    scipy_ode_baseline,
)
from .decisive_advantage_protocol import (
    DecisiveAdvantageProtocol,
    default_decisive_advantage_protocol,
    evaluate_decision,
)
from .isolated_host_readiness import HostReadiness, capture_host_readiness
from .mps_baseline import exact_memory, mps_memory

#: Repository root, used to resolve the git commit stamped into every row.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Tracked dependencies whose versions are stamped into the provenance record.
_TRACKED_DEPENDENCIES: tuple[str, ...] = ("numpy", "scipy", "qiskit", "quimb", "qutip")


@dataclass(frozen=True)
class DecisiveRunConfig:
    """Deterministic inputs for one decisive classical-baseline run.

    Parameters
    ----------
    t_max
        Total evolution time; must be finite and positive.
    dt
        Time step; must be finite, positive, and not exceed ``t_max``.
    mps_bond_dim
        Bond dimension for the MPS TEBD baseline; must be positive.
    include_mps
        When ``False`` the MPS row is emitted as an explicit configuration-gated
        skip instead of being run. Lets callers avoid the optional tensor-network
        dependency without fabricating a row.
    reserved_core
        CPU core index whose isolation state is captured for the timing-grade
        label; must be non-negative.
    """

    t_max: float = 1.0
    dt: float = 0.1
    mps_bond_dim: int = 32
    include_mps: bool = True
    reserved_core: int = 0

    def __post_init__(self) -> None:
        """Validate the run configuration.

        Raises
        ------
        ValueError
            If any field falls outside its documented bound.
        """
        if not isfinite(self.t_max) or self.t_max <= 0.0:
            raise ValueError("t_max must be finite and positive")
        if not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.dt > self.t_max:
            raise ValueError("dt must not exceed t_max")
        if self.mps_bond_dim < 1:
            raise ValueError("mps_bond_dim must be positive")
        if self.reserved_core < 0:
            raise ValueError("reserved_core must be non-negative")


@dataclass(frozen=True)
class DecisiveRunArtifact:
    """Serialisable record of one decisive classical-baseline run."""

    protocol_id: str
    n_qubits: int
    reference_baseline: str
    reference_order_parameter: float
    timing_grade: str
    rows: tuple[dict[str, Any], ...]
    validation: dict[str, Any]
    decision: dict[str, Any]
    provenance: dict[str, Any]
    host_readiness: dict[str, Any]
    claim_boundary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the full artifact."""
        return {
            "protocol_id": self.protocol_id,
            "n_qubits": self.n_qubits,
            "reference_baseline": self.reference_baseline,
            "reference_order_parameter": self.reference_order_parameter,
            "timing_grade": self.timing_grade,
            "rows": [dict(row) for row in self.rows],
            "validation": dict(self.validation),
            "decision": dict(self.decision),
            "provenance": dict(self.provenance),
            "host_readiness": dict(self.host_readiness),
            "claim_boundary": self.claim_boundary,
            "metadata": dict(self.metadata),
        }


#: Bounded-claim statement embedded in every artifact.
CLAIM_BOUNDARY = (
    "This artifact records measured classical wall-clock timings and the "
    "documented analytic memory model for one Kuramoto-XY decision size. It "
    "contains no quantum-hardware row, so the decisive gate degrades to "
    "'inconclusive' and no quantum advantage is claimed. Timings are "
    "decision-grade only when the host-isolation verdict is 'ready'; otherwise "
    "they are advisory."
)


def _resolve_git_executable() -> str | None:
    """Return the absolute git executable path when git is available."""
    located = shutil.which("git")
    if located is None:
        return None
    try:
        resolved = Path(located).resolve(strict=True)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None
    return str(resolved)


def git_commit() -> str:
    """Return the current repository ``HEAD`` commit, or ``"unknown"``."""
    git_executable = _resolve_git_executable()
    if git_executable is None:
        return "unknown"
    try:
        completed = subprocess.run(  # nosec B603
            [git_executable, "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def command_line() -> str:
    """Return the invoking command line, or ``"python"`` when unavailable."""
    if not sys.argv:
        return "python"
    return " ".join(sys.argv)


def dependency_versions() -> dict[str, str]:
    """Return versions of the tracked runtime dependencies.

    Returns
    -------
    dict of str to str
        Maps ``"python"`` and each tracked package to its installed version, or
        ``"not installed"`` when the package is absent. Read from installed
        metadata so the record reflects the true environment, never a guess.
    """
    versions = {"python": sys.version.split()[0]}
    for package in _TRACKED_DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _provenance() -> dict[str, Any]:
    """Return the shared provenance stamp for every row and the artifact."""
    return {
        "git_commit": git_commit(),
        "command": command_line(),
        "machine": platform.platform(),
        "dependencies": dependency_versions(),
    }


def _relative_error(value: float, reference: float) -> float:
    """Return the relative error of ``value`` against ``reference``.

    Falls back to the absolute error when the reference magnitude is below the
    floating-point noise floor, so a near-zero reference cannot manufacture a
    spuriously huge or infinite relative error.
    """
    magnitude = abs(reference)
    if magnitude < 1e-12:
        return abs(value - reference)
    return abs(value - reference) / magnitude


def _ode_memory_bytes(n_qubits: int, t_max: float, dt: float) -> int:
    """Return the analytic trajectory-storage memory model for the ODE row."""
    n_times = max(1, round(t_max / dt)) + 1
    return n_times * n_qubits * 8


def _base_row(baseline: str, n_qubits: int, protocol_id: str) -> dict[str, Any]:
    """Return a provenance-stamped row skeleton for one baseline."""
    provenance = _provenance()
    return {
        "protocol_id": protocol_id,
        "n_qubits": n_qubits,
        "baseline": baseline,
        "command": provenance["command"],
        "machine": provenance["machine"],
        "dependencies": provenance["dependencies"],
        "git_commit": provenance["git_commit"],
    }


def dense_reference_row(
    n_qubits: int,
    protocol_id: str,
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    t_max: float,
    dt: float,
) -> tuple[dict[str, Any], float]:
    """Run the dense statevector reference and return its row and final ``R``.

    Parameters
    ----------
    n_qubits
        System size in qubits.
    protocol_id
        Protocol identifier stamped into the row.
    K, omega
        Coupling matrix and frequency vector for the Kuramoto-XY problem.
    t_max, dt
        Evolution horizon and step.

    Returns
    -------
    tuple of (dict, float)
        The schema-valid ``ok`` row (with ``reference_error`` of zero, since this
        row *is* the reference) and the reference order parameter ``R``.
    """
    start = time.perf_counter()
    evolution = classical_exact_evolution(n_qubits, t_max, dt, K=K, omega=omega)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    reference_r = float(evolution["R"][-1])
    # Force the sparse eigsh ground-state path (k_eigenvalues set) so the dense
    # diagonalisation branch — which lazily imports the GPU accelerator and can
    # trip a NumPy re-import sentinel mismatch under the coverage tracer — is
    # never taken; a single lowest eigenpair is all the ground energy needs.
    diagonalisation = classical_exact_diag(n_qubits, K=K, omega=omega, k_eigenvalues=1)
    ground_energy = float(diagonalisation["ground_energy"])
    row = _base_row("dense_statevector_evolution", n_qubits, protocol_id)
    row.update(
        {
            "status": "ok",
            "wall_time_ms": elapsed_ms,
            "memory_bytes": exact_memory(n_qubits),
            "metric_payload": {
                "reference_error": 0.0,
                "order_parameter_R": reference_r,
                "ground_energy": ground_energy,
            },
            "notes": ["exact dynamical reference; reference_error is zero by definition"],
        }
    )
    return row, reference_r


def ode_row(
    n_qubits: int,
    protocol_id: str,
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    reference_r: float,
    *,
    t_max: float,
    dt: float,
) -> dict[str, Any]:
    """Run the classical ODE baseline and return its schema-valid row.

    The relative error is taken against ``reference_r``; the classical phase
    model is a different model from the quantum XY dynamics, so a ``reference_error``
    above the accuracy target is an honest, expected result rather than a defect.
    """
    run = scipy_ode_baseline(K, omega, t_max=t_max, dt=dt)
    r_final = run.r_final
    row = _base_row("classical_ode", n_qubits, protocol_id)
    row.update(
        {
            "status": "ok",
            "wall_time_ms": run.elapsed_ms,
            "memory_bytes": _ode_memory_bytes(n_qubits, t_max, dt),
            "metric_payload": {
                "reference_error": _relative_error(float(r_final), reference_r)
                if r_final is not None
                else float("inf"),
                "order_parameter_R": r_final,
            },
            "notes": ["classical phase model; not the quantum XY Hamiltonian"],
        }
    )
    return row


def mps_row(
    run: ClassicalBaselineRun,
    n_qubits: int,
    protocol_id: str,
    reference_r: float,
    *,
    bond_dim: int,
) -> dict[str, Any]:
    """Map an MPS baseline run to an ``ok`` or size-gated ``skipped`` row.

    Parameters
    ----------
    run
        The result of :func:`~.classical_baselines.mps_tebd_baseline`; an
        unavailable run becomes an explicit ``skipped`` row with notes.
    n_qubits, protocol_id, reference_r, bond_dim
        Row metadata and the reference order parameter for the relative error.
    """
    row = _base_row("mps_tensor_network", n_qubits, protocol_id)
    if not run.available:
        row.update(
            {
                "status": "skipped",
                "wall_time_ms": 0.0,
                "memory_bytes": 0,
                "metric_payload": {"unavailable_reason": run.unavailable_reason},
                "notes": [
                    "size-gated skip: tensor-network extra unavailable",
                    str(run.unavailable_reason),
                ],
            }
        )
        return row
    r_final = run.r_final
    row.update(
        {
            "status": "ok",
            "wall_time_ms": run.elapsed_ms,
            "memory_bytes": mps_memory(n_qubits, bond_dim),
            "metric_payload": {
                "reference_error": _relative_error(float(r_final), reference_r)
                if r_final is not None
                else float("inf"),
                "order_parameter_R": r_final,
                "max_bond": bond_dim,
            },
            "notes": ["quimb TEBD evolution of the XY model"],
        }
    )
    return row


def _config_gated_mps_row(n_qubits: int, protocol_id: str) -> dict[str, Any]:
    """Return the ``skipped`` MPS row emitted when ``include_mps`` is ``False``."""
    row = _base_row("mps_tensor_network", n_qubits, protocol_id)
    row.update(
        {
            "status": "skipped",
            "wall_time_ms": 0.0,
            "memory_bytes": 0,
            "metric_payload": {"unavailable_reason": "disabled by configuration"},
            "notes": ["configuration-gated skip: include_mps is False"],
        }
    )
    return row


def _host_readiness_dict(readiness: HostReadiness) -> dict[str, Any]:
    """Return a JSON-serialisable mapping of a host-readiness verdict."""
    return {
        "ready": readiness.ready,
        "reserved_core": readiness.reserved_core,
        "governor": readiness.governor,
        "governor_is_stable": readiness.governor_is_stable,
        "frequency_mhz": readiness.frequency_mhz,
        "load_average": list(readiness.load_average)
        if readiness.load_average is not None
        else None,
        "load_is_low": readiness.load_is_low,
        "blockers": list(readiness.blockers),
    }


def run_decisive_benchmark(
    protocol: DecisiveAdvantageProtocol | None = None,
    config: DecisiveRunConfig | None = None,
    *,
    host_readiness: HostReadiness | None = None,
) -> DecisiveRunArtifact:
    """Run the classical baselines for one decisive-advantage protocol.

    Parameters
    ----------
    protocol
        The decisive protocol to decide; defaults to
        :func:`~.decisive_advantage_protocol.default_decisive_advantage_protocol`.
    config
        Deterministic run configuration; defaults to :class:`DecisiveRunConfig`.
    host_readiness
        Pre-captured host-isolation verdict; when ``None`` the live host is
        assessed via :func:`~.isolated_host_readiness.capture_host_readiness`.

    Returns
    -------
    DecisiveRunArtifact
        The measured rows, the delegated protocol validation, and the
        fail-closed decision (``inconclusive`` for the default no-QPU protocol),
        with full provenance and the timing-grade label.
    """
    protocol = protocol or default_decisive_advantage_protocol()
    config = config or DecisiveRunConfig()
    n_qubits = protocol.criterion.target_size
    protocol_id = protocol.protocol.protocol_id

    coupling = build_knm_paper27(L=n_qubits)
    frequencies = omega_for_oscillators(n_qubits)

    dense, reference_r = dense_reference_row(
        n_qubits, protocol_id, coupling, frequencies, t_max=config.t_max, dt=config.dt
    )
    classical_ode = ode_row(
        n_qubits, protocol_id, coupling, frequencies, reference_r, t_max=config.t_max, dt=config.dt
    )
    if config.include_mps:
        run = mps_tebd_baseline(
            coupling, frequencies, t_max=config.t_max, dt=config.dt, bond_dim=config.mps_bond_dim
        )
        mps = mps_row(run, n_qubits, protocol_id, reference_r, bond_dim=config.mps_bond_dim)
    else:
        mps = _config_gated_mps_row(n_qubits, protocol_id)

    rows = (classical_ode, mps, dense)
    row_list = [dict(row) for row in rows]
    validation = protocol.validate_rows(row_list)
    decision = evaluate_decision(protocol, row_list)

    readiness = host_readiness or capture_host_readiness(config.reserved_core)
    timing_grade = "isolated_measured" if readiness.ready else "advisory_shared_host"

    return DecisiveRunArtifact(
        protocol_id=protocol_id,
        n_qubits=n_qubits,
        reference_baseline="dense_statevector_evolution",
        reference_order_parameter=reference_r,
        timing_grade=timing_grade,
        rows=rows,
        validation=validation.to_dict(),
        decision=decision.to_dict(),
        provenance=_provenance(),
        host_readiness=_host_readiness_dict(readiness),
        claim_boundary=CLAIM_BOUNDARY,
        metadata={
            "t_max": config.t_max,
            "dt": config.dt,
            "mps_bond_dim": config.mps_bond_dim,
            "coupling_source": "paper27",
            "omega_source": "omega_for_oscillators",
            "memory_model": "analytic (statevector/MPS/ODE); wall_time_ms is measured",
        },
    )
