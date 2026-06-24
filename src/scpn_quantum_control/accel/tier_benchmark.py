# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-tier benchmark measurement and provenance core
"""Reusable measurement, provenance, and artefact assembly for tier benchmarks.

This module is the deterministic core shared by the consolidated tier-benchmark
runner (``scripts/bench_kuramoto_tiers.py``) and the side-by-side documentation
renderer (``tools/render_tier_benchmarks.py``). It follows the unified GOTM
benchmark standard (``agentic-shared/BENCHMARK_STANDARD.md``) and the dispatcher
contract in ``docs/language_policy.md``:

* **Reproducible measurement, never a single run.** :func:`measure` warms up
  (discarding the first calls), then collects ``repeats`` per-call samples with
  :func:`time.perf_counter_ns`. :func:`compute_stats` reduces them to the full
  ``P50 / P95 / P99 / mean / min / max`` set plus throughput — never a lone
  wall-clock average.
* **One row per (operation, backend).** :class:`BackendRow` records a measured
  tier or an explicitly *unavailable* tier with the reason, so a fast backend's
  number publishes automatically the moment its binding is installed and is
  never silently omitted.
* **Full provenance.** :func:`capture_provenance` records the CPU model, host
  platform, the Python / NumPy / Rust-engine / juliacall / rustc tool versions,
  the git commit, the CPU affinity, and the load average so the artefact is
  regenerable and auditable months later.
* **Tamper-evident artefacts.** :func:`build_primitive_artifact` and
  :func:`build_manifest` close each payload with a SHA-256 digest over the
  body, matching ``scripts/benchmark_native_speedup.py``.

The measurement host matters: per the dispatcher design rules, the canonical
number is measured on the fixed CI runner, with the workstation figure committed
as a ``*.local`` sibling and presented side by side. This module is host-neutral
— it records *where* it ran in the provenance block and leaves the comparison to
the renderer.

Language policy: EXEMPT from the Rust-path rule. This is benchmark
instrumentation — timing loops, percentile reduction, and JSON assembly over
the already-compiled tiers it measures; it has no compute hot loop of its own.
See docs/language_policy.md §"Current-state audit".
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .rust_import import optional_rust_engine

# Schema identifiers. Bump the trailing version when the artefact shape changes
# so a reader can branch on layout instead of guessing.
PRIMITIVE_SCHEMA = "scpn-quantum-control.tier-benchmark.v1"
MANIFEST_SCHEMA = "scpn-quantum-control.tier-benchmark-manifest.v1"

# Status strings for a backend row. A measured row carries stats; an unavailable
# row carries a human-readable reason instead.
STATUS_MEASURED = "measured"
STATUS_UNAVAILABLE = "unavailable"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUST_CARGO = _REPO_ROOT / "scpn_quantum_engine" / "Cargo.toml"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierStats:
    """Reduced timing statistics for one (operation, backend) measurement.

    All latency fields are microseconds per call; ``throughput_ops_s`` is the
    reciprocal of the mean. ``samples`` is the number of per-call samples the
    percentiles were drawn from, recorded so a reader can judge their weight.
    """

    p50_us: float
    p95_us: float
    p99_us: float
    mean_us: float
    min_us: float
    max_us: float
    throughput_ops_s: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-ready mapping of the statistics."""

        return {
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "mean_us": self.mean_us,
            "min_us": self.min_us,
            "max_us": self.max_us,
            "throughput_ops_s": self.throughput_ops_s,
            "samples": self.samples,
        }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` quantile of an already-sorted sample list.

    Uses the nearest-rank convention on the closed index range, matching
    ``scripts/benchmark_native_speedup.py`` so the two harnesses agree.
    """

    count = len(ordered)
    index = min(count - 1, int(fraction * (count - 1)))
    return float(ordered[index])


def compute_stats(samples_us: Sequence[float]) -> TierStats:
    """Reduce per-call microsecond samples to the standard statistic set.

    Raises :class:`ValueError` on an empty sample list — a measurement with no
    samples is a bug in the caller, not a zero-cost call.
    """

    if not samples_us:
        raise ValueError("cannot compute statistics from an empty sample list")
    ordered = sorted(float(value) for value in samples_us)
    mean_us = math.fsum(ordered) / len(ordered)
    throughput = 1.0e6 / mean_us if mean_us > 0.0 else 0.0
    return TierStats(
        p50_us=_percentile(ordered, 0.50),
        p95_us=_percentile(ordered, 0.95),
        p99_us=_percentile(ordered, 0.99),
        mean_us=mean_us,
        min_us=ordered[0],
        max_us=ordered[-1],
        throughput_ops_s=throughput,
        samples=len(ordered),
    )


def measure(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    inner: int = 1,
) -> TierStats:
    """Warm up (discarded), then collect ``repeats`` per-call timing samples.

    Each sample times ``inner`` back-to-back calls and divides by ``inner`` so
    that sub-microsecond kernels — where the ``perf_counter_ns`` resolution and
    the loop overhead would otherwise dominate a single call — are measured
    against a hot loop. ``inner`` defaults to 1 for coarse kernels.
    """

    if warmup < 0 or repeats < 1 or inner < 1:
        raise ValueError("require warmup >= 0, repeats >= 1, inner >= 1")
    from time import perf_counter_ns

    for _ in range(warmup):
        fn()
    samples_us: list[float] = []
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(inner):
            fn()
        elapsed_ns = perf_counter_ns() - start
        samples_us.append(elapsed_ns / 1_000.0 / inner)
    return compute_stats(samples_us)


# ---------------------------------------------------------------------------
# Backend and primitive rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendRow:
    """One (operation, backend) measurement row.

    A measured row sets ``status == STATUS_MEASURED`` and carries ``stats``; an
    unavailable row sets ``status == STATUS_UNAVAILABLE``, leaves ``stats`` as
    ``None``, and records ``reason``.
    """

    backend: str
    status: str
    stats: TierStats | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the backend row."""

        return {
            "backend": self.backend,
            "status": self.status,
            "stats": self.stats.to_dict() if self.stats is not None else None,
            "reason": self.reason,
        }


def measured_row(backend: str, stats: TierStats) -> BackendRow:
    """Return a measured backend row carrying ``stats``."""

    return BackendRow(backend=backend, status=STATUS_MEASURED, stats=stats)


def unavailable_row(backend: str, reason: str) -> BackendRow:
    """Return an unavailable backend row carrying ``reason``."""

    return BackendRow(backend=backend, status=STATUS_UNAVAILABLE, reason=reason)


@dataclass(frozen=True)
class PrimitiveResult:
    """All backend rows for one compute primitive at one problem size.

    ``parity_max_abs_diff`` records the largest absolute deviation of any
    measured non-floor tier from the Python floor on the same input, so the
    artefact carries cross-tier agreement evidence alongside the timings.
    """

    operation: str
    size: int
    rows: tuple[BackendRow, ...]
    parity_max_abs_diff: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def fastest_backend(self) -> str | None:
        """Return the measured backend with the lowest P50, or ``None``."""

        measured = [row for row in self.rows if row.stats is not None]
        if not measured:
            return None
        return min(measured, key=lambda row: row.stats.p50_us).backend  # type: ignore[union-attr]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the primitive result."""

        return {
            "operation": self.operation,
            "size": self.size,
            "fastest_backend": self.fastest_backend(),
            "parity_max_abs_diff": self.parity_max_abs_diff,
            "rows": [row.to_dict() for row in self.rows],
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    """Return the CPU model string from ``/proc/cpuinfo`` when available."""

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _affinity() -> list[int] | None:
    """Return the CPU affinity set, or ``None`` where it is unavailable."""

    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:
        return None
    return sorted(getaffinity(0))


def _loadavg() -> list[float] | None:
    """Return the 1/5/15-minute load average, or ``None`` where unavailable."""

    try:
        return [round(value, 2) for value in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def _git_commit() -> str:
    """Return the HEAD commit, or ``unknown`` outside a checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _rustc_version() -> str:
    """Return the host ``rustc`` version line, or ``absent`` when not installed."""

    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "absent"
    return result.stdout.strip() or "absent"


def _engine_label() -> str:
    """Return an identifier for the installed Rust engine, or its absence."""

    engine = optional_rust_engine()
    if engine is None:
        return "absent"
    return str(getattr(engine, "__version__", "installed"))


def _distribution_version(name: str) -> str:
    """Return an installed distribution version, or ``absent`` when missing."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


@dataclass(frozen=True)
class Provenance:
    """Host, toolchain, and revision provenance for a benchmark run."""

    cpu_model: str
    cpu_count: int | None
    python: str
    numpy: str
    engine: str
    juliacall: str
    rustc: str
    platform: str
    machine: str
    commit: str
    cpu_affinity: list[int] | None
    loadavg: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the provenance block."""

        return {
            "cpu_model": self.cpu_model,
            "cpu_count": self.cpu_count,
            "python": self.python,
            "numpy": self.numpy,
            "engine": self.engine,
            "juliacall": self.juliacall,
            "rustc": self.rustc,
            "platform": self.platform,
            "machine": self.machine,
            "commit": self.commit,
            "cpu_affinity": self.cpu_affinity,
            "loadavg": self.loadavg,
        }


def capture_provenance() -> Provenance:
    """Capture the current host, toolchain, and revision provenance."""

    return Provenance(
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count(),
        python=platform.python_version(),
        numpy=np.__version__,
        engine=_engine_label(),
        juliacall=_distribution_version("juliacall"),
        rustc=_rustc_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        commit=_git_commit(),
        cpu_affinity=_affinity(),
        loadavg=_loadavg(),
    )


# ---------------------------------------------------------------------------
# Artefact assembly
# ---------------------------------------------------------------------------


def payload_digest(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a payload, excluding any existing digest field."""

    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    serialised = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialised).hexdigest()


def build_primitive_artifact(
    *,
    environment: str,
    generated_utc: str,
    provenance: Provenance,
    parameters: Mapping[str, Any],
    results: Sequence[PrimitiveResult],
) -> dict[str, Any]:
    """Assemble a single-environment artefact over a set of primitive results.

    ``environment`` is a short label (``ci`` / ``local``) that the renderer uses
    to place the run in the side-by-side comparison. ``parameters`` records the
    sizes, warm-up, repeats, and seed so the run is regenerable.
    """

    artifact: dict[str, Any] = {
        "schema_version": PRIMITIVE_SCHEMA,
        "environment": environment,
        "generated_utc": generated_utc,
        "production_claim_allowed": False,
        "provenance": provenance.to_dict(),
        "parameters": dict(parameters),
        "results": [result.to_dict() for result in results],
    }
    artifact["payload_sha256"] = payload_digest(artifact)
    return artifact


def build_manifest(
    *,
    generated_utc: str,
    provenance: Provenance,
    parameters: Mapping[str, Any],
    results: Sequence[PrimitiveResult],
    tier_availability: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the consolidated reproducibility manifest for one run.

    The manifest is the single auditable record of a benchmark run: the host and
    toolchain provenance, the run parameters, the per-tier availability summary
    (each tier marked available or unavailable with the reason), and a compact
    per-primitive index of the fastest measured backend and its cross-tier
    parity. It does not duplicate the full timing samples — those live in the
    per-environment artefact — but it is sufficient to know exactly what ran,
    where, against which toolchain, and which tier won each primitive.
    """

    index = [
        {
            "operation": result.operation,
            "size": result.size,
            "fastest_backend": result.fastest_backend(),
            "parity_max_abs_diff": result.parity_max_abs_diff,
            "backends": [row.backend for row in result.rows],
        }
        for result in results
    ]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "generated_utc": generated_utc,
        "production_claim_allowed": False,
        "provenance": provenance.to_dict(),
        "parameters": dict(parameters),
        "tier_availability": dict(tier_availability),
        "primitive_count": len(results),
        "primitives": index,
    }
    manifest["payload_sha256"] = payload_digest(manifest)
    return manifest
