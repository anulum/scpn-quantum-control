# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto external competitive benchmark harness
"""Measured head-to-head comparison of our Kuramoto toolkit against external solvers.

This is the external competitive harness: it runs the *same* deterministic
Kuramoto forward problem through our integrators and through real third-party
solvers, records each solver's final order parameter, its accuracy error against
a high-precision reference, and its wall-clock time, and serialises the whole
comparison with the provenance a credible cross-package claim requires (package
versions, numerical tolerances, host-load context, and an explicit claim
boundary).

The comparison is split across three modules so no single file carries both the
orchestration and every external integration: :mod:`kuramoto_competitive_types`
holds the shared :class:`KuramotoProblem`/:class:`CompetitorRow` value types,
:mod:`kuramoto_external_competitors` holds the external-solver adapters (SciPy,
Julia ``DifferentialEquations``/``DynamicalSystems``/``NetworkDynamics``/
``SciMLSensitivity``, and JIT-compiled-C ``jitcdde``), and this module runs our
own integrator tiers and assembles the record.

Our fixed-step rows run the production RK4 integrator on each installed tier
explicitly — the accelerated Rust ``scpn_quantum_engine`` kernel and the NumPy
Python floor — so the head-to-head reflects the accelerated kernel and records
the true executing language rather than the dispatcher's default choice; the Rust
row fails closed to a build command when the engine tier is not built, exactly as
the external competitors do. Our adaptive row runs the pure-Python DOPRI5
orchestration (which has no accelerated tier).

Claim boundary
--------------
The timings are functional and reproducibility evidence on the recorded host,
not a production-latency, SLA, or universal-hardware claim. A comparison is only
meaningful when the host-readiness context is recorded alongside it, and the
verdict reports honestly where a competitor is faster than our toolkit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any

import numpy as np
from numpy.typing import NDArray

import oscillatools as kuramoto
from oscillatools.accel import diff_kuramoto_rk4 as _rk4
from oscillatools.accel import dispatcher as _dispatcher
from oscillatools.accel import tier_benchmark as _tier

from . import kuramoto_external_competitors as _external
from .isolated_host_readiness import HostReadiness, capture_host_readiness
from .kuramoto_competitive_types import (
    CompetitorRow,
    KuramotoProblem,
    build_default_problem,
)
from .kuramoto_external_competitors import default_julia_runner

#: Documented failure modes shared by every competitive-comparison artefact.
FAILURE_MODES: tuple[str, ...] = (
    "external_unavailable: a competitor that is not installed yields an "
    "available=False row with its install command, not a number; absent rows "
    "do not contribute to the verdict.",
    "julia_cold_start: the first call into a Julia package pays package "
    "precompilation, so its timing is discarded (a warm second solve is timed).",
    "reference_accuracy: the accuracy error is measured against the reference "
    "solver only; if the reference itself is biased the error column is "
    "relative, not absolute ground truth.",
    "fixed_vs_adaptive: fixed-step rows (our RK4) and adaptive rows (SciPy "
    "RK45, our DOPRI5) trade accuracy for time differently, so a faster row is "
    "not automatically better without reading its error column.",
)

#: Bounded-claim statement embedded in every comparison artefact.
CLAIM_BOUNDARY = (
    "Timings are functional and reproducibility evidence on the recorded host, "
    "not a production-latency, SLA, or universal-hardware claim. Competitor "
    "package versions and numerical tolerances are recorded; the verdict states "
    "honestly where a competitor is faster than our toolkit."
)

#: Determinism statement embedded in every comparison artefact.
DETERMINISM = (
    "The Kuramoto problem is built deterministically from the recorded seed, so "
    "the order-parameter values are reproducible across runs and machines. "
    "Wall-clock timing is host-dependent and excluded from the reproducible set."
)

#: Documented command that builds the Rust ``scpn_quantum_engine`` tier when absent.
RUST_ENGINE_BUILD_COMMAND = "cd scpn_quantum_engine && maturin develop --release"

#: Discarded warm-up iterations before our in-process integrators are timed.
_TIMING_WARMUP = 3
#: Timed repeats whose median (P50) per-call wall time is recorded for our rows.
_TIMING_REPEATS = 15

#: Per-family timing methodology, recorded in the artefact so the comparison is honest.
TIMING_METHOD = (
    "Our in-process integrator rows report the median (P50) per-call wall time "
    f"over {_TIMING_REPEATS} timed repeats after {_TIMING_WARMUP} discarded "
    "warm-ups; the SciPy row reports its single high-precision solve; each Julia "
    "and the jitcdde row reports its second (warm) in-solver solve. Absolute "
    "times are inflated under host load, so the reproducible timing quantity is "
    "the within-toolkit Rust-over-Python-floor ratio, not the absolute "
    "milliseconds."
)


@dataclass(frozen=True)
class KuramotoCompetitiveComparison:
    """Serialisable record of one Kuramoto external competitive comparison."""

    n_oscillators: int
    t_max: float
    dt: float
    seed: int
    reference_method: str
    generated_utc: str
    rows: tuple[CompetitorRow, ...]
    host_readiness: HostReadiness
    failure_modes: tuple[str, ...] = FAILURE_MODES
    claim_boundary: str = CLAIM_BOUNDARY
    determinism: str = DETERMINISM
    metadata: dict[str, Any] = field(default_factory=dict)

    def row(self, method: str) -> CompetitorRow:
        """Return the row for ``method``.

        Raises
        ------
        KeyError
            If no row carries the requested method identifier.
        """
        for candidate in self.rows:
            if candidate.method == method:
                return candidate
        raise KeyError(method)

    def fastest_available(self) -> CompetitorRow | None:
        """Return the available row with the smallest elapsed time, if any."""
        timed = [r for r in self.rows if r.available and r.elapsed_ms is not None]
        if not timed:
            return None
        return min(timed, key=lambda r: r.elapsed_ms or float("inf"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for the full comparison."""
        readiness = self.host_readiness
        return {
            "n_oscillators": self.n_oscillators,
            "t_max": self.t_max,
            "dt": self.dt,
            "seed": self.seed,
            "reference_method": self.reference_method,
            "generated_utc": self.generated_utc,
            "rows": [row.to_dict() for row in self.rows],
            "host_readiness": {
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
            },
            "failure_modes": list(self.failure_modes),
            "claim_boundary": self.claim_boundary,
            "determinism": self.determinism,
            "metadata": dict(self.metadata),
        }


#: Clock returning the current UTC timestamp string.
Clock = Callable[[], str]


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _final_r_from_trajectory(trajectory: NDArray[np.float64]) -> float:
    """Return the order parameter of the final phase row of a trajectory."""
    return float(kuramoto.order_parameter(np.asarray(trajectory[-1], dtype=np.float64)))


def _measure_call(call: Callable[[], object]) -> _tier.TierStats:
    """Return warm-up-then-repeat timing statistics for a zero-argument callable.

    Reuses the tier-benchmark measurement loop so our in-process integrator rows
    report a median-of-repeats per-call wall time rather than a single noisy
    ``perf_counter`` reading, which for a few-millisecond kernel is dominated by
    scheduler jitter under host load.
    """
    return _tier.measure(call, warmup=_TIMING_WARMUP, repeats=_TIMING_REPEATS)


def _package_version() -> str:
    """Return the installed ``scpn_quantum_control`` version for our Python-floor rows."""
    from scpn_quantum_control import __version__ as our_version

    return str(our_version)


def _rust_engine_version() -> str:
    """Return the installed Rust engine distribution version, or a coarse label."""
    try:
        return _distribution_version("scpn_quantum_engine")
    except PackageNotFoundError:  # pragma: no cover - the engine is packaged when built
        return "installed"


def _rust_rk4_kernel() -> Callable[..., Any] | None:
    """Return the built Rust RK4 kernel, or ``None`` when the engine/kernel is absent.

    ``dispatcher.available_tiers`` cannot answer this: it probes only the engine's
    ``order_parameter`` export, so an older engine build lacking the
    ``kuramoto_rk4_trajectory`` kernel still reports ``rust`` as available. This
    checks the specific kernel the Rust row actually runs.
    """
    engine = _dispatcher.optional_rust_engine()
    if engine is None:
        return None
    kernel = getattr(engine, "kuramoto_rk4_trajectory", None)
    return kernel if callable(kernel) else None


_Rk4Tier = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, int],
    NDArray[np.float64],
]


def _rk4_forced(tier_call: _Rk4Tier, problem: KuramotoProblem) -> NDArray[np.float64]:
    """Run one forced-tier RK4 trajectory for ``problem`` (bypassing the dispatcher)."""
    return np.asarray(
        tier_call(problem.theta0, problem.omega, problem.coupling, problem.dt, problem.n_steps),
        dtype=np.float64,
    )


def _dispatched_rk4_tier(problem: KuramotoProblem) -> str | None:
    """Return the tier the RK4 facade serves by default for ``problem``.

    This is the accelerated kernel the toolkit runs when the caller does not force
    a tier; the ``ours_rk4_rust`` / ``ours_rk4_python`` rows then force each tier
    explicitly so the artefact carries a head-to-head, not just the served choice.
    """
    kuramoto.kuramoto_rk4_trajectory(
        problem.theta0, problem.omega, problem.coupling, problem.dt, problem.n_steps
    )
    return kuramoto.last_kuramoto_rk4_trajectory_tier_used()


def _rk4_rust_python_parity(problem: KuramotoProblem) -> float | None:
    """Return ``max|rust − python|`` over the RK4 trajectory, or ``None`` if Rust absent.

    Cross-tier agreement evidence: the accelerated kernel is only a credible
    substitute for the floor when the two trajectories coincide to machine
    precision on the same input.
    """
    if _rust_rk4_kernel() is None:
        return None
    rust = _rk4_forced(_rk4._rust_kuramoto_rk4_trajectory, problem)
    python = _rk4_forced(_rk4._python_kuramoto_rk4_trajectory, problem)
    return float(np.max(np.abs(rust - python)))


def _rk4_rust_speedup(rust_row: CompetitorRow, python_row: CompetitorRow) -> float | None:
    """Return the Python-floor-over-Rust median-time ratio, or ``None`` when Rust absent.

    The absolute milliseconds are host-load-dependent, but this within-toolkit
    ratio — both tiers measured back-to-back under the same contention — is the
    reproducible timing quantity the artefact stands behind.
    """
    if not rust_row.available or rust_row.elapsed_ms is None or rust_row.elapsed_ms <= 0.0:
        return None
    if python_row.elapsed_ms is None:
        return None
    return python_row.elapsed_ms / rust_row.elapsed_ms


def _ours_rk4_rust_row(problem: KuramotoProblem) -> CompetitorRow:
    """Time the Rust RK4 tier, or fail closed with the build command when absent."""
    if _rust_rk4_kernel() is None:
        engine_present = _dispatcher.optional_rust_engine() is not None
        reason = (
            "scpn_quantum_engine is installed but this build lacks the "
            "kuramoto_rk4_trajectory kernel; rebuild it to enable the Rust tier"
            if engine_present
            else "scpn_quantum_engine (Rust tier) is not built"
        )
        return CompetitorRow(
            method="ours_rk4_rust",
            backend="scpn kuramoto_rk4_trajectory [Rust tier]",
            family="ours",
            language="rust",
            available=False,
            version=None,
            r_final=None,
            r_error_vs_reference=None,
            elapsed_ms=None,
            install_command=RUST_ENGINE_BUILD_COMMAND,
            unavailable_reason=reason,
        )
    stats = _measure_call(lambda: _rk4_forced(_rk4._rust_kuramoto_rk4_trajectory, problem))
    trajectory = _rk4_forced(_rk4._rust_kuramoto_rk4_trajectory, problem)
    return CompetitorRow(
        method="ours_rk4_rust",
        backend="scpn kuramoto_rk4_trajectory [Rust tier]",
        family="ours",
        language="rust",
        available=True,
        version=_rust_engine_version(),
        r_final=_final_r_from_trajectory(trajectory),
        r_error_vs_reference=None,
        elapsed_ms=stats.p50_us / 1000.0,
    )


def _ours_rk4_python_row(problem: KuramotoProblem) -> CompetitorRow:
    """Time the Python RK4 floor — the always-available correctness reference tier."""
    stats = _measure_call(lambda: _rk4_forced(_rk4._python_kuramoto_rk4_trajectory, problem))
    trajectory = _rk4_forced(_rk4._python_kuramoto_rk4_trajectory, problem)
    return CompetitorRow(
        method="ours_rk4_python",
        backend="scpn kuramoto_rk4_trajectory [Python floor]",
        family="ours",
        language="python",
        available=True,
        version=_package_version(),
        r_final=_final_r_from_trajectory(trajectory),
        r_error_vs_reference=None,
        elapsed_ms=stats.p50_us / 1000.0,
    )


def _ours_dopri_row(problem: KuramotoProblem) -> CompetitorRow:
    """Time our adaptive DOPRI5 — a pure-Python orchestration with no accelerated tier."""

    def _call() -> object:
        return kuramoto.kuramoto_dopri_trajectory(
            problem.theta0, problem.omega, problem.coupling, t_end=problem.t_max
        )

    stats = _measure_call(_call)
    result = kuramoto.kuramoto_dopri_trajectory(
        problem.theta0, problem.omega, problem.coupling, t_end=problem.t_max
    )
    phases = np.asarray(result.phases, dtype=np.float64)
    return CompetitorRow(
        method="ours_dopri",
        backend="scpn kuramoto_dopri_trajectory [Python, adaptive — no accelerated tier]",
        family="ours",
        language="python",
        available=True,
        version=_package_version(),
        r_final=float(kuramoto.order_parameter(phases[-1])),
        r_error_vs_reference=None,
        elapsed_ms=stats.p50_us / 1000.0,
    )


def _with_error(
    row: CompetitorRow, reference_method: str, reference_r: float | None
) -> CompetitorRow:
    """Return ``row`` with its accuracy error against the reference filled in."""
    if not row.available or row.method == reference_method or row.r_final is None:
        return row
    if reference_r is None:
        return row
    from dataclasses import replace

    return replace(row, r_error_vs_reference=abs(row.r_final - reference_r))


def run_kuramoto_competitive_comparison(
    problem: KuramotoProblem | None = None,
    *,
    timeout: float = 180.0,
    clock: Clock = _utc_now,
) -> KuramotoCompetitiveComparison:
    """Run the Kuramoto external competitive comparison.

    Integrates the shared problem with our RK4 fixed-step integrator on each
    installed tier explicitly (``ours_rk4_rust`` forcing the Rust kernel,
    ``ours_rk4_python`` forcing the NumPy floor), with our adaptive DOPRI5, and
    with every external competitor (SciPy, and the Julia ``DifferentialEquations``,
    ``DynamicalSystems``, ``NetworkDynamics`` and ``SciMLSensitivity`` packages,
    and JIT-compiled-C ``jitcdde``); records which tier the RK4 facade serves by
    default and the Rust-over-Python-floor speedup and cross-tier parity;
    designates the SciPy high-precision run as the accuracy reference when present
    (otherwise our DOPRI5), fills each available row's error against that
    reference, and records the full build and host provenance.

    Parameters
    ----------
    problem:
        The shared Kuramoto problem; defaults to :func:`build_default_problem`.
    timeout:
        Hard wall-clock limit in seconds for each external subprocess.
    clock:
        Injectable UTC-timestamp source (defaults to the real wall clock).

    Returns
    -------
    KuramotoCompetitiveComparison
        The full, serialisable comparison record.
    """
    resolved = problem if problem is not None else build_default_problem()

    dispatched_tier = _dispatched_rk4_tier(resolved)
    rust_rk4 = _ours_rk4_rust_row(resolved)
    python_rk4 = _ours_rk4_python_row(resolved)
    rows: list[CompetitorRow] = [
        rust_rk4,
        python_rk4,
        _ours_dopri_row(resolved),
        _external.scipy_row(resolved),
        _external.julia_diffeq_row(resolved, timeout=timeout),
        _external.dynamicalsystems_row(resolved, timeout=timeout),
        _external.networkdynamics_row(resolved, timeout=timeout),
        _external.scimlsensitivity_row(resolved, timeout=timeout),
        _external.jitcdde_row(resolved, timeout=timeout),
    ]

    scipy_available = next(r for r in rows if r.method == "scipy_solve_ivp").available
    reference_method = "scipy_solve_ivp" if scipy_available else "ours_dopri"
    reference_row = next(r for r in rows if r.method == reference_method)
    reference_r = reference_row.r_final

    scored = tuple(_with_error(row, reference_method, reference_r) for row in rows)

    provenance = _tier.capture_provenance()
    return KuramotoCompetitiveComparison(
        n_oscillators=resolved.n_oscillators,
        t_max=resolved.t_max,
        dt=resolved.dt,
        seed=resolved.seed,
        reference_method=reference_method,
        generated_utc=clock(),
        rows=scored,
        host_readiness=capture_host_readiness(),
        metadata={
            "n_steps": resolved.n_steps,
            "available_count": sum(1 for r in scored if r.available),
            "competitor_count": len(scored),
            "dispatched_rk4_tier": dispatched_tier,
            "rk4_rust_python_parity_max_abs_diff": _rk4_rust_python_parity(resolved),
            "rk4_rust_speedup_vs_python_floor": _rk4_rust_speedup(rust_rk4, python_rk4),
            "timing_method": TIMING_METHOD,
            "timing_warmup": _TIMING_WARMUP,
            "timing_repeats": _TIMING_REPEATS,
            "build_provenance": {
                "rust_engine": _rust_engine_version(),
                "rustc": provenance.rustc,
                "juliacall": provenance.juliacall,
                "commit": provenance.commit,
            },
        },
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "CompetitorRow",
    "DETERMINISM",
    "FAILURE_MODES",
    "KuramotoCompetitiveComparison",
    "KuramotoProblem",
    "RUST_ENGINE_BUILD_COMMAND",
    "TIMING_METHOD",
    "build_default_problem",
    "default_julia_runner",
    "run_kuramoto_competitive_comparison",
]
