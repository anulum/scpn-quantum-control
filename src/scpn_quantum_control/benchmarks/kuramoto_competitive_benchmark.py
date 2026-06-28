# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto external competitive benchmark harness
"""Measured head-to-head comparison of our Kuramoto toolkit against external solvers.

This is the external competitive harness deferred by the Kuramoto Phase-5
benchmark closeout: it runs the *same* deterministic Kuramoto forward problem
through our integrators and through real third-party solvers, records each
solver's final order parameter, its accuracy error against a high-precision
reference, and its wall-clock time, and serialises the whole comparison with the
provenance a credible cross-package claim requires (package versions, numerical
tolerances, host-load context, and an explicit claim boundary).

It does not reimplement any solver. Our rows call the public
``scpn_quantum_control.kuramoto`` facade; the SciPy row reuses
:func:`scipy_ode_baseline`; the Julia row shells out to ``DifferentialEquations.jl``.

Fail-closed contract
--------------------
Every external competitor is probed for availability before it is run. A solver
that is not installed (or whose subprocess errors or times out) produces an
``available=False`` row carrying the documented install command and the reason,
never a fabricated number. The harness is therefore complete and reproducible on
any host: installed competitors yield real rows, absent ones yield honest
unavailable rows that flip to live once the package is added.

Claim boundary
--------------
The timings are functional and reproducibility evidence on the recorded host,
not a production-latency, SLA, or universal-hardware claim. A comparison is only
meaningful when the host-readiness context is recorded alongside it, and the
verdict reports honestly where a competitor is faster than our toolkit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control import kuramoto

from .classical_baselines import scipy_ode_baseline
from .isolated_host_readiness import HostReadiness, capture_host_readiness

#: Install command per external competitor, surfaced on every unavailable row.
INSTALL_COMMANDS: Mapping[str, str] = {
    "scipy_solve_ivp": "pip install scipy",
    "julia_diffeq": "julia -e 'using Pkg; Pkg.add(\"DifferentialEquations\")'",
    "networkdynamics_jl": "julia -e 'using Pkg; Pkg.add(\"NetworkDynamics\")'",
    "dynamicalsystems_jl": "julia -e 'using Pkg; Pkg.add(\"DynamicalSystems\")'",
    "scimlsensitivity_jl": "julia -e 'using Pkg; Pkg.add(\"SciMLSensitivity\")'",
    "jitcdde": "pip install jitcdde",
}

#: Documented failure modes shared by every competitive-comparison artifact.
FAILURE_MODES: tuple[str, ...] = (
    "external_unavailable: a competitor that is not installed yields an "
    "available=False row with its install command, not a number; absent rows "
    "do not contribute to the verdict.",
    "julia_cold_start: the first DifferentialEquations.jl call pays package "
    "precompilation, so its timing is discarded unless a warm run is requested.",
    "reference_accuracy: the accuracy error is measured against the reference "
    "solver only; if the reference itself is biased the error column is "
    "relative, not absolute ground truth.",
    "fixed_vs_adaptive: fixed-step rows (our RK4) and adaptive rows (SciPy "
    "RK45, our DOPRI5) trade accuracy for time differently, so a faster row is "
    "not automatically better without reading its error column.",
)

#: Bounded-claim statement embedded in every comparison artifact.
CLAIM_BOUNDARY = (
    "Timings are functional and reproducibility evidence on the recorded host, "
    "not a production-latency, SLA, or universal-hardware claim. Competitor "
    "package versions and numerical tolerances are recorded; the verdict states "
    "honestly where a competitor is faster than our toolkit."
)

#: Determinism statement embedded in every comparison artifact.
DETERMINISM = (
    "The Kuramoto problem is built deterministically from the recorded seed, so "
    "the order-parameter values are reproducible across runs and machines. "
    "Wall-clock timing is host-dependent and excluded from the reproducible set."
)


@dataclass(frozen=True)
class KuramotoProblem:
    """A deterministic Kuramoto forward problem shared by every competitor.

    Parameters
    ----------
    coupling:
        Symmetric ``(n, n)`` coupling matrix ``K_ij``.
    omega:
        Natural-frequency vector of length ``n``.
    theta0:
        Initial phase vector of length ``n``.
    t_max:
        Total integration time; positive.
    dt:
        Fixed-grid step for the fixed-step methods and the SciPy evaluation
        grid; positive and not exceeding ``t_max``.
    seed:
        Seed the problem was generated from, recorded for provenance.
    """

    coupling: NDArray[np.float64]
    omega: NDArray[np.float64]
    theta0: NDArray[np.float64]
    t_max: float
    dt: float
    seed: int

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators in the problem."""
        return int(self.omega.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of fixed-grid steps, ``round(t_max / dt)``."""
        return int(round(self.t_max / self.dt))


def build_default_problem(
    n_oscillators: int = 12,
    *,
    seed: int = 20260628,
    t_max: float = 6.0,
    dt: float = 0.01,
) -> KuramotoProblem:
    """Build the canonical deterministic Kuramoto benchmark problem.

    The coupling is a seeded symmetric non-negative matrix scaled above the
    synchronisation threshold so the dynamics are non-trivial, the frequencies
    are seeded standard-normal, and the initial phases are seeded uniform on
    ``[0, 2*pi)``. The construction is domain-agnostic and depends only on the
    seed, so the comparison is reproducible without any project constant.

    Parameters
    ----------
    n_oscillators:
        Number of oscillators, ``n >= 2``.
    seed:
        Seed for the coupling, frequencies, and initial phases.
    t_max:
        Total integration time; positive.
    dt:
        Fixed-grid step; positive and not exceeding ``t_max``.

    Returns
    -------
    KuramotoProblem
        The deterministic problem.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    if n_oscillators < 2:
        raise ValueError(f"n_oscillators must be >= 2, got {n_oscillators}")
    if t_max <= 0.0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if dt > t_max:
        raise ValueError(f"dt ({dt}) must not exceed t_max ({t_max})")

    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n_oscillators, n_oscillators))
    symmetric = 0.5 * (raw + raw.T)
    np.fill_diagonal(symmetric, 0.0)
    coupling = np.asarray(2.0 * symmetric / n_oscillators, dtype=np.float64)
    omega = np.asarray(rng.standard_normal(n_oscillators), dtype=np.float64)
    theta0 = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators), dtype=np.float64)
    return KuramotoProblem(
        coupling=coupling,
        omega=omega,
        theta0=theta0,
        t_max=float(t_max),
        dt=float(dt),
        seed=int(seed),
    )


@dataclass(frozen=True)
class CompetitorRow:
    """One solver's contribution to the competitive comparison.

    Parameters
    ----------
    method:
        Stable identifier (e.g. ``ours_rk4``, ``scipy_solve_ivp``).
    backend:
        Human-readable backend description.
    family:
        ``ours`` for our toolkit, ``external`` for a third-party competitor.
    language:
        Implementation language of the backend (``python`` or ``julia``).
    available:
        Whether the solver produced a result.
    version:
        Recorded package version, or ``None`` when unavailable/not captured.
    r_final:
        Final Kuramoto order parameter, or ``None`` when unavailable.
    r_error_vs_reference:
        Absolute final-order-parameter error against the reference method, or
        ``None`` for the reference row itself or an unavailable row.
    elapsed_ms:
        Advisory wall-clock time in milliseconds, or ``None`` when unavailable.
    install_command:
        Command to install an absent external competitor, or ``None`` for our
        rows.
    unavailable_reason:
        Populated only when ``available`` is ``False``.
    """

    method: str
    backend: str
    family: str
    language: str
    available: bool
    version: str | None
    r_final: float | None
    r_error_vs_reference: float | None
    elapsed_ms: float | None
    install_command: str | None = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping for the row."""
        return {
            "method": self.method,
            "backend": self.backend,
            "family": self.family,
            "language": self.language,
            "available": self.available,
            "version": self.version,
            "r_final": self.r_final,
            "r_error_vs_reference": self.r_error_vs_reference,
            "elapsed_ms": self.elapsed_ms,
            "install_command": self.install_command,
            "unavailable_reason": self.unavailable_reason,
        }


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


#: Result of a successful external-solver subprocess run.
JuliaRunner = Callable[[KuramotoProblem, float], "dict[str, Any]"]
#: Probe returning whether a named package is importable/installed.
PresenceProbe = Callable[[str], bool]
#: Clock returning the current UTC timestamp string.
Clock = Callable[[], str]


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _final_r_from_trajectory(trajectory: NDArray[np.float64]) -> float:
    """Return the order parameter of the final phase row of a trajectory."""
    return float(kuramoto.order_parameter(np.asarray(trajectory[-1], dtype=np.float64)))


def _run_ours_rk4(problem: KuramotoProblem) -> tuple[float, float]:
    """Integrate the problem with our facade RK4 and return ``(r_final, ms)``."""
    start = time.perf_counter()
    trajectory = kuramoto.kuramoto_rk4_trajectory(
        problem.theta0, problem.omega, problem.coupling, problem.dt, problem.n_steps
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return _final_r_from_trajectory(np.asarray(trajectory, dtype=np.float64)), elapsed_ms


def _run_ours_dopri(problem: KuramotoProblem) -> tuple[float, float]:
    """Integrate the problem with our facade DOPRI5 and return ``(r_final, ms)``."""
    start = time.perf_counter()
    result = kuramoto.kuramoto_dopri_trajectory(
        problem.theta0, problem.omega, problem.coupling, t_end=problem.t_max
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    phases = np.asarray(result.phases, dtype=np.float64)
    return float(kuramoto.order_parameter(phases[-1])), elapsed_ms


def default_julia_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the Kuramoto problem through Julia ``DifferentialEquations.jl``.

    Shells out to a Julia process that integrates the identical networked
    Kuramoto field with ``Tsit5`` and reports the final order parameter, the
    in-Julia solve time, and the package version as JSON on stdout.

    Parameters
    ----------
    problem:
        The shared Kuramoto problem.
    timeout:
        Hard wall-clock limit in seconds for the subprocess.

    Returns
    -------
    dict
        ``{"r_final": float, "elapsed_ms": float, "version": str}``.

    Raises
    ------
    FileNotFoundError
        If no ``julia`` executable is on ``PATH``.
    RuntimeError
        If the subprocess fails, times out, or emits unparsable output (for
        example because ``DifferentialEquations`` is not installed).
    """
    julia = shutil.which("julia")
    if julia is None:
        raise FileNotFoundError("julia executable not found on PATH")

    payload = json.dumps(
        {
            "K": problem.coupling.tolist(),
            "omega": problem.omega.tolist(),
            "theta0": problem.theta0.tolist(),
            "t_max": problem.t_max,
        }
    )
    script = _JULIA_DIFFEQ_SCRIPT
    try:
        completed = subprocess.run(
            [julia, "--startup-file=no", "-e", script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"julia subprocess timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"julia subprocess failed (code {completed.returncode}): "
            f"{completed.stderr.strip()[:400]}"
        )
    try:
        parsed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(f"could not parse julia output: {completed.stdout[:200]!r}") from exc
    return {
        "r_final": float(parsed["r_final"]),
        "elapsed_ms": float(parsed["elapsed_ms"]),
        "version": str(parsed["version"]),
    }


#: Julia program reading the problem as JSON on stdin and emitting a JSON result.
_JULIA_DIFFEQ_SCRIPT = r"""
import Pkg
using DifferentialEquations
using JSON
data = JSON.parse(read(stdin, String))
K = reduce(vcat, [reshape(Float64.(r), 1, :) for r in data["K"]])
omega = Float64.(data["omega"])
theta0 = Float64.(data["theta0"])
tmax = Float64(data["t_max"])
function kuramoto!(du, u, p, t)
    @inbounds for i in eachindex(u)
        acc = omega[i]
        for j in eachindex(u)
            acc += K[i, j] * sin(u[j] - u[i])
        end
        du[i] = acc
    end
end
prob = ODEProblem(kuramoto!, theta0, (0.0, tmax))
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
t0 = time()
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
elapsed_ms = (time() - t0) * 1000.0
final = sol.u[end]
z = sum(exp.(im .* final)) / length(final)
ver = string(Pkg.installed()["DifferentialEquations"])
println(JSON.json(Dict("r_final" => abs(z), "elapsed_ms" => elapsed_ms, "version" => ver)))
"""


def _python_module_present(module: str) -> bool:
    """Return whether ``module`` can be imported in the current interpreter."""
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _ours_row(method: str, backend: str, r_final: float, elapsed_ms: float) -> CompetitorRow:
    """Build an available row for one of our integrators."""
    from scpn_quantum_control import __version__ as our_version

    return CompetitorRow(
        method=method,
        backend=backend,
        family="ours",
        language="python",
        available=True,
        version=str(our_version),
        r_final=r_final,
        r_error_vs_reference=None,
        elapsed_ms=elapsed_ms,
    )


def _unavailable_row(method: str, backend: str, language: str, reason: str) -> CompetitorRow:
    """Build a fail-closed row for an absent or failed external competitor."""
    return CompetitorRow(
        method=method,
        backend=backend,
        family="external",
        language=language,
        available=False,
        version=None,
        r_final=None,
        r_error_vs_reference=None,
        elapsed_ms=None,
        install_command=INSTALL_COMMANDS.get(method),
        unavailable_reason=reason,
    )


def _scipy_row(problem: KuramotoProblem) -> CompetitorRow:
    """Run the SciPy ``solve_ivp`` competitor, failing closed if absent."""
    if not _python_module_present("scipy"):
        return _unavailable_row(
            "scipy_solve_ivp", "scipy.solve_ivp(RK45)", "python", "scipy not installed"
        )
    import scipy

    run = scipy_ode_baseline(
        problem.coupling,
        problem.omega,
        t_max=problem.t_max,
        dt=problem.dt,
        theta0=problem.theta0,
        rtol=1e-10,
        atol=1e-12,
    )
    return CompetitorRow(
        method="scipy_solve_ivp",
        backend=run.backend,
        family="external",
        language="python",
        available=True,
        version=str(scipy.__version__),
        r_final=run.r_final,
        r_error_vs_reference=None,
        elapsed_ms=run.elapsed_ms,
    )


def _julia_diffeq_row(
    problem: KuramotoProblem, *, timeout: float, runner: JuliaRunner
) -> CompetitorRow:
    """Run the Julia ``DifferentialEquations.jl`` competitor, failing closed."""
    try:
        result = runner(problem, timeout)
    except FileNotFoundError:
        return _unavailable_row(
            "julia_diffeq",
            "DifferentialEquations.jl(Tsit5)",
            "julia",
            "julia executable not found on PATH",
        )
    except RuntimeError as exc:
        return _unavailable_row(
            "julia_diffeq", "DifferentialEquations.jl(Tsit5)", "julia", str(exc)
        )
    return CompetitorRow(
        method="julia_diffeq",
        backend="DifferentialEquations.jl(Tsit5)",
        family="external",
        language="julia",
        available=True,
        version=result["version"],
        r_final=result["r_final"],
        r_error_vs_reference=None,
        elapsed_ms=result["elapsed_ms"],
    )


#: External competitors that are declared targets but not yet wired to a live run.
_DECLARED_TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("networkdynamics_jl", "NetworkDynamics.jl", "julia", "NetworkDynamics"),
    ("dynamicalsystems_jl", "DynamicalSystems.jl", "julia", "DynamicalSystems"),
    ("scimlsensitivity_jl", "SciMLSensitivity.jl", "julia", "SciMLSensitivity"),
    ("jitcdde", "jitcdde (just-in-time C)", "python", "jitcdde"),
)


def _declared_target_rows(julia_present: PresenceProbe) -> tuple[CompetitorRow, ...]:
    """Build fail-closed rows for the declared-but-unwired competitor targets.

    A target whose package is installed reports that its live adapter is a
    logged Phase-6.1 follow-up (honest WIP); an absent target reports its
    install command. Neither fabricates a result.
    """
    rows: list[CompetitorRow] = []
    for method, backend, language, package in _DECLARED_TARGETS:
        if language == "python":
            present = _python_module_present(package)
        else:
            present = julia_present(package)
        reason = (
            f"{package} is installed but its live adapter is a logged Phase-6.1 "
            "follow-up, not yet wired"
            if present
            else f"{package} not installed; install with: {INSTALL_COMMANDS[method]}"
        )
        rows.append(_unavailable_row(method, backend, language, reason))
    return tuple(rows)


def _julia_package_present(package: str) -> bool:
    """Return whether a Julia ``package`` is in the active project's manifest."""
    julia = shutil.which("julia")
    if julia is None:
        return False
    try:
        completed = subprocess.run(
            [
                julia,
                "--startup-file=no",
                "-e",
                f'using Pkg; print(haskey(Pkg.project().dependencies, "{package}"))',
            ],
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def run_kuramoto_competitive_comparison(
    problem: KuramotoProblem | None = None,
    *,
    julia_timeout: float = 180.0,
    julia_runner: JuliaRunner = default_julia_runner,
    julia_present: PresenceProbe = _julia_package_present,
    clock: Clock = _utc_now,
) -> KuramotoCompetitiveComparison:
    """Run the Kuramoto external competitive comparison.

    Integrates the shared problem with our RK4 and DOPRI5 integrators and with
    every available external competitor, designates the SciPy high-precision run
    as the accuracy reference when present (otherwise our DOPRI5), fills each
    available row's error against that reference, and records the full provenance.

    Parameters
    ----------
    problem:
        The shared Kuramoto problem; defaults to :func:`build_default_problem`.
    julia_timeout:
        Hard wall-clock limit in seconds for each Julia subprocess.
    julia_runner:
        Injectable runner for the ``DifferentialEquations.jl`` row (defaults to
        the real subprocess runner; overridden in tests).
    julia_present:
        Injectable probe for Julia package presence (defaults to the real
        ``Pkg`` query; overridden in tests).
    clock:
        Injectable UTC-timestamp source (defaults to the real wall clock).

    Returns
    -------
    KuramotoCompetitiveComparison
        The full, serialisable comparison record.
    """
    resolved = problem if problem is not None else build_default_problem()

    rk4_r, rk4_ms = _run_ours_rk4(resolved)
    dopri_r, dopri_ms = _run_ours_dopri(resolved)
    rows: list[CompetitorRow] = [
        _ours_row("ours_rk4", "scpn kuramoto.kuramoto_rk4_trajectory", rk4_r, rk4_ms),
        _ours_row("ours_dopri", "scpn kuramoto.kuramoto_dopri_trajectory", dopri_r, dopri_ms),
        _scipy_row(resolved),
        _julia_diffeq_row(resolved, timeout=julia_timeout, runner=julia_runner),
    ]
    rows.extend(_declared_target_rows(julia_present))

    reference_method = "scipy_solve_ivp" if rows[2].available else "ours_dopri"
    reference_row = next(r for r in rows if r.method == reference_method)
    reference_r = reference_row.r_final

    scored = tuple(_with_error(row, reference_method, reference_r) for row in rows)

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
        },
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
