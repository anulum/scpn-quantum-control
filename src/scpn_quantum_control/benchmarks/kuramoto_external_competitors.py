# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — External-solver adapters for the Kuramoto competitive benchmark
"""Real third-party solver adapters for the Kuramoto competitive comparison.

Each adapter runs the *same* deterministic Kuramoto forward problem through an
independent external library and reports its final order parameter, wall-clock
time, and package version as a :class:`~.kuramoto_competitive_types.CompetitorRow`.
No solver is reimplemented: the SciPy row reuses
:func:`~.classical_baselines.scipy_ode_baseline`, and every other competitor is
driven through a subprocess that shells out to the real library (Julia
``DifferentialEquations``/``DynamicalSystems``/``NetworkDynamics``/
``SciMLSensitivity``, or the JIT-compiled-C ``jitcdde``) and returns its result
as JSON.

Fail-closed contract
--------------------
Every adapter is fail-closed: a solver whose toolchain is not installed, or whose
subprocess errors or times out, yields an ``available=False`` row carrying its
install command and the reason, never a fabricated number. The comparison is
therefore complete and reproducible on any host — installed competitors yield
real rows, absent ones honest unavailable rows that flip to live once the package
is added.

The ``jitcdde`` adapter runs its subprocess in a neutral working directory: the
library JIT-compiles its integrator through an in-process ``setuptools`` build,
which would otherwise read the repository ``pyproject.toml`` and abort on its
PEP-639-deprecated ``License ::`` classifier under modern setuptools.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .classical_baselines import scipy_ode_baseline
from .kuramoto_competitive_types import CompetitorRow, KuramotoProblem

#: Install (or build) command per external competitor, surfaced on every unavailable row.
INSTALL_COMMANDS: Mapping[str, str] = {
    "scipy_solve_ivp": "pip install scipy",
    "julia_diffeq": "julia -e 'using Pkg; Pkg.add(\"DifferentialEquations\")'",
    "networkdynamics_jl": 'julia -e \'using Pkg; Pkg.add(["NetworkDynamics", "Graphs"])\'',
    "dynamicalsystems_jl": "julia -e 'using Pkg; Pkg.add(\"DynamicalSystems\")'",
    "scimlsensitivity_jl": "julia -e 'using Pkg; Pkg.add(\"SciMLSensitivity\")'",
    "jitcdde": "pip install jitcdde",
}

#: A subprocess runner: maps a problem and a timeout to a ``{r_final, elapsed_ms, version}`` dict.
SubprocessRunner = Callable[[KuramotoProblem, float], "dict[str, Any]"]


def _python_module_present(module: str) -> bool:
    """Return whether ``module`` can be imported in the current interpreter."""
    import importlib.util

    return importlib.util.find_spec(module) is not None


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


def _problem_payload(problem: KuramotoProblem) -> str:
    """Serialise the shared problem as the JSON every subprocess reads on stdin."""
    return json.dumps(
        {
            "K": problem.coupling.tolist(),
            "omega": problem.omega.tolist(),
            "theta0": problem.theta0.tolist(),
            "t_max": problem.t_max,
        }
    )


def _parse_subprocess_result(stdout: str) -> dict[str, Any]:
    """Parse the trailing JSON line a competitor subprocess emits."""
    try:
        parsed = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(f"could not parse subprocess output: {stdout[:200]!r}") from exc
    return {
        "r_final": float(parsed["r_final"]),
        "elapsed_ms": float(parsed["elapsed_ms"]),
        "version": str(parsed["version"]),
    }


def _resolve_julia_executable() -> str:
    """Return the admitted absolute Julia executable path from ``PATH``."""
    located = shutil.which("julia")
    if located is None:
        raise FileNotFoundError("julia executable not found on PATH")
    try:
        executable_path = Path(located)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"julia executable must resolve to an absolute executable path: {located!r}"
        ) from exc
    if not executable_path.is_absolute():
        raise FileNotFoundError(
            f"julia executable must resolve to an absolute executable path: {located!r}"
        )
    try:
        resolved = executable_path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"julia executable must point to an executable file: {located!r}"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"julia executable must point to an executable file: {resolved}")
    return str(resolved)


def _validated_python_interpreter() -> str:
    """Return the current interpreter path after executable admission."""
    raw = sys.executable
    if not raw:
        raise RuntimeError("python interpreter path is not configured")
    try:
        executable_path = Path(raw)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"python interpreter path is not executable: {raw!r}") from exc
    if not executable_path.is_absolute():
        raise RuntimeError(f"python interpreter path must be absolute: {raw!r}")
    try:
        resolved = executable_path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"python interpreter path is not executable: {raw!r}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"python interpreter path is not executable: {resolved}")
    return str(resolved)


def _run_julia_script(script: str, problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run one Julia competitor ``script`` on the shared problem, failing closed.

    Raises
    ------
    FileNotFoundError
        If no ``julia`` executable is on ``PATH``.
    RuntimeError
        If the subprocess fails, times out, or emits unparsable output (for
        example because the competitor package is not installed).
    """
    julia = _resolve_julia_executable()
    try:
        completed = subprocess.run(  # nosec B603
            [julia, "--startup-file=no", "-e", script],
            input=_problem_payload(problem),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"julia subprocess timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"julia subprocess failed (code {completed.returncode}): "
            f"{completed.stderr.strip()[:400]}"
        )
    return _parse_subprocess_result(completed.stdout)


def _run_jitcdde(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the JIT-compiled-C ``jitcdde`` competitor, failing closed.

    The subprocess runs in a temporary working directory so ``jitcdde``'s
    in-process ``setuptools`` build of its native integrator does not read the
    repository ``pyproject.toml`` (whose deprecated ``License ::`` classifier
    modern setuptools rejects, which would otherwise silently drop ``jitcdde`` to
    its slow Python-lambdified fallback).

    Raises
    ------
    FileNotFoundError
        If ``jitcdde`` is not importable in the current interpreter.
    RuntimeError
        If the subprocess fails, times out, or emits unparsable output.
    """
    if not _python_module_present("jitcdde"):
        raise FileNotFoundError("jitcdde is not installed")
    python = _validated_python_interpreter()
    try:
        with tempfile.TemporaryDirectory() as neutral_cwd:
            completed = subprocess.run(  # nosec B603
                [python, "-c", _JITCDDE_SCRIPT],
                input=_problem_payload(problem),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=neutral_cwd,
                check=False,
                shell=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"jitcdde subprocess timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"jitcdde subprocess failed (code {completed.returncode}): "
            f"{completed.stderr.strip()[:400]}"
        )
    return _parse_subprocess_result(completed.stdout)


# --------------------------------------------------------------------------- #
# Embedded competitor programs (each reads the problem JSON on stdin, emits a
# ``{r_final, elapsed_ms, version}`` JSON line, and warms once before timing).
# --------------------------------------------------------------------------- #

_JULIA_DIFFEQ_SCRIPT = r"""
import Pkg
using DifferentialEquations
using JSON
data = JSON.parse(read(stdin, String))
K = reduce(vcat, [reshape(Float64.(r), 1, :) for r in data["K"]])
omega = Float64.(data["omega"]); theta0 = Float64.(data["theta0"]); tmax = Float64(data["t_max"])
function kuramoto!(du, u, p, t)
    @inbounds for i in eachindex(u)
        acc = omega[i]
        for j in eachindex(u); acc += K[i, j] * sin(u[j] - u[i]); end
        du[i] = acc
    end
end
prob = ODEProblem(kuramoto!, theta0, (0.0, tmax))
solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
t0 = time()
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
elapsed_ms = (time() - t0) * 1000.0
z = sum(exp.(im .* sol.u[end])) / length(sol.u[end])
ver = ""
for (uuid, info) in Pkg.dependencies(); if info.name == "DifferentialEquations"; global ver = string(info.version); end; end
println(JSON.json(Dict("r_final" => abs(z), "elapsed_ms" => elapsed_ms, "version" => ver)))
"""

_JULIA_DYNAMICALSYSTEMS_SCRIPT = r"""
import Pkg
using DynamicalSystems
using JSON
data = JSON.parse(read(stdin, String))
K = reduce(vcat, [reshape(Float64.(r), 1, :) for r in data["K"]])
omega = Float64.(data["omega"]); theta0 = Float64.(data["theta0"]); tmax = Float64(data["t_max"])
function kuramoto!(du, u, p, t)
    K, omega = p
    @inbounds for i in eachindex(u)
        acc = omega[i]
        for j in eachindex(u); acc += K[i, j] * sin(u[j] - u[i]); end
        du[i] = acc
    end
end
function final_state()
    ds = CoupledODEs(kuramoto!, copy(theta0), (K, omega); diffeq=(reltol=1e-8, abstol=1e-10))
    step!(ds, tmax, true)
    return copy(current_state(ds))
end
final_state()
t0 = time()
final = final_state()
elapsed_ms = (time() - t0) * 1000.0
z = sum(exp.(im .* final)) / length(final)
ver = ""
for (uuid, info) in Pkg.dependencies(); if info.name == "DynamicalSystems"; global ver = string(info.version); end; end
println(JSON.json(Dict("r_final" => abs(z), "elapsed_ms" => elapsed_ms, "version" => ver)))
"""

_JULIA_NETWORKDYNAMICS_SCRIPT = r"""
import Pkg
using NetworkDynamics
using Graphs
using DifferentialEquations
using JSON
data = JSON.parse(read(stdin, String))
K = reduce(vcat, [reshape(Float64.(r), 1, :) for r in data["K"]])
omega = Float64.(data["omega"]); theta0 = Float64.(data["theta0"]); tmax = Float64(data["t_max"])
n = length(omega)
Base.@propagate_inbounds function kuramoto_edge!(e, θ_s, θ_d, (K,), t)
    e .= K .* sin(θ_s[1] - θ_d[1])
end
Base.@propagate_inbounds function kuramoto_vertex!(dθ, θ, esum, (ω,), t)
    dθ[1] = ω + esum[1]
end
vertexf = VertexModel(; f=kuramoto_vertex!, sym=[:θ], psym=[:ω], g=1)
edgef = EdgeModel(; g=AntiSymmetric(kuramoto_edge!), outsym=[:P], psym=[:K])
g = complete_graph(n)
nw = Network(g, vertexf, edgef)
s0 = NWState(nw)
for i in 1:n
    s0.v[i, :θ] = theta0[i]
    s0.p.v[i, :ω] = omega[i]
end
for (k, e) in enumerate(edges(g))
    s0.p.e[k, :K] = K[src(e), dst(e)]
end
prob = ODEProblem(nw, uflat(s0), (0.0, tmax), pflat(s0))
solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
t0 = time()
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
elapsed_ms = (time() - t0) * 1000.0
final = sol.u[end]
z = sum(exp.(im .* final)) / length(final)
ver = ""
for (uuid, info) in Pkg.dependencies(); if info.name == "NetworkDynamics"; global ver = string(info.version); end; end
println(JSON.json(Dict("r_final" => abs(z), "elapsed_ms" => elapsed_ms, "version" => ver)))
"""

_JULIA_SCIMLSENSITIVITY_SCRIPT = r"""
import Pkg
using DifferentialEquations
using SciMLSensitivity
using JSON
data = JSON.parse(read(stdin, String))
K = reduce(vcat, [reshape(Float64.(r), 1, :) for r in data["K"]])
omega = Float64.(data["omega"]); theta0 = Float64.(data["theta0"]); tmax = Float64(data["t_max"])
function kuramoto!(du, u, p, t)
    @inbounds for i in eachindex(u)
        acc = omega[i]
        for j in eachindex(u); acc += K[i, j] * sin(u[j] - u[i]); end
        du[i] = acc
    end
end
prob = ODEProblem(kuramoto!, theta0, (0.0, tmax))
solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
t0 = time()
sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, save_everystep=false)
elapsed_ms = (time() - t0) * 1000.0
z = sum(exp.(im .* sol.u[end])) / length(sol.u[end])
ver = ""
for (uuid, info) in Pkg.dependencies(); if info.name == "SciMLSensitivity"; global ver = string(info.version); end; end
println(JSON.json(Dict("r_final" => abs(z), "elapsed_ms" => elapsed_ms, "version" => ver)))
"""

_JITCDDE_SCRIPT = r"""
import json, sys, time
import numpy as np
import symengine
from jitcdde import jitcdde, y
data = json.loads(sys.stdin.read())
K = np.asarray(data["K"], dtype=float)
omega = np.asarray(data["omega"], dtype=float)
theta0 = [float(x) for x in data["theta0"]]
tmax = float(data["t_max"])
n = len(omega)
f = [omega[i] + symengine.Add(*[K[i, j] * symengine.sin(y(j) - y(i)) for j in range(n)]) for i in range(n)]
DDE = jitcdde(f, n=n, verbose=False)
DDE.constant_past(theta0)
DDE.initial_discontinuities_handled = True
DDE.compile_C()
DDE.set_integration_parameters(rtol=1e-8, atol=1e-10)
t0 = time.perf_counter()
final = np.asarray(DDE.integrate(tmax), dtype=float)
elapsed_ms = (time.perf_counter() - t0) * 1000.0
z = np.mean(np.exp(1j * final))
import jitcdde as J
print(json.dumps({"r_final": float(abs(z)), "elapsed_ms": elapsed_ms, "version": J.__version__}))
"""


def default_julia_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the Julia ``DifferentialEquations.jl`` competitor (Tsit5)."""
    return _run_julia_script(_JULIA_DIFFEQ_SCRIPT, problem, timeout)


def default_dynamicalsystems_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the Julia ``DynamicalSystems.jl`` ``CoupledODEs`` competitor."""
    return _run_julia_script(_JULIA_DYNAMICALSYSTEMS_SCRIPT, problem, timeout)


def default_networkdynamics_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the Julia ``NetworkDynamics.jl`` complete-graph Kuramoto competitor."""
    return _run_julia_script(_JULIA_NETWORKDYNAMICS_SCRIPT, problem, timeout)


def default_scimlsensitivity_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the Julia ``SciMLSensitivity.jl`` (differentiable stack) forward solve."""
    return _run_julia_script(_JULIA_SCIMLSENSITIVITY_SCRIPT, problem, timeout)


def default_jitcdde_runner(problem: KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Run the JIT-compiled-C ``jitcdde`` competitor from a neutral directory."""
    return _run_jitcdde(problem, timeout)


def scipy_row(problem: KuramotoProblem) -> CompetitorRow:
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


def _subprocess_row(
    method: str,
    backend: str,
    language: str,
    problem: KuramotoProblem,
    *,
    timeout: float,
    runner: SubprocessRunner,
) -> CompetitorRow:
    """Run one subprocess competitor and build its row, failing closed on error."""
    try:
        result = runner(problem, timeout)
    except FileNotFoundError as exc:
        return _unavailable_row(method, backend, language, str(exc))
    except RuntimeError as exc:
        return _unavailable_row(method, backend, language, str(exc))
    return CompetitorRow(
        method=method,
        backend=backend,
        family="external",
        language=language,
        available=True,
        version=result["version"],
        r_final=result["r_final"],
        r_error_vs_reference=None,
        elapsed_ms=result["elapsed_ms"],
    )


def julia_diffeq_row(
    problem: KuramotoProblem,
    *,
    timeout: float = 180.0,
    runner: SubprocessRunner = default_julia_runner,
) -> CompetitorRow:
    """Run the Julia ``DifferentialEquations.jl(Tsit5)`` competitor, failing closed."""
    return _subprocess_row(
        "julia_diffeq",
        "DifferentialEquations.jl(Tsit5)",
        "julia",
        problem,
        timeout=timeout,
        runner=runner,
    )


def dynamicalsystems_row(
    problem: KuramotoProblem,
    *,
    timeout: float = 180.0,
    runner: SubprocessRunner = default_dynamicalsystems_runner,
) -> CompetitorRow:
    """Run the Julia ``DynamicalSystems.jl(CoupledODEs)`` competitor, failing closed."""
    return _subprocess_row(
        "dynamicalsystems_jl",
        "DynamicalSystems.jl(CoupledODEs)",
        "julia",
        problem,
        timeout=timeout,
        runner=runner,
    )


def networkdynamics_row(
    problem: KuramotoProblem,
    *,
    timeout: float = 180.0,
    runner: SubprocessRunner = default_networkdynamics_runner,
) -> CompetitorRow:
    """Run the Julia ``NetworkDynamics.jl`` complete-graph competitor, failing closed."""
    return _subprocess_row(
        "networkdynamics_jl",
        "NetworkDynamics.jl(complete graph)",
        "julia",
        problem,
        timeout=timeout,
        runner=runner,
    )


def scimlsensitivity_row(
    problem: KuramotoProblem,
    *,
    timeout: float = 180.0,
    runner: SubprocessRunner = default_scimlsensitivity_runner,
) -> CompetitorRow:
    """Run the Julia ``SciMLSensitivity.jl`` differentiable-stack forward solve."""
    return _subprocess_row(
        "scimlsensitivity_jl",
        "SciMLSensitivity.jl(Tsit5 forward)",
        "julia",
        problem,
        timeout=timeout,
        runner=runner,
    )


def jitcdde_row(
    problem: KuramotoProblem,
    *,
    timeout: float = 300.0,
    runner: SubprocessRunner = default_jitcdde_runner,
) -> CompetitorRow:
    """Run the JIT-compiled-C ``jitcdde`` competitor, failing closed if absent."""
    return _subprocess_row(
        "jitcdde",
        "jitcdde(just-in-time C)",
        "python",
        problem,
        timeout=timeout,
        runner=runner,
    )
