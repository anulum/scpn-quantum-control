# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Kuramoto external competitive benchmark
"""Module-specific tests for :mod:`kuramoto_competitive_benchmark`.

The suite exercises the real production surfaces — the public facade
integrators, the SciPy baseline, and the serialisable comparison contract —
while the Julia subprocess boundary is driven through injected runners/probes
and monkeypatched ``subprocess``/``shutil`` so the tests are deterministic and
do not require a Julia toolchain on the runner.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import kuramoto_competitive_benchmark as m
from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness


def _small_problem() -> m.KuramotoProblem:
    """A tiny but non-trivial deterministic problem for fast real runs."""
    return m.build_default_problem(n_oscillators=6, t_max=0.5, dt=0.1, seed=7)


def _fake_runner_ok(problem: m.KuramotoProblem, timeout: float) -> dict[str, Any]:
    """Injected Julia runner returning a fixed valid result."""
    return {"r_final": 0.5, "elapsed_ms": 2.5, "version": "7.17.0"}


# --------------------------------------------------------------------------- #
# build_default_problem + KuramotoProblem
# --------------------------------------------------------------------------- #


def test_build_default_problem_shapes_and_properties() -> None:
    problem = m.build_default_problem(n_oscillators=8, t_max=1.0, dt=0.05, seed=3)
    assert problem.coupling.shape == (8, 8)
    assert problem.omega.shape == (8,)
    assert problem.theta0.shape == (8,)
    assert problem.n_oscillators == 8
    assert problem.n_steps == 20
    # symmetric, zero diagonal, non-negative
    assert np.allclose(problem.coupling, problem.coupling.T)
    assert np.allclose(np.diag(problem.coupling), 0.0)
    assert np.all(problem.coupling >= 0.0)


def test_build_default_problem_is_deterministic() -> None:
    a = m.build_default_problem(seed=11)
    b = m.build_default_problem(seed=11)
    assert np.array_equal(a.coupling, b.coupling)
    assert np.array_equal(a.omega, b.omega)
    assert np.array_equal(a.theta0, b.theta0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_oscillators": 1}, "n_oscillators must be >= 2"),
        ({"t_max": 0.0}, "t_max must be positive"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"t_max": 0.1, "dt": 0.2}, "must not exceed"),
    ],
)
def test_build_default_problem_rejects_bad_bounds(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        m.build_default_problem(**kwargs)


# --------------------------------------------------------------------------- #
# Row + comparison dataclasses
# --------------------------------------------------------------------------- #


def test_competitor_row_to_dict_round_trips_fields() -> None:
    row = m.CompetitorRow(
        method="x",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="1.2.3",
        r_final=0.4,
        r_error_vs_reference=0.01,
        elapsed_ms=3.0,
        install_command=None,
        unavailable_reason=None,
    )
    assert row.to_dict()["method"] == "x"
    assert json.dumps(row.to_dict())  # serialisable


def _manual_comparison(
    rows: tuple[m.CompetitorRow, ...], *, load_none: bool = False
) -> m.KuramotoCompetitiveComparison:
    readiness = HostReadiness(
        ready=False,
        reserved_core=0,
        governor="performance",
        governor_is_stable=True,
        frequency_mhz=3900.0,
        load_average=None if load_none else (0.1, 0.2, 0.3),
        load_is_low=True,
        blockers=("demo",),
    )
    return m.KuramotoCompetitiveComparison(
        n_oscillators=4,
        t_max=1.0,
        dt=0.1,
        seed=1,
        reference_method="scipy_solve_ivp",
        generated_utc="2026-06-28T00:00:00Z",
        rows=rows,
        host_readiness=readiness,
    )


def test_comparison_row_lookup_and_missing() -> None:
    row = m.CompetitorRow(
        method="ours_rk4",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="v",
        r_final=0.5,
        r_error_vs_reference=None,
        elapsed_ms=1.0,
    )
    comp = _manual_comparison((row,))
    assert comp.row("ours_rk4") is row
    with pytest.raises(KeyError):
        comp.row("absent")


def test_fastest_available_picks_min_time_and_handles_empty() -> None:
    fast = m.CompetitorRow(
        method="a",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="v",
        r_final=0.5,
        r_error_vs_reference=None,
        elapsed_ms=1.0,
    )
    slow = m.CompetitorRow(
        method="c",
        backend="b",
        family="external",
        language="julia",
        available=True,
        version="v",
        r_final=0.5,
        r_error_vs_reference=0.0,
        elapsed_ms=9.0,
    )
    absent = m.CompetitorRow(
        method="d",
        backend="b",
        family="external",
        language="julia",
        available=False,
        version=None,
        r_final=None,
        r_error_vs_reference=None,
        elapsed_ms=None,
    )
    assert _manual_comparison((slow, fast, absent)).fastest_available() is fast
    assert _manual_comparison((absent,)).fastest_available() is None


def test_comparison_to_dict_serialisable_both_load_branches() -> None:
    row = m.CompetitorRow(
        method="ours_rk4",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="v",
        r_final=0.5,
        r_error_vs_reference=None,
        elapsed_ms=1.0,
    )
    with_load = _manual_comparison((row,)).to_dict()
    assert with_load["host_readiness"]["load_average"] == [0.1, 0.2, 0.3]
    assert json.dumps(with_load)
    without_load = _manual_comparison((row,), load_none=True).to_dict()
    assert without_load["host_readiness"]["load_average"] is None
    assert json.dumps(without_load)


# --------------------------------------------------------------------------- #
# Helpers + our-integrator adapters (real surfaces)
# --------------------------------------------------------------------------- #


def test_utc_now_returns_zulu_timestamp() -> None:
    stamp = m._utc_now()
    assert stamp.endswith("Z") and "T" in stamp


def test_python_module_present_true_and_false() -> None:
    assert m._python_module_present("numpy") is True
    assert m._python_module_present("definitely_not_a_real_module_xyz") is False


def test_ours_python_and_dopri_rows_agree_on_order_parameter() -> None:
    problem = _small_problem()
    python_row = m._ours_rk4_python_row(problem)
    dopri_row = m._ours_dopri_row(problem)
    assert python_row.available and dopri_row.available
    assert python_row.method == "ours_rk4_python" and dopri_row.method == "ours_dopri"
    assert python_row.language == "python" and dopri_row.language == "python"
    assert python_row.version == m._package_version()
    assert python_row.elapsed_ms is not None and python_row.elapsed_ms >= 0.0
    assert dopri_row.elapsed_ms is not None and dopri_row.elapsed_ms >= 0.0
    assert python_row.r_final is not None and dopri_row.r_final is not None
    assert abs(python_row.r_final - dopri_row.r_final) < 1e-3
    assert 0.0 <= python_row.r_final <= 1.0


def test_final_r_from_trajectory_synchronised_is_one() -> None:
    trajectory = np.zeros((3, 5), dtype=np.float64)
    assert m._final_r_from_trajectory(trajectory) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# SciPy competitor row
# --------------------------------------------------------------------------- #


def test_scipy_row_available_and_matches_ours() -> None:
    problem = _small_problem()
    row = m._scipy_row(problem)
    assert row.available is True
    assert row.family == "external" and row.language == "python"
    assert row.version is not None
    rk4_r = m._ours_rk4_python_row(problem).r_final
    assert rk4_r is not None
    assert row.r_final is not None and abs(row.r_final - rk4_r) < 1e-2


def test_scipy_row_fails_closed_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_python_module_present", lambda module: False)
    row = m._scipy_row(_small_problem())
    assert row.available is False
    assert row.unavailable_reason == "scipy not installed"
    assert row.install_command == m.INSTALL_COMMANDS["scipy_solve_ivp"]


# --------------------------------------------------------------------------- #
# Julia DifferentialEquations.jl runner + row (subprocess boundary mocked)
# --------------------------------------------------------------------------- #


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_default_julia_runner_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError):
        m.default_julia_runner(_small_problem(), 5.0)


def test_default_julia_runner_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"r_final": 0.42, "elapsed_ms": 3.1, "version": "7.17.0"})
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Completed(0, stdout=f"noise\n{payload}\n")
    )
    result = m.default_julia_runner(_small_problem(), 5.0)
    assert result == {"r_final": 0.42, "elapsed_ms": 3.1, "version": "7.17.0"}


def test_default_julia_runner_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Completed(1, stderr="boom: package missing")
    )
    with pytest.raises(RuntimeError, match="julia subprocess failed"):
        m.default_julia_runner(_small_problem(), 5.0)


def test_default_julia_runner_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="julia", timeout=5.0)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", _raise)
    with pytest.raises(RuntimeError, match="timed out"):
        m.default_julia_runner(_small_problem(), 5.0)


def test_default_julia_runner_unparsable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout="not json"))
    with pytest.raises(RuntimeError, match="could not parse"):
        m.default_julia_runner(_small_problem(), 5.0)


def test_default_julia_runner_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout=""))
    with pytest.raises(RuntimeError, match="could not parse"):
        m.default_julia_runner(_small_problem(), 5.0)


def test_julia_diffeq_row_success() -> None:
    row = m._julia_diffeq_row(_small_problem(), timeout=5.0, runner=_fake_runner_ok)
    assert row.available is True
    assert row.method == "julia_diffeq" and row.language == "julia"
    assert row.version == "7.17.0"


def test_julia_diffeq_row_missing_executable() -> None:
    def _missing(problem: m.KuramotoProblem, timeout: float) -> dict[str, Any]:
        raise FileNotFoundError("julia executable not found on PATH")

    row = m._julia_diffeq_row(_small_problem(), timeout=5.0, runner=_missing)
    assert row.available is False
    assert row.unavailable_reason == "julia executable not found on PATH"
    assert row.install_command == m.INSTALL_COMMANDS["julia_diffeq"]


def test_julia_diffeq_row_runtime_error() -> None:
    def _boom(problem: m.KuramotoProblem, timeout: float) -> dict[str, Any]:
        raise RuntimeError("DifferentialEquations not installed")

    row = m._julia_diffeq_row(_small_problem(), timeout=5.0, runner=_boom)
    assert row.available is False
    assert row.unavailable_reason == "DifferentialEquations not installed"


# --------------------------------------------------------------------------- #
# Declared targets + Julia package probe
# --------------------------------------------------------------------------- #


def test_declared_targets_absent_carry_install_commands() -> None:
    rows = m._declared_target_rows(julia_present=lambda _pkg: False)
    methods = {r.method for r in rows}
    assert methods == {
        "networkdynamics_jl",
        "dynamicalsystems_jl",
        "scimlsensitivity_jl",
        "jitcdde",
    }
    for row in rows:
        assert row.available is False
        assert row.install_command == m.INSTALL_COMMANDS[row.method]
        assert "not installed" in (row.unavailable_reason or "")


def test_declared_targets_present_report_unwired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_python_module_present", lambda _module: True)
    rows = m._declared_target_rows(julia_present=lambda _pkg: True)
    for row in rows:
        assert row.available is False
        assert "not yet wired" in (row.unavailable_reason or "")


def test_julia_package_present_no_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert m._julia_package_present("DifferentialEquations") is False


def test_julia_package_present_true_and_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout="true"))
    assert m._julia_package_present("DifferentialEquations") is True
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout="false"))
    assert m._julia_package_present("NetworkDynamics") is False


def test_julia_package_present_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_k: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="julia", timeout=60.0)

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/julia")
    monkeypatch.setattr(subprocess, "run", _raise)
    assert m._julia_package_present("DifferentialEquations") is False


# --------------------------------------------------------------------------- #
# _with_error edge
# --------------------------------------------------------------------------- #


def test_with_error_none_reference_is_passthrough() -> None:
    row = m.CompetitorRow(
        method="ours_rk4",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="v",
        r_final=0.5,
        r_error_vs_reference=None,
        elapsed_ms=1.0,
    )
    assert m._with_error(row, "scipy_solve_ivp", None) is row


def test_with_error_fills_absolute_error() -> None:
    row = m.CompetitorRow(
        method="ours_rk4",
        backend="b",
        family="ours",
        language="python",
        available=True,
        version="v",
        r_final=0.5,
        r_error_vs_reference=None,
        elapsed_ms=1.0,
    )
    scored = m._with_error(row, "scipy_solve_ivp", 0.4)
    assert scored.r_error_vs_reference == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Timing + version + engine helpers
# --------------------------------------------------------------------------- #


def test_measure_call_returns_positive_p50() -> None:
    stats = m._measure_call(lambda: sum(range(10)))
    assert stats.samples == m._TIMING_REPEATS
    assert stats.p50_us >= 0.0


def test_package_version_matches_distribution() -> None:
    from scpn_quantum_control import __version__ as expected

    assert m._package_version() == str(expected)


def test_rust_engine_version_is_a_string() -> None:
    assert isinstance(m._rust_engine_version(), str)
    assert m._rust_engine_version() != ""


def test_rust_engine_version_falls_back_when_metadata_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_name: str) -> str:
        raise m.PackageNotFoundError(_name)

    monkeypatch.setattr(m, "_distribution_version", _raise)
    assert m._rust_engine_version() == "installed"


def test_rk4_forced_returns_full_trajectory() -> None:
    problem = _small_problem()
    trajectory = m._rk4_forced(m._rk4._python_kuramoto_rk4_trajectory, problem)
    assert trajectory.shape == (problem.n_steps + 1, problem.n_oscillators)


def test_dispatched_rk4_tier_reports_served_tier() -> None:
    tier = m._dispatched_rk4_tier(_small_problem())
    assert tier in {"rust", "julia", "python"}


# --------------------------------------------------------------------------- #
# Rust / Python-floor tier rows (engine boundary mocked for determinism)
# --------------------------------------------------------------------------- #


class _FakeEngine:
    """Stand-in Rust engine exposing (or omitting) the RK4 kernel attribute."""

    def __init__(self, *, with_kernel: bool) -> None:
        if with_kernel:
            self.kuramoto_rk4_trajectory = lambda *a, **k: None


def _install_fake_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the Rust tier appear built and route its kernel to the Python floor."""
    monkeypatch.setattr(
        m._dispatcher, "optional_rust_engine", lambda: _FakeEngine(with_kernel=True)
    )
    monkeypatch.setattr(
        m._rk4, "_rust_kuramoto_rk4_trajectory", m._rk4._python_kuramoto_rk4_trajectory
    )


def test_rust_rk4_kernel_present_absent_and_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m._dispatcher, "optional_rust_engine", lambda: None)
    assert m._rust_rk4_kernel() is None
    monkeypatch.setattr(
        m._dispatcher, "optional_rust_engine", lambda: _FakeEngine(with_kernel=False)
    )
    assert m._rust_rk4_kernel() is None
    monkeypatch.setattr(
        m._dispatcher, "optional_rust_engine", lambda: _FakeEngine(with_kernel=True)
    )
    assert m._rust_rk4_kernel() is not None


def test_ours_rk4_rust_row_available_records_true_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rust(monkeypatch)
    problem = _small_problem()
    row = m._ours_rk4_rust_row(problem)
    assert row.available is True
    assert row.method == "ours_rk4_rust" and row.family == "ours"
    assert row.language == "rust"
    assert row.version == m._rust_engine_version()
    assert row.elapsed_ms is not None and row.elapsed_ms >= 0.0
    # Routed to the Python floor, so it must equal the floor's order parameter.
    assert row.r_final == pytest.approx(m._ours_rk4_python_row(problem).r_final)


def test_ours_rk4_rust_row_fails_closed_when_engine_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m._dispatcher, "optional_rust_engine", lambda: None)
    row = m._ours_rk4_rust_row(_small_problem())
    assert row.available is False
    assert row.language == "rust"
    assert row.install_command == m.RUST_ENGINE_BUILD_COMMAND
    assert "not built" in (row.unavailable_reason or "")


def test_ours_rk4_rust_row_fails_closed_when_kernel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m._dispatcher, "optional_rust_engine", lambda: _FakeEngine(with_kernel=False)
    )
    row = m._ours_rk4_rust_row(_small_problem())
    assert row.available is False
    assert "lacks the" in (row.unavailable_reason or "")
    assert row.install_command == m.RUST_ENGINE_BUILD_COMMAND


def test_rk4_rust_python_parity_present_and_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m._dispatcher, "optional_rust_engine", lambda: None)
    assert m._rk4_rust_python_parity(_small_problem()) is None
    _install_fake_rust(monkeypatch)
    parity = m._rk4_rust_python_parity(_small_problem())
    assert parity is not None and parity == pytest.approx(0.0, abs=1e-12)


def _row(method: str, *, available: bool, elapsed_ms: float | None) -> m.CompetitorRow:
    return m.CompetitorRow(
        method=method,
        backend="b",
        family="ours",
        language="rust" if "rust" in method else "python",
        available=available,
        version="v" if available else None,
        r_final=0.5 if available else None,
        r_error_vs_reference=None,
        elapsed_ms=elapsed_ms,
    )


def test_rk4_rust_speedup_ratio_and_none_paths() -> None:
    rust = _row("ours_rk4_rust", available=True, elapsed_ms=2.0)
    python = _row("ours_rk4_python", available=True, elapsed_ms=18.0)
    assert m._rk4_rust_speedup(rust, python) == pytest.approx(9.0)
    absent = _row("ours_rk4_rust", available=False, elapsed_ms=None)
    assert m._rk4_rust_speedup(absent, python) is None
    zero = _row("ours_rk4_rust", available=True, elapsed_ms=0.0)
    assert m._rk4_rust_speedup(zero, python) is None
    no_python = _row("ours_rk4_python", available=True, elapsed_ms=None)
    assert m._rk4_rust_speedup(rust, no_python) is None


# --------------------------------------------------------------------------- #
# Full orchestration
# --------------------------------------------------------------------------- #


def test_run_comparison_scipy_reference_and_errors() -> None:
    comp = m.run_kuramoto_competitive_comparison(
        _small_problem(),
        julia_runner=_fake_runner_ok,
        julia_present=lambda _pkg: False,
        clock=lambda: "2026-06-28T00:00:00Z",
    )
    assert comp.reference_method == "scipy_solve_ivp"
    assert comp.generated_utc == "2026-06-28T00:00:00Z"
    assert comp.row("scipy_solve_ivp").r_error_vs_reference is None
    rk4_error = comp.row("ours_rk4_python").r_error_vs_reference
    assert rk4_error is not None
    assert rk4_error < 1e-2
    # Both fixed-step tiers are present, one row per language.
    assert comp.row("ours_rk4_python").language == "python"
    assert comp.row("ours_rk4_rust").language == "rust"
    assert comp.metadata["competitor_count"] == len(comp.rows)
    assert comp.metadata["available_count"] >= 3  # python rk4, dopri, scipy, julia(fake)
    # The dispatched tier and timing methodology are recorded for provenance.
    assert comp.metadata["dispatched_rk4_tier"] in {"rust", "julia", "python"}
    assert comp.metadata["timing_repeats"] == m._TIMING_REPEATS
    assert comp.metadata["timing_warmup"] == m._TIMING_WARMUP
    assert set(comp.metadata["build_provenance"]) == {
        "rust_engine",
        "rustc",
        "juliacall",
        "commit",
    }
    assert json.dumps(comp.to_dict())


def test_run_comparison_default_problem_branch() -> None:
    comp = m.run_kuramoto_competitive_comparison(
        julia_runner=_fake_runner_ok,
        julia_present=lambda _pkg: False,
        clock=lambda: "2026-06-28T00:00:00Z",
    )
    assert comp.n_oscillators == m.build_default_problem().n_oscillators


def test_run_comparison_falls_back_to_dopri_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_python_module_present", lambda _module: False)
    comp = m.run_kuramoto_competitive_comparison(
        _small_problem(),
        julia_runner=_fake_runner_ok,
        julia_present=lambda _pkg: False,
        clock=lambda: "2026-06-28T00:00:00Z",
    )
    assert comp.reference_method == "ours_dopri"
    assert comp.row("scipy_solve_ivp").available is False
    assert comp.row("ours_rk4_python").r_error_vs_reference is not None
