# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Kuramoto competitive benchmark orchestrator
"""Tests for :mod:`kuramoto_competitive_benchmark`.

The orchestrator runs our own integrator tiers (the accelerated Rust RK4 kernel,
the NumPy floor, and the adaptive DOPRI5) against the real production facade and
assembles the serialisable comparison. The external adapters are exercised in
:mod:`test_kuramoto_external_competitors`; here they are monkeypatched to canned
rows so the orchestration is tested without a Julia toolchain or a C compiler,
while the Rust-tier engine boundary is mocked for determinism on runners that do
not build the native engine.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import kuramoto_competitive_benchmark as m
from scpn_quantum_control.benchmarks.isolated_host_readiness import HostReadiness
from scpn_quantum_control.benchmarks.kuramoto_competitive_types import (
    CompetitorRow,
    build_default_problem,
)


def _small_problem() -> m.KuramotoProblem:
    """A tiny but non-trivial deterministic problem for fast real runs."""
    return build_default_problem(n_oscillators=6, t_max=0.5, dt=0.1, seed=7)


# --------------------------------------------------------------------------- #
# Comparison dataclass
# --------------------------------------------------------------------------- #


def _row(method: str, *, available: bool = True, elapsed_ms: float | None = 1.0) -> CompetitorRow:
    return CompetitorRow(
        method=method,
        backend="b",
        family="external",
        language="julia" if "julia" in method else "python",
        available=available,
        version="v" if available else None,
        r_final=0.5 if available else None,
        r_error_vs_reference=None,
        elapsed_ms=elapsed_ms,
    )


def _manual_comparison(
    rows: tuple[CompetitorRow, ...], *, load_none: bool = False
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
        generated_utc="2026-07-03T00:00:00Z",
        rows=rows,
        host_readiness=readiness,
    )


def test_comparison_row_lookup_and_missing() -> None:
    row = _row("ours_rk4_python")
    comp = _manual_comparison((row,))
    assert comp.row("ours_rk4_python") is row
    with pytest.raises(KeyError):
        comp.row("absent")


def test_fastest_available_picks_min_time_and_handles_empty() -> None:
    fast = _row("a", elapsed_ms=1.0)
    slow = _row("c", elapsed_ms=9.0)
    absent = _row("d", available=False, elapsed_ms=None)
    assert _manual_comparison((slow, fast, absent)).fastest_available() is fast
    assert _manual_comparison((absent,)).fastest_available() is None


def test_comparison_to_dict_serialisable_both_load_branches() -> None:
    with_load = _manual_comparison((_row("ours_rk4_python"),)).to_dict()
    assert with_load["host_readiness"]["load_average"] == [0.1, 0.2, 0.3]
    assert json.dumps(with_load)
    without_load = _manual_comparison((_row("ours_rk4_python"),), load_none=True).to_dict()
    assert without_load["host_readiness"]["load_average"] is None
    assert json.dumps(without_load)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def test_utc_now_returns_zulu_timestamp() -> None:
    stamp = m._utc_now()
    assert stamp.endswith("Z") and "T" in stamp


def test_final_r_from_trajectory_synchronised_is_one() -> None:
    trajectory = np.zeros((3, 5), dtype=np.float64)
    assert m._final_r_from_trajectory(trajectory) == pytest.approx(1.0)


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


def test_ours_python_and_dopri_rows_agree_on_order_parameter() -> None:
    problem = _small_problem()
    python_row = m._ours_rk4_python_row(problem)
    dopri_row = m._ours_dopri_row(problem)
    assert python_row.available and dopri_row.available
    assert python_row.method == "ours_rk4_python" and dopri_row.method == "ours_dopri"
    assert python_row.language == "python"
    assert dopri_row.language in {"rust", "julia", "python"}  # the served adaptive-DOPRI tier
    assert python_row.version == m._package_version()
    assert python_row.elapsed_ms is not None and python_row.elapsed_ms >= 0.0
    assert dopri_row.elapsed_ms is not None and dopri_row.elapsed_ms >= 0.0
    assert python_row.r_final is not None and dopri_row.r_final is not None
    assert abs(python_row.r_final - dopri_row.r_final) < 1e-3
    assert 0.0 <= python_row.r_final <= 1.0


def test_ours_rk4_rust_row_available_records_true_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_rust(monkeypatch)
    problem = _small_problem()
    row = m._ours_rk4_rust_row(problem)
    assert row.available is True
    assert row.method == "ours_rk4_rust" and row.family == "ours"
    assert row.language == "rust"
    assert row.version == m._rust_engine_version()
    assert row.elapsed_ms is not None and row.elapsed_ms >= 0.0
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


def test_rk4_rust_speedup_ratio_and_none_paths() -> None:
    rust = _row("ours_rk4_rust", elapsed_ms=2.0)
    python = _row("ours_rk4_python", elapsed_ms=18.0)
    assert m._rk4_rust_speedup(rust, python) == pytest.approx(9.0)
    absent = _row("ours_rk4_rust", available=False, elapsed_ms=None)
    assert m._rk4_rust_speedup(absent, python) is None
    zero = _row("ours_rk4_rust", elapsed_ms=0.0)
    assert m._rk4_rust_speedup(zero, python) is None
    no_python = _row("ours_rk4_python", elapsed_ms=None)
    assert m._rk4_rust_speedup(rust, no_python) is None


# --------------------------------------------------------------------------- #
# _with_error
# --------------------------------------------------------------------------- #


def test_with_error_none_reference_is_passthrough() -> None:
    row = _row("ours_rk4_python")
    assert m._with_error(row, "scipy_solve_ivp", None) is row


def test_with_error_fills_absolute_error() -> None:
    row = _row("ours_rk4_python")  # r_final = 0.5
    scored = m._with_error(row, "scipy_solve_ivp", 0.4)
    assert scored.r_error_vs_reference == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Full orchestration (external subprocess rows monkeypatched to canned rows)
# --------------------------------------------------------------------------- #


def _external_row(method: str, language: str, r_final: float) -> CompetitorRow:
    return CompetitorRow(
        method=method,
        backend=f"{method} backend",
        family="external",
        language=language,
        available=True,
        version="ext",
        r_final=r_final,
        r_error_vs_reference=None,
        elapsed_ms=5.0,
    )


def _stub_external(monkeypatch: pytest.MonkeyPatch, *, scipy_available: bool = True) -> None:
    """Replace every subprocess external adapter with a canned in-process row."""
    r = 0.3775532
    if scipy_available:
        monkeypatch.setattr(
            m._external, "scipy_row", lambda p: _external_row("scipy_solve_ivp", "python", r)
        )
    else:
        monkeypatch.setattr(
            m._external,
            "scipy_row",
            lambda p: CompetitorRow(
                method="scipy_solve_ivp",
                backend="scipy",
                family="external",
                language="python",
                available=False,
                version=None,
                r_final=None,
                r_error_vs_reference=None,
                elapsed_ms=None,
            ),
        )
    monkeypatch.setattr(
        m._external,
        "julia_diffeq_row",
        lambda p, *, timeout: _external_row("julia_diffeq", "julia", r),
    )
    monkeypatch.setattr(
        m._external,
        "dynamicalsystems_row",
        lambda p, *, timeout: _external_row("dynamicalsystems_jl", "julia", r),
    )
    monkeypatch.setattr(
        m._external,
        "networkdynamics_row",
        lambda p, *, timeout: _external_row("networkdynamics_jl", "julia", r),
    )
    monkeypatch.setattr(
        m._external,
        "scimlsensitivity_row",
        lambda p, *, timeout: _external_row("scimlsensitivity_jl", "julia", r),
    )
    monkeypatch.setattr(
        m._external, "jitcdde_row", lambda p, *, timeout: _external_row("jitcdde", "python", r)
    )


def test_run_comparison_scipy_reference_rows_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_external(monkeypatch)
    comp = m.run_kuramoto_competitive_comparison(
        _small_problem(), timeout=5.0, clock=lambda: "2026-07-03T00:00:00Z"
    )
    assert comp.reference_method == "scipy_solve_ivp"
    assert comp.generated_utc == "2026-07-03T00:00:00Z"
    assert comp.row("scipy_solve_ivp").r_error_vs_reference is None
    # Every competitor is present, once per language for our fixed-step tiers.
    methods = {row.method for row in comp.rows}
    assert {
        "ours_rk4_rust",
        "ours_rk4_python",
        "ours_dopri",
        "scipy_solve_ivp",
        "julia_diffeq",
        "dynamicalsystems_jl",
        "networkdynamics_jl",
        "scimlsensitivity_jl",
        "jitcdde",
    } <= methods
    assert comp.row("ours_rk4_python").language == "python"
    assert comp.row("ours_rk4_rust").language == "rust"
    assert comp.row("ours_rk4_python").r_error_vs_reference is not None
    assert comp.metadata["competitor_count"] == len(comp.rows)
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


def test_run_comparison_default_problem_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_external(monkeypatch)
    comp = m.run_kuramoto_competitive_comparison(timeout=5.0, clock=lambda: "2026-07-03T00:00:00Z")
    assert comp.n_oscillators == build_default_problem().n_oscillators


def test_run_comparison_falls_back_to_dopri_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_external(monkeypatch, scipy_available=False)
    comp = m.run_kuramoto_competitive_comparison(
        _small_problem(), timeout=5.0, clock=lambda: "2026-07-03T00:00:00Z"
    )
    assert comp.reference_method == "ours_dopri"
    assert comp.row("scipy_solve_ivp").available is False
    assert comp.row("ours_rk4_python").r_error_vs_reference is not None
