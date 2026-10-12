# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Classical Baselines
"""Tests for documented classical baselines."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks.classical_baselines import (
    ClassicalBaselineRun,
    available_baselines,
    mps_tebd_baseline,
    qutip_lindblad_baseline,
    run_documented_classical_baselines,
    scipy_ode_baseline,
)
from scpn_quantum_control.phase import mps_evolution


def _chain_problem(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build a finite nearest-neighbour test model in site order."""
    K = np.zeros((n, n), dtype=float)
    for site in range(n - 1):
        K[site, site + 1] = 0.25
        K[site + 1, site] = 0.25
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


def test_available_baselines_reports_all_three_names() -> None:
    """List actual runtime availability for the documented baseline families."""
    availability = available_baselines()
    assert set(availability) == {"scipy_ode", "qutip_lindblad", "mps_tebd"}
    assert availability["scipy_ode"] is True


def test_scipy_ode_baseline_returns_bounded_order_parameter() -> None:
    """Exercise the public ODE path and its returned observation grid."""
    K, omega = _chain_problem()
    result = scipy_ode_baseline(K, omega, t_max=0.2, dt=0.1)

    assert isinstance(result, ClassicalBaselineRun)
    assert result.available
    assert result.name == "scipy_ode"
    assert result.times.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert result.order_parameter.shape == result.times.shape
    assert np.all((result.order_parameter >= 0.0) & (result.order_parameter <= 1.0))
    assert result.r_final == pytest.approx(float(result.order_parameter[-1]))


def test_scipy_ode_baseline_accepts_explicit_initial_phase() -> None:
    """Preserve the caller-selected phase preparation in the actual ODE run."""
    K, omega = _chain_problem()
    theta0 = np.array([0.0, 0.2, 0.4])
    result = scipy_ode_baseline(K, omega, t_max=0.1, dt=0.1, theta0=theta0)

    assert result.metadata["theta_final"] != []
    assert result.order_parameter[0] == pytest.approx(abs(np.mean(np.exp(1j * theta0))))


@pytest.mark.parametrize(
    ("K", "omega", "match"),
    [
        (np.ones((2, 3)), np.ones(2), "K must be a square matrix"),
        (np.eye(3), np.ones(2), "omega must have shape"),
        (np.array([[0.0, np.nan], [0.0, 0.0]]), np.ones(2), "K must contain"),
        (np.eye(2), np.array([1.0, np.inf]), "omega must contain"),
    ],
)
def test_baseline_input_validation(
    K: NDArray[np.float64], omega: NDArray[np.float64], match: str
) -> None:
    """Refuse malformed model arrays before starting the classical solver."""
    with pytest.raises(ValueError, match=match):
        scipy_ode_baseline(K, omega)


def test_baseline_time_grid_validation() -> None:
    """Reject invalid observation times before integrating the model."""
    K, omega = _chain_problem()
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        scipy_ode_baseline(K, omega, dt=0.0)
    with pytest.raises(ValueError, match="t_max must be finite and non-negative"):
        scipy_ode_baseline(K, omega, t_max=-0.1)


def test_qutip_lindblad_reports_unavailable_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report an absent optional solver without a fabricated trajectory."""
    K, omega = _chain_problem()
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str) -> object | None:
        """Inject only the optional dependency absence boundary."""
        if name == "qutip":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    result = qutip_lindblad_baseline(K, omega)

    assert not result.available
    assert result.unavailable_reason == "qutip missing"
    assert result.r_final is None


def test_qutip_lindblad_runs_when_installed() -> None:
    """Exercise the actual QuTiP density route and retain its state trace."""
    pytest.importorskip("qutip")
    K, omega = _chain_problem(n=2)

    result = qutip_lindblad_baseline(K, omega, gamma=0.02, t_max=0.1, dt=0.1)

    assert result.available
    assert result.backend == "qutip.mesolve"
    assert result.metadata["final_trace"] == pytest.approx(1.0)
    assert np.all((result.order_parameter >= 0.0) & (result.order_parameter <= 1.01))


def test_qutip_lindblad_validates_gamma() -> None:
    """Reject a negative physical damping rate at the public boundary."""
    K, omega = _chain_problem()
    with pytest.raises(ValueError, match="gamma must be finite and non-negative"):
        qutip_lindblad_baseline(K, omega, gamma=-0.1)


def test_mps_tebd_reports_unavailable_when_quimb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report unavailable Quimb without executing another numerical route."""
    K, omega = _chain_problem()

    monkeypatch.setattr(mps_evolution, "is_quimb_available", lambda: False)

    result = mps_tebd_baseline(K, omega)

    assert not result.available
    assert result.unavailable_reason == "quimb missing"


def test_mps_tebd_runs_when_quimb_installed() -> None:
    """Retain real MPS observations and memory admission through the wrapper."""
    pytest.importorskip("quimb")
    K, omega = _chain_problem(n=4)

    result = mps_tebd_baseline(
        K, omega, t_max=0.1, dt=0.1, bond_dim=8, max_memory_bytes=1024 * 1024
    )

    assert result.available
    assert result.backend == "quimb.TEBD"
    assert len(result.metadata["bond_dims_final"]) == 3
    assert np.all((result.order_parameter >= 0.0) & (result.order_parameter <= 1.01))
    diagnostics = result.metadata["mps_diagnostics"]
    assert diagnostics["allocation"]["budget_bytes"] <= 1024 * 1024
    assert len(diagnostics["observations"]) == len(result.times)
    assert "not a global" in diagnostics["discarded_weight_scope"]


def test_mps_tebd_validates_parameters() -> None:
    """Refuse invalid bond and cutoff parameters before optional execution."""
    K, omega = _chain_problem()
    with pytest.raises(ValueError, match="bond_dim must be positive"):
        mps_tebd_baseline(K, omega, bond_dim=0)
    with pytest.raises(ValueError, match="cutoff must be finite and positive"):
        mps_tebd_baseline(K, omega, cutoff=0.0)


def test_documented_baseline_suite_runs_scipy_only() -> None:
    """Honor explicit ODE-only execution through the original suite facade."""
    K, omega = _chain_problem()

    results = run_documented_classical_baselines(
        K,
        omega,
        t_max=0.1,
        dt=0.1,
        include_optional=False,
    )

    assert set(results) == {"scipy_ode"}
    assert results["scipy_ode"].available


def test_documented_baseline_suite_includes_optional_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep absent optional routes labelled unavailable alongside actual ODE output."""
    K, omega = _chain_problem()

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "qutip" else object()
    )
    monkeypatch.setattr(mps_evolution, "is_quimb_available", lambda: False)

    results = run_documented_classical_baselines(K, omega, t_max=0.1, dt=0.1)

    assert set(results) == {"scipy_ode", "qutip_lindblad", "mps_tebd"}
    assert results["scipy_ode"].available
    assert not results["qutip_lindblad"].available
    assert not results["mps_tebd"].available


def test_qutip_baseline_refuses_exponential_workspace_before_runtime() -> None:
    """Reject a non-addressable density workspace before optional imports."""
    with pytest.raises(MemoryError, match="native addressable memory"):
        qutip_lindblad_baseline(np.zeros((64, 64)), np.zeros(64), t_max=0.1, dt=0.05)


def test_qutip_pure_state_route_and_dense_damping_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit a bounded pure route while refusing a larger dense Liouvillian.

    Parameters
    ----------
    monkeypatch
        Scoped process allocation policy override, restored after the test.

    """
    monkeypatch.setenv("SCPN_MAX_DENSE_GIB", str(1 / 1024))
    coupling, frequencies = np.zeros((4, 4)), np.zeros(4)
    with pytest.raises(MemoryError, match="active dense budget"):
        qutip_lindblad_baseline(coupling, frequencies, gamma=0.05, t_max=0.1, dt=0.05)
    result = qutip_lindblad_baseline(coupling, frequencies, gamma=0.0, t_max=0.1, dt=0.05)
    assert result.available
    assert result.metadata["final_trace"] == pytest.approx(1.0, abs=1e-12)
    assert result.r_final == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "coupling,frequencies",
    [
        (np.array([[0.0, 0.5 + 1j], [0.5 + 1j, 0.0]]), np.array([0.1, 0.2])),
        (np.array([[0.0, 0.5], [0.5, 0.0]]), np.array([0.1 + 1j, 0.2])),
    ],
)
def test_baseline_refuses_silent_complex_to_real_conversion(
    coupling: NDArray[np.float64],
    frequencies: NDArray[np.float64],
) -> None:
    """Reject complex model data before its imaginary components are discarded.

    Parameters
    ----------
    coupling
        Deliberately invalid complex coupling or a valid companion array.
    frequencies
        Deliberately invalid complex frequency or a valid companion array.

    """
    before = (coupling.tobytes(), frequencies.tobytes())
    with pytest.raises(ValueError, match="real numeric"):
        scipy_ode_baseline(coupling, frequencies, t_max=0.1, dt=0.05)
    assert (coupling.tobytes(), frequencies.tobytes()) == before
