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

from scpn_quantum_control.benchmarks.classical_baselines import (
    ClassicalBaselineRun,
    available_baselines,
    mps_tebd_baseline,
    qutip_lindblad_baseline,
    run_documented_classical_baselines,
    scipy_ode_baseline,
)


def _chain_problem(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    K = np.zeros((n, n), dtype=float)
    for site in range(n - 1):
        K[site, site + 1] = 0.25
        K[site + 1, site] = 0.25
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


def test_available_baselines_reports_all_three_names():
    availability = available_baselines()
    assert set(availability) == {"scipy_ode", "qutip_lindblad", "mps_tebd"}
    assert availability["scipy_ode"] is True


def test_scipy_ode_baseline_returns_bounded_order_parameter():
    K, omega = _chain_problem()
    result = scipy_ode_baseline(K, omega, t_max=0.2, dt=0.1)

    assert isinstance(result, ClassicalBaselineRun)
    assert result.available
    assert result.name == "scipy_ode"
    assert result.times.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert result.order_parameter.shape == result.times.shape
    assert np.all((result.order_parameter >= 0.0) & (result.order_parameter <= 1.0))
    assert result.r_final == pytest.approx(float(result.order_parameter[-1]))


def test_scipy_ode_baseline_accepts_explicit_initial_phase():
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
def test_baseline_input_validation(K: np.ndarray, omega: np.ndarray, match: str):
    with pytest.raises(ValueError, match=match):
        scipy_ode_baseline(K, omega)


def test_baseline_time_grid_validation():
    K, omega = _chain_problem()
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        scipy_ode_baseline(K, omega, dt=0.0)
    with pytest.raises(ValueError, match="t_max must be finite and non-negative"):
        scipy_ode_baseline(K, omega, t_max=-0.1)


def test_qutip_lindblad_reports_unavailable_when_missing(monkeypatch):
    K, omega = _chain_problem()
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "qutip":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    result = qutip_lindblad_baseline(K, omega)

    assert not result.available
    assert result.unavailable_reason == "qutip missing"
    assert result.r_final is None


def test_qutip_lindblad_runs_when_installed():
    pytest.importorskip("qutip")
    K, omega = _chain_problem(n=2)

    result = qutip_lindblad_baseline(K, omega, gamma=0.02, t_max=0.1, dt=0.1)

    assert result.available
    assert result.backend == "qutip.mesolve"
    assert result.metadata["final_trace"] == pytest.approx(1.0)
    assert np.all((result.order_parameter >= 0.0) & (result.order_parameter <= 1.01))


def test_qutip_lindblad_validates_gamma():
    K, omega = _chain_problem()
    with pytest.raises(ValueError, match="gamma must be finite and non-negative"):
        qutip_lindblad_baseline(K, omega, gamma=-0.1)


def test_mps_tebd_reports_unavailable_when_quimb_missing(monkeypatch):
    K, omega = _chain_problem()
    import scpn_quantum_control.benchmarks.classical_baselines as baselines

    monkeypatch.setattr(baselines.mps_evolution, "is_quimb_available", lambda: False)

    result = mps_tebd_baseline(K, omega)

    assert not result.available
    assert result.unavailable_reason == "quimb missing"


def test_mps_tebd_runs_when_quimb_installed():
    pytest.importorskip("quimb")
    K, omega = _chain_problem(n=4)

    result = mps_tebd_baseline(K, omega, t_max=0.1, dt=0.1, bond_dim=8)

    assert result.available
    assert result.backend == "quimb.TEBD"
    assert len(result.metadata["bond_dims_final"]) == 3
    assert np.all((result.order_parameter >= 0.0) & (result.order_parameter <= 1.01))


def test_mps_tebd_validates_parameters():
    K, omega = _chain_problem()
    with pytest.raises(ValueError, match="bond_dim must be positive"):
        mps_tebd_baseline(K, omega, bond_dim=0)
    with pytest.raises(ValueError, match="cutoff must be finite and positive"):
        mps_tebd_baseline(K, omega, cutoff=0.0)


def test_documented_baseline_suite_runs_scipy_only():
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


def test_documented_baseline_suite_includes_optional_statuses(monkeypatch):
    K, omega = _chain_problem()
    import scpn_quantum_control.benchmarks.classical_baselines as baselines

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name == "qutip" else object()
    )
    monkeypatch.setattr(baselines.mps_evolution, "is_quimb_available", lambda: False)

    results = run_documented_classical_baselines(K, omega, t_max=0.1, dt=0.1)

    assert set(results) == {"scipy_ode", "qutip_lindblad", "mps_tebd"}
    assert results["scipy_ode"].available
    assert not results["qutip_lindblad"].available
    assert not results["mps_tebd"].available
