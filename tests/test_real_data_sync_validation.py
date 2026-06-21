# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation and fallback tests for real-data sync forecasting
"""Validation, guard, and backend-fallback tests for real-data sync forecasting.

Covers the dataset validation guards, the train-window affine fit edge cases,
the improvement-fraction degenerate baseline, the step-count and source-path
helpers, the implicit-baseline inference branches, and the native/scipy
trajectory backends including their Python fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.forecasting import real_data_sync
from scpn_quantum_control.forecasting.real_data_sync import (
    SynchronisationForecastDataset,
    _baseline_values,
    _fit_train_window_affine,
    _improvement_fraction,
    _integrate_kuramoto_r,
    _n_steps,
    _relative_source_path,
    _scipy_reference_trace,
    _TrajectoryResult,
    _validate_dataset,
    load_ieee5bus_sync_forecast_case,
    run_real_data_sync_forecast_benchmark,
)

FloatArray = NDArray[np.float64]


def _dataset(**overrides: Any) -> SynchronisationForecastDataset:
    """Build a valid synchronisation-forecast dataset, applying field overrides."""
    base: dict[str, Any] = {
        "name": "unit",
        "domain": "test",
        "source_path": "src/scpn_quantum_control/forecasting/real_data_sync.py",
        "times": np.linspace(0.0, 1.0, 5, dtype=np.float64),
        "observed_order_parameter": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        "coupling": np.eye(2, dtype=np.float64),
        "omega": np.array([1.0, 2.0], dtype=np.float64),
        "theta0": np.array([0.0, 0.0], dtype=np.float64),
        "train_size": 3,
        "source_kind": "unit",
    }
    base.update(overrides)
    return SynchronisationForecastDataset(**base)


def test_validate_rejects_observed_shape_mismatch() -> None:
    """Observed trace length must match the time grid."""
    with pytest.raises(ValueError, match="equal-length vectors"):
        _validate_dataset(
            _dataset(observed_order_parameter=np.array([0.1, 0.2], dtype=np.float64))
        )


def test_validate_rejects_non_increasing_times() -> None:
    """Times must be strictly increasing."""
    bad = np.array([0.0, 0.5, 0.5, 0.75, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_dataset(_dataset(times=bad))


def test_validate_rejects_non_square_coupling() -> None:
    """The coupling matrix must be square."""
    with pytest.raises(ValueError, match="coupling must be a square matrix"):
        _validate_dataset(_dataset(coupling=np.zeros((2, 3), dtype=np.float64)))


def test_validate_rejects_dimension_mismatch() -> None:
    """omega and theta0 must match the coupling dimension."""
    with pytest.raises(ValueError, match="must match coupling dimension"):
        _validate_dataset(_dataset(omega=np.array([1.0, 2.0, 3.0], dtype=np.float64)))


def test_validate_rejects_non_finite_coupling() -> None:
    """Coupling and omega must be finite."""
    bad = np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite values"):
        _validate_dataset(_dataset(coupling=bad))


def test_validate_rejects_baseline_shape_mismatch() -> None:
    """A supplied baseline trace must match the observed trace shape."""
    with pytest.raises(ValueError, match="baseline_order_parameter must match"):
        _validate_dataset(
            _dataset(baseline_order_parameter=np.array([0.1, 0.2], dtype=np.float64))
        )


def test_validate_rejects_baseline_out_of_range() -> None:
    """Baseline order parameters must lie in [0, 1]."""
    bad = np.array([0.1, 0.2, 0.3, 0.4, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="baseline order parameters must lie"):
        _validate_dataset(_dataset(baseline_order_parameter=bad))


def test_fit_train_window_rejects_mismatched_arrays() -> None:
    """The affine fit requires non-empty, equal-length arrays."""
    with pytest.raises(ValueError, match="non-empty and equal length"):
        _fit_train_window_affine(
            np.array([1.0, 2.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
        )


def test_fit_train_window_constant_x_is_identity_shift() -> None:
    """A constant abscissa collapses the fit to a unit-slope mean shift."""
    x = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    y = np.array([6.0, 7.0, 8.0], dtype=np.float64)
    slope, intercept = _fit_train_window_affine(x, y)
    assert slope == 1.0
    assert intercept == pytest.approx(2.0)


def test_improvement_fraction_degenerate_baseline() -> None:
    """A near-zero baseline MSE yields a clamped or minus-infinite improvement."""
    assert _improvement_fraction(0.0, 0.0) == 0.0
    assert _improvement_fraction(0.0, 1.0) == -float("inf")


def test_n_steps_rejects_non_positive_t_max() -> None:
    """The step count requires a finite positive horizon."""
    with pytest.raises(ValueError, match="t_max must be finite and positive"):
        _n_steps(0.0, 0.1)


def test_n_steps_rejects_non_positive_dt() -> None:
    """The step count requires a finite positive time step."""
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        _n_steps(1.0, 0.0)


def test_relative_source_path_outside_repo_is_passthrough() -> None:
    """A path outside the repository root is returned verbatim as POSIX."""
    outside = Path("/nonexistent/outside/source.py")
    assert _relative_source_path(outside) == "/nonexistent/outside/source.py"


def test_ieee5bus_case_rejects_non_positive_disturbance() -> None:
    """The IEEE 5-bus case requires a finite positive disturbance scale."""
    with pytest.raises(ValueError, match="disturbance_scale must be finite and positive"):
        load_ieee5bus_sync_forecast_case(disturbance_scale=-1.0)


def test_benchmark_rejects_negative_min_improvement() -> None:
    """The benchmark requires a finite non-negative improvement threshold."""
    with pytest.raises(ValueError, match="min_improvement_fraction must be finite"):
        run_real_data_sync_forecast_benchmark(min_improvement_fraction=-1.0)


def test_baseline_values_requires_two_samples_to_infer_dt() -> None:
    """Inferring an implicit baseline needs at least two time samples."""
    case = _dataset(
        times=np.array([0.0], dtype=np.float64),
        observed_order_parameter=np.array([0.1], dtype=np.float64),
        baseline_order_parameter=None,
    )
    with pytest.raises(ValueError, match="at least two samples"):
        _baseline_values(case)


def test_baseline_values_requires_even_time_grid() -> None:
    """An implicit Kuramoto baseline needs an evenly spaced time grid."""
    case = _dataset(
        times=np.array([0.0, 1.0, 3.0], dtype=np.float64),
        observed_order_parameter=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        baseline_order_parameter=None,
    )
    with pytest.raises(ValueError, match="evenly spaced time grid"):
        _baseline_values(case)


def test_baseline_values_integrates_even_grid() -> None:
    """An evenly spaced grid without a baseline integrates the Kuramoto reference."""
    n = 4
    case = _dataset(
        times=np.linspace(0.0, 0.4, 5, dtype=np.float64),
        observed_order_parameter=np.linspace(0.2, 0.6, 5, dtype=np.float64),
        coupling=np.eye(n, dtype=np.float64),
        omega=np.zeros(n, dtype=np.float64),
        theta0=np.zeros(n, dtype=np.float64),
        baseline_order_parameter=None,
    )
    values, backend = _baseline_values(case)
    assert values.ndim == 1
    assert isinstance(backend, str)


def test_integrate_kuramoto_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the native trajectory kernel is unavailable the Euler path is used."""
    n = 4
    coupling = np.eye(n, dtype=np.float64)
    omega = np.zeros(n, dtype=np.float64)
    theta0 = np.zeros(n, dtype=np.float64)

    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "kuramoto_trajectory"):
            monkeypatch.delattr(engine, "kuramoto_trajectory")
    except ImportError:
        pass

    result = _integrate_kuramoto_r(coupling, omega, theta0=theta0, t_max=0.4, dt=0.1)
    assert result.backend == "python:euler"
    assert result.values.ndim == 1


def test_scipy_reference_trace_raises_on_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed ODE integration is surfaced as a runtime error."""

    class _FailedSolution:
        success = False
        message = "synthetic solver failure"

    def _failed_solve(*_args: object, **_kwargs: object) -> _FailedSolution:
        return _FailedSolution()

    monkeypatch.setattr("scipy.integrate.solve_ivp", _failed_solve)
    coupling = np.eye(2, dtype=np.float64)
    omega = np.zeros(2, dtype=np.float64)
    theta0 = np.zeros(2, dtype=np.float64)
    with pytest.raises(RuntimeError, match="reference replay failed"):
        _scipy_reference_trace(coupling, omega, theta0=theta0, t_max=0.4, dt=0.1)


def test_validate_requires_three_samples() -> None:
    """At least three time samples are required to split train and holdout."""
    case = _dataset(
        times=np.array([0.0, 1.0], dtype=np.float64),
        observed_order_parameter=np.array([0.1, 0.2], dtype=np.float64),
        train_size=1,
    )
    with pytest.raises(ValueError, match="at least three time samples"):
        _validate_dataset(case)


def test_baseline_values_trims_leading_sample_off_zero_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trajectory one sample longer than an off-zero grid drops its t=0 sample."""
    times = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    case = _dataset(
        times=times,
        observed_order_parameter=np.linspace(0.2, 0.5, 4, dtype=np.float64),
        coupling=np.eye(4, dtype=np.float64),
        omega=np.zeros(4, dtype=np.float64),
        theta0=np.zeros(4, dtype=np.float64),
        baseline_order_parameter=None,
    )
    trajectory = np.arange(times.size + 1, dtype=np.float64)

    def _fake_integrate(*_args: object, **_kwargs: object) -> _TrajectoryResult:
        return _TrajectoryResult(trajectory, "test:fixture")

    monkeypatch.setattr(real_data_sync, "_integrate_kuramoto_r", _fake_integrate)
    values, backend = _baseline_values(case)
    assert backend == "test:fixture"
    np.testing.assert_array_equal(values, trajectory[1:])
