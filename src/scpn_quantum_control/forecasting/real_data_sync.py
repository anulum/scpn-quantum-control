# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real-Data Synchronisation Forecasting
"""Held-out synchronisation forecasting on observed or source-backed traces."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control._paths import project_data_root
from scpn_quantum_control.applications.power_grid import ieee_5bus_coupling_matrix
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.classical import classical_kuramoto_reference

FloatArray = NDArray[np.float64]

_REPO_ROOT = project_data_root("results/hw_kuramoto_4osc.json")
_DEFAULT_HARDWARE_TRACE = _REPO_ROOT / "results" / "hw_kuramoto_4osc.json"


@dataclass(frozen=True)
class SynchronisationForecastDataset:
    """Observed synchronisation trace plus physical coupling provenance."""

    name: str
    domain: str
    source_path: str
    times: FloatArray
    observed_order_parameter: FloatArray
    coupling: FloatArray
    omega: FloatArray
    theta0: FloatArray
    train_size: int
    source_kind: str
    baseline_order_parameter: FloatArray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_oscillators(self) -> int:
        """Number of oscillators represented by the coupling matrix."""
        return int(self.coupling.shape[0])

    @property
    def holdout_size(self) -> int:
        """Number of synchronisation samples withheld from calibration."""
        return int(self.times.size - self.train_size)


@dataclass(frozen=True)
class ForecastModelRun:
    """Forecast metrics for one model on a held-out synchronisation window."""

    name: str
    backend: str
    predictions: FloatArray
    elapsed_ms: float
    train_mse: float
    holdout_mse: float
    holdout_mae: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SynchronisationForecastBenchmarkResult:
    """Baseline versus calibrated forecast on one dataset."""

    dataset: SynchronisationForecastDataset
    baseline: ForecastModelRun
    calibrated: ForecastModelRun
    improvement_fraction: float
    min_improvement_fraction: float
    passes: bool
    summary: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the benchmark result without NumPy objects."""
        return {
            "dataset": {
                "name": self.dataset.name,
                "domain": self.dataset.domain,
                "source_path": self.dataset.source_path,
                "source_kind": self.dataset.source_kind,
                "n_oscillators": self.dataset.n_oscillators,
                "train_size": self.dataset.train_size,
                "holdout_size": self.dataset.holdout_size,
            },
            "baseline": _model_to_dict(self.baseline),
            "calibrated": _model_to_dict(self.calibrated),
            "improvement_fraction": self.improvement_fraction,
            "min_improvement_fraction": self.min_improvement_fraction,
            "passes": self.passes,
            "summary": self.summary,
            "provenance": self.provenance,
        }


def load_hardware_kuramoto_4osc_trace(
    path: str | Path | None = None,
    *,
    train_size: int = 2,
) -> SynchronisationForecastDataset:
    """Load the committed IBM Heron r2 four-oscillator synchronisation trace."""
    trace_path = Path(path) if path is not None else _DEFAULT_HARDWARE_TRACE
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = payload["time_steps"]
    times = np.asarray([step["t"] for step in steps], dtype=np.float64)
    observed = np.asarray([step["hw_R"] for step in steps], dtype=np.float64)
    exact = np.asarray([step["exact_R"] for step in steps], dtype=np.float64)
    coupling = np.asarray(build_knm_paper27(L=4), dtype=np.float64)
    omega = np.asarray(OMEGA_N_16[:4], dtype=np.float64)
    theta0 = np.mod(omega, 2.0 * np.pi)
    return _validate_dataset(
        SynchronisationForecastDataset(
            name="ibm_heron_r2_kuramoto_4osc",
            domain="hardware-kuramoto",
            source_path=_relative_source_path(trace_path),
            times=times,
            observed_order_parameter=observed,
            baseline_order_parameter=exact,
            coupling=coupling,
            omega=omega,
            theta0=theta0,
            train_size=train_size,
            source_kind="qpu_hardware_measurement",
            provenance={
                "backend": payload.get("backend"),
                "processor": payload.get("processor"),
                "job_id": payload.get("job_id"),
                "shots": payload.get("shots"),
                "timestamp": payload.get("timestamp"),
                "baseline": "stored exact_R from the same hardware campaign file",
            },
        )
    )


def load_ieee5bus_sync_forecast_case(
    *,
    train_size: int = 4,
    t_max: float = 0.6,
    dt: float = 0.1,
    disturbance_scale: float = 0.035,
) -> SynchronisationForecastDataset:
    """Build a held-out forecast case from the public IEEE 5-bus topology.

    The source data are the IEEE 5-bus susceptance, inertia, voltage, and
    frequency-deviation constants in :mod:`applications.power_grid`. The
    target trace is generated by a high-resolution classical replay of that
    source topology with a deterministic rotor-angle disturbance.
    """
    if disturbance_scale <= 0.0 or not np.isfinite(disturbance_scale):
        raise ValueError("disturbance_scale must be finite and positive")
    coupling, omega = ieee_5bus_coupling_matrix()
    theta0 = np.asarray([0.0, 0.03, -0.025, 0.04, -0.015], dtype=np.float64)
    observed = _scipy_reference_trace(
        coupling,
        omega,
        theta0=theta0 + disturbance_scale * np.arange(coupling.shape[0]),
        t_max=t_max,
        dt=dt,
    )
    baseline = _integrate_kuramoto_r(
        coupling,
        omega,
        theta0=theta0,
        t_max=t_max,
        dt=dt,
    )
    times = np.linspace(0.0, float(t_max), observed.size, dtype=np.float64)
    return _validate_dataset(
        SynchronisationForecastDataset(
            name="ieee5bus_frequency_disturbance",
            domain="power-grid",
            source_path="src/scpn_quantum_control/applications/power_grid.py",
            times=times,
            observed_order_parameter=observed,
            baseline_order_parameter=baseline.values,
            coupling=np.asarray(coupling, dtype=np.float64),
            omega=np.asarray(omega, dtype=np.float64),
            theta0=theta0,
            train_size=train_size,
            source_kind="public_topology_classical_replay",
            provenance={
                "source": "IEEE 5-bus public benchmark constants",
                "disturbance_scale_rad": disturbance_scale,
                "baseline_backend": baseline.backend,
                "reference_backend": "scipy.solve_ivp(DOP853)",
            },
        )
    )


def run_real_data_sync_forecast_benchmark(
    dataset: SynchronisationForecastDataset | None = None,
    *,
    min_improvement_fraction: float = 0.10,
) -> SynchronisationForecastBenchmarkResult:
    """Run train-window calibration and held-out synchronisation forecasting."""
    if min_improvement_fraction < 0.0 or not np.isfinite(min_improvement_fraction):
        raise ValueError("min_improvement_fraction must be finite and non-negative")
    case = _validate_dataset(dataset or load_hardware_kuramoto_4osc_trace())
    observed = case.observed_order_parameter
    baseline_values, baseline_backend = _baseline_values(case)

    start = time.perf_counter()
    baseline_train_mse, baseline_holdout_mse, baseline_holdout_mae = _split_metrics(
        observed,
        baseline_values,
        case.train_size,
    )
    baseline_elapsed_ms = (time.perf_counter() - start) * 1000.0
    baseline = ForecastModelRun(
        name="physical_baseline",
        backend=baseline_backend,
        predictions=baseline_values,
        elapsed_ms=baseline_elapsed_ms,
        train_mse=baseline_train_mse,
        holdout_mse=baseline_holdout_mse,
        holdout_mae=baseline_holdout_mae,
        metadata={"uses_training_observations": False},
    )

    start = time.perf_counter()
    scale, offset = _fit_train_window_affine(
        baseline_values[: case.train_size],
        observed[: case.train_size],
    )
    calibrated_values = np.clip(scale * baseline_values + offset, 0.0, 1.0)
    calibrated_train_mse, calibrated_holdout_mse, calibrated_holdout_mae = _split_metrics(
        observed,
        calibrated_values,
        case.train_size,
    )
    calibrated_elapsed_ms = (time.perf_counter() - start) * 1000.0
    calibrated = ForecastModelRun(
        name="train_window_affine_sync_calibration",
        backend="numpy.linalg.lstsq",
        predictions=calibrated_values,
        elapsed_ms=calibrated_elapsed_ms,
        train_mse=calibrated_train_mse,
        holdout_mse=calibrated_holdout_mse,
        holdout_mae=calibrated_holdout_mae,
        metadata={
            "scale": scale,
            "offset": offset,
            "uses_training_observations": True,
            "holdout_values_visible_to_fit": False,
        },
    )

    improvement = _improvement_fraction(baseline.holdout_mse, calibrated.holdout_mse)
    passes = bool(improvement >= min_improvement_fraction)
    summary = (
        f"{case.name}: baseline MSE={baseline.holdout_mse:.6f}, "
        f"calibrated MSE={calibrated.holdout_mse:.6f}, "
        f"improvement={100.0 * improvement:.1f}%"
    )
    return SynchronisationForecastBenchmarkResult(
        dataset=case,
        baseline=baseline,
        calibrated=calibrated,
        improvement_fraction=improvement,
        min_improvement_fraction=min_improvement_fraction,
        passes=passes,
        summary=summary,
        provenance={
            "fit_window": [0, case.train_size],
            "holdout_window": [case.train_size, int(case.times.size)],
            "metric": "mean squared error on Kuramoto order parameter R(t)",
        },
    )


def run_real_data_sync_forecast_suite(
    *,
    include_topology_replay: bool = True,
    min_improvement_fraction: float = 0.10,
) -> list[SynchronisationForecastBenchmarkResult]:
    """Run the real-data synchronisation forecasting benchmark suite."""
    datasets = [load_hardware_kuramoto_4osc_trace()]
    if include_topology_replay:
        datasets.append(load_ieee5bus_sync_forecast_case())
    return [
        run_real_data_sync_forecast_benchmark(
            dataset,
            min_improvement_fraction=min_improvement_fraction,
        )
        for dataset in datasets
    ]


@dataclass(frozen=True)
class _TrajectoryResult:
    values: FloatArray
    backend: str


def _integrate_kuramoto_r(
    coupling: FloatArray,
    omega: FloatArray,
    *,
    theta0: FloatArray,
    t_max: float,
    dt: float,
) -> _TrajectoryResult:
    n_steps = _n_steps(t_max, dt)
    try:
        import scpn_quantum_engine as engine

        _times, values = engine.kuramoto_trajectory(theta0, omega, coupling, dt, n_steps)
        return _TrajectoryResult(np.asarray(values, dtype=np.float64), "rust:kuramoto_trajectory")
    except (ImportError, AttributeError):
        result = classical_kuramoto_reference(
            coupling.shape[0],
            t_max=t_max,
            dt=dt,
            K=coupling,
            omega=omega,
            theta0=theta0,
        )
        return _TrajectoryResult(np.asarray(result["R"], dtype=np.float64), "python:euler")


def _scipy_reference_trace(
    coupling: FloatArray,
    omega: FloatArray,
    *,
    theta0: FloatArray,
    t_max: float,
    dt: float,
) -> FloatArray:
    from scipy.integrate import solve_ivp

    n_steps = _n_steps(t_max, dt)
    times = np.linspace(0.0, float(t_max), n_steps + 1, dtype=np.float64)

    def rhs(_t: float, theta: FloatArray) -> FloatArray:
        phase_delta = theta[None, :] - theta[:, None]
        coupling_term = np.sum(coupling * np.sin(phase_delta), axis=1)
        return np.asarray(omega + coupling_term, dtype=np.float64)

    solution = solve_ivp(
        rhs,
        (0.0, float(t_max)),
        theta0,
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"reference replay failed: {solution.message}")
    return np.asarray([_order_parameter(row) for row in solution.y.T], dtype=np.float64)


def _baseline_values(case: SynchronisationForecastDataset) -> tuple[FloatArray, str]:
    if case.baseline_order_parameter is not None:
        backend = str(case.provenance.get("baseline_backend", "dataset:baseline_R"))
        if backend != "dataset:baseline_R":
            backend = f"dataset:baseline_R:{backend}"
        return np.asarray(case.baseline_order_parameter, dtype=np.float64), backend
    if case.times.size < 2:
        raise ValueError("at least two samples are required to infer dt")
    dt_values = np.diff(case.times)
    if not np.allclose(dt_values, dt_values[0], rtol=1e-10, atol=1e-12):
        raise ValueError("implicit Kuramoto baseline requires an evenly spaced time grid")
    trajectory = _integrate_kuramoto_r(
        case.coupling,
        case.omega,
        theta0=case.theta0,
        t_max=float(case.times[-1] - case.times[0]),
        dt=float(dt_values[0]),
    )
    if trajectory.values.size == case.times.size + 1 and not np.isclose(case.times[0], 0.0):
        return trajectory.values[1:], trajectory.backend
    return trajectory.values, trajectory.backend


def _fit_train_window_affine(x: FloatArray, y: FloatArray) -> tuple[float, float]:
    if x.size != y.size or x.size == 0:
        raise ValueError("training arrays must be non-empty and equal length")
    if x.size == 1 or float(np.ptp(x)) < 1e-14:
        return 1.0, float(np.mean(y - x))
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _split_metrics(
    observed: FloatArray,
    predicted: FloatArray,
    train_size: int,
) -> tuple[float, float, float]:
    residual = np.asarray(predicted - observed, dtype=np.float64)
    train_residual = residual[:train_size]
    holdout_residual = residual[train_size:]
    return (
        float(np.mean(train_residual**2)),
        float(np.mean(holdout_residual**2)),
        float(np.mean(np.abs(holdout_residual))),
    )


def _improvement_fraction(baseline_mse: float, candidate_mse: float) -> float:
    if baseline_mse <= 1e-15:
        return 0.0 if candidate_mse <= baseline_mse else -float("inf")
    return float((baseline_mse - candidate_mse) / baseline_mse)


def _validate_dataset(dataset: SynchronisationForecastDataset) -> SynchronisationForecastDataset:
    times = np.asarray(dataset.times, dtype=np.float64)
    observed = np.asarray(dataset.observed_order_parameter, dtype=np.float64)
    coupling = np.asarray(dataset.coupling, dtype=np.float64)
    omega = np.asarray(dataset.omega, dtype=np.float64)
    theta0 = np.asarray(dataset.theta0, dtype=np.float64)
    if times.ndim != 1 or observed.shape != times.shape:
        raise ValueError("times and observed_order_parameter must be equal-length vectors")
    if times.size < 3:
        raise ValueError("at least three time samples are required for train and holdout")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    if not np.all((observed >= 0.0) & (observed <= 1.0)):
        raise ValueError("observed order parameters must lie in [0, 1]")
    if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
        raise ValueError("coupling must be a square matrix")
    if omega.shape != (coupling.shape[0],) or theta0.shape != omega.shape:
        raise ValueError("omega and theta0 must match coupling dimension")
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(omega)):
        raise ValueError("coupling and omega must contain only finite values")
    if dataset.baseline_order_parameter is not None:
        baseline = np.asarray(dataset.baseline_order_parameter, dtype=np.float64)
        if baseline.shape != observed.shape:
            raise ValueError("baseline_order_parameter must match observed trace shape")
        if not np.all((baseline >= 0.0) & (baseline <= 1.0)):
            raise ValueError("baseline order parameters must lie in [0, 1]")
    if dataset.train_size < 1 or dataset.train_size >= times.size:
        raise ValueError("train_size must leave at least one held-out sample")
    return dataset


def _n_steps(t_max: float, dt: float) -> int:
    if t_max <= 0.0 or not np.isfinite(t_max):
        raise ValueError("t_max must be finite and positive")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    return max(1, round(float(t_max) / float(dt)))


def _order_parameter(theta: FloatArray) -> float:
    return float(abs(np.mean(np.exp(1j * theta))))


def _relative_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _model_to_dict(model: ForecastModelRun) -> dict[str, Any]:
    return {
        "name": model.name,
        "backend": model.backend,
        "predictions": model.predictions.tolist(),
        "elapsed_ms": model.elapsed_ms,
        "train_mse": model.train_mse,
        "holdout_mse": model.holdout_mse,
        "holdout_mae": model.holdout_mae,
        "metadata": model.metadata,
    }
