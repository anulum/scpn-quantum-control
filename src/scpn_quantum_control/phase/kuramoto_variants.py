# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto variant trajectories
"""Higher-order, monitored, and PT-symmetric Kuramoto trajectories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
JsonScalar = str | int | float | bool | None


class KuramotoVariant(str, Enum):
    """Supported Kuramoto trajectory variants."""

    HIGHER_ORDER = "higher_order"
    MONITORED = "monitored"
    PT_SYMMETRIC = "pt_symmetric"


@dataclass(frozen=True)
class KuramotoVariantResult:
    """Trajectory and diagnostics for one Kuramoto variant run."""

    variant: KuramotoVariant
    times: FloatArray
    r_values: FloatArray
    backend: str
    diagnostics: Mapping[str, FloatArray | float | int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = _readonly_float_array(self.times, "times")
        r_values = _readonly_float_array(self.r_values, "r_values")
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if r_values.shape != times.shape:
            raise ValueError("r_values must have the same shape as times")
        if np.any(r_values < -1e-12) or np.any(r_values > 1.0 + 1e-12):
            raise ValueError("r_values must stay inside [0, 1]")

        diagnostics: dict[str, FloatArray | float | int | str | bool] = {}
        for key, value in self.diagnostics.items():
            if isinstance(value, np.ndarray):
                arr = _readonly_float_array(value, key)
                if arr.shape != times.shape:
                    raise ValueError(f"diagnostic {key!r} must match times shape")
                diagnostics[key] = arr
            elif isinstance(value, str | bool | int | float):
                diagnostics[key] = value
            else:
                raise TypeError(f"unsupported diagnostic value for {key!r}")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "r_values", r_values)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    @property
    def final_r(self) -> float:
        """Final Kuramoto order parameter."""
        return float(self.r_values[-1])

    @property
    def peak_r(self) -> float:
        """Peak Kuramoto order parameter along the trajectory."""
        return float(np.max(self.r_values))

    def to_metadata(self) -> dict[str, Any]:
        """Return a serialisable trajectory summary."""
        return {
            "variant": self.variant.value,
            "backend": self.backend,
            "n_steps": int(self.times.size - 1),
            "final_r": self.final_r,
            "peak_r": self.peak_r,
            "diagnostics": {
                key: _serialise_diagnostic(value) for key, value in self.diagnostics.items()
            },
        }


@dataclass(frozen=True)
class HigherOrderKuramotoSpec:
    """Pairwise Kuramoto system plus anchored triadic simplicial couplings."""

    K_nm: FloatArray
    omega: FloatArray
    hyperedges: IntArray
    hyper_weights: FloatArray
    theta0: FloatArray | None = None
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K_nm, omega = validate_variant_kuramoto_inputs(self.K_nm, self.omega)
        hyperedges = _validate_hyperedges(self.hyperedges, omega.size)
        hyper_weights = _readonly_float_array(self.hyper_weights, "hyper_weights")
        if hyper_weights.shape != (hyperedges.shape[0],):
            raise ValueError(
                f"hyper_weights must have shape ({hyperedges.shape[0]},), "
                f"got {hyper_weights.shape}"
            )
        theta0 = _prepare_theta0(self.theta0, omega)
        metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "K_nm", K_nm)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "hyperedges", hyperedges)
        object.__setattr__(self, "hyper_weights", hyper_weights)
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class MonitoredKuramotoSpec:
    """Kuramoto trajectory with deterministic measurement-feedback closure."""

    K_nm: FloatArray
    omega: FloatArray
    target_r: float = 0.75
    monitor_gain: float = 0.8
    measurement_strength: float = 0.2
    theta0: FloatArray | None = None
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K_nm, omega = validate_variant_kuramoto_inputs(self.K_nm, self.omega)
        _require_range(self.target_r, 0.0, 1.0, "target_r")
        _require_non_negative(self.monitor_gain, "monitor_gain")
        _require_range(self.measurement_strength, 0.0, 1.0, "measurement_strength")
        theta0 = _prepare_theta0(self.theta0, omega)
        metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "K_nm", K_nm)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class PTSymmetricKuramotoSpec:
    """Complex Kuramoto oscillator system with balanced gain/loss channels."""

    K_nm: FloatArray
    omega: FloatArray
    gain_loss: FloatArray
    theta0: FloatArray | None = None
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K_nm, omega = validate_variant_kuramoto_inputs(self.K_nm, self.omega)
        gain_loss = _readonly_float_array(self.gain_loss, "gain_loss")
        if gain_loss.shape != omega.shape:
            raise ValueError(f"gain_loss must have shape {omega.shape}, got {gain_loss.shape}")
        if abs(float(np.sum(gain_loss))) > 1e-10:
            raise ValueError("gain_loss must sum to zero for balanced PT symmetry")
        theta0 = _prepare_theta0(self.theta0, omega)
        metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "K_nm", K_nm)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "gain_loss", gain_loss)
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


def build_triadic_ring_terms(n_oscillators: int, weight: float) -> tuple[IntArray, FloatArray]:
    """Build anchored nearest-neighbour triadic terms on a periodic ring."""
    if not isinstance(n_oscillators, int) or n_oscillators < 3:
        raise ValueError("n_oscillators must be an integer at least 3")
    if not np.isfinite(weight):
        raise ValueError("weight must be finite")
    hyperedges = np.array(
        [(i, (i - 1) % n_oscillators, (i + 1) % n_oscillators) for i in range(n_oscillators)],
        dtype=np.int64,
    )
    weights = np.full(n_oscillators, float(weight), dtype=np.float64)
    hyperedges.setflags(write=False)
    weights.setflags(write=False)
    return hyperedges, weights


def simulate_higher_order_kuramoto(
    spec: HigherOrderKuramotoSpec,
    *,
    dt: float,
    n_steps: int,
    prefer_rust: bool = True,
) -> KuramotoVariantResult:
    """Simulate pairwise plus anchored triadic Kuramoto dynamics."""
    _validate_time_grid(dt, n_steps)
    if prefer_rust:
        try:
            import scpn_quantum_engine as _engine

            times, r_values = _engine.higher_order_kuramoto_trajectory(
                _required_theta0(spec.theta0),
                spec.omega,
                spec.K_nm,
                spec.hyperedges,
                spec.hyper_weights,
                dt,
                n_steps,
            )
            return KuramotoVariantResult(
                variant=KuramotoVariant.HIGHER_ORDER,
                times=np.asarray(times, dtype=np.float64),
                r_values=np.asarray(r_values, dtype=np.float64),
                backend="rust:higher_order_kuramoto_trajectory",
                diagnostics={"n_hyperedges": int(spec.hyperedges.shape[0])},
            )
        except (ImportError, AttributeError):
            pass

    times, r_values = _higher_order_numpy(spec, dt=dt, n_steps=n_steps)
    return KuramotoVariantResult(
        variant=KuramotoVariant.HIGHER_ORDER,
        times=times,
        r_values=r_values,
        backend="numpy:higher_order_kuramoto_trajectory",
        diagnostics={"n_hyperedges": int(spec.hyperedges.shape[0])},
    )


def simulate_monitored_kuramoto(
    spec: MonitoredKuramotoSpec,
    *,
    dt: float,
    n_steps: int,
    prefer_rust: bool = True,
) -> KuramotoVariantResult:
    """Simulate monitored Kuramoto dynamics with order-parameter feedback."""
    _validate_time_grid(dt, n_steps)
    if prefer_rust:
        try:
            import scpn_quantum_engine as _engine

            times, r_values, readouts, feedback = _engine.monitored_kuramoto_trajectory(
                _required_theta0(spec.theta0),
                spec.omega,
                spec.K_nm,
                spec.target_r,
                spec.monitor_gain,
                spec.measurement_strength,
                dt,
                n_steps,
            )
            return KuramotoVariantResult(
                variant=KuramotoVariant.MONITORED,
                times=np.asarray(times, dtype=np.float64),
                r_values=np.asarray(r_values, dtype=np.float64),
                backend="rust:monitored_kuramoto_trajectory",
                diagnostics={
                    "readout_r": np.asarray(readouts, dtype=np.float64),
                    "feedback": np.asarray(feedback, dtype=np.float64),
                    "target_r": float(spec.target_r),
                },
            )
        except (ImportError, AttributeError):
            pass

    times, r_values, readouts, feedback = _monitored_numpy(spec, dt=dt, n_steps=n_steps)
    return KuramotoVariantResult(
        variant=KuramotoVariant.MONITORED,
        times=times,
        r_values=r_values,
        backend="numpy:monitored_kuramoto_trajectory",
        diagnostics={
            "readout_r": readouts,
            "feedback": feedback,
            "target_r": float(spec.target_r),
        },
    )


def simulate_pt_symmetric_kuramoto(
    spec: PTSymmetricKuramotoSpec,
    *,
    dt: float,
    n_steps: int,
    prefer_rust: bool = True,
) -> KuramotoVariantResult:
    """Simulate balanced gain/loss PT-symmetric Kuramoto dynamics."""
    _validate_time_grid(dt, n_steps)
    if prefer_rust:
        try:
            import scpn_quantum_engine as _engine

            times, r_values, pt_norm, imbalance = _engine.pt_symmetric_kuramoto_trajectory(
                _required_theta0(spec.theta0),
                spec.omega,
                spec.K_nm,
                spec.gain_loss,
                dt,
                n_steps,
            )
            return KuramotoVariantResult(
                variant=KuramotoVariant.PT_SYMMETRIC,
                times=np.asarray(times, dtype=np.float64),
                r_values=np.asarray(r_values, dtype=np.float64),
                backend="rust:pt_symmetric_kuramoto_trajectory",
                diagnostics={
                    "pt_norm": np.asarray(pt_norm, dtype=np.float64),
                    "gain_loss_imbalance": np.asarray(imbalance, dtype=np.float64),
                },
            )
        except (ImportError, AttributeError):
            pass

    times, r_values, pt_norm, imbalance = _pt_symmetric_numpy(spec, dt=dt, n_steps=n_steps)
    return KuramotoVariantResult(
        variant=KuramotoVariant.PT_SYMMETRIC,
        times=times,
        r_values=r_values,
        backend="numpy:pt_symmetric_kuramoto_trajectory",
        diagnostics={"pt_norm": pt_norm, "gain_loss_imbalance": imbalance},
    )


def _higher_order_numpy(
    spec: HigherOrderKuramotoSpec,
    *,
    dt: float,
    n_steps: int,
) -> tuple[FloatArray, FloatArray]:
    theta = np.array(_required_theta0(spec.theta0), dtype=np.float64, copy=True)
    times, r_values = _empty_trajectory(n_steps, dt)
    r_values[0] = _order_parameter(theta)
    for step in range(n_steps):
        dtheta = _pairwise_velocity(theta, spec.omega, spec.K_nm)
        for (i, j, k), weight in zip(spec.hyperedges, spec.hyper_weights, strict=True):
            dtheta[i] += weight * np.sin(theta[j] + theta[k] - 2.0 * theta[i])
        theta += dt * dtheta
        r_values[step + 1] = _order_parameter(theta)
    return times, r_values


def _monitored_numpy(
    spec: MonitoredKuramotoSpec,
    *,
    dt: float,
    n_steps: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    theta = np.array(_required_theta0(spec.theta0), dtype=np.float64, copy=True)
    times, r_values = _empty_trajectory(n_steps, dt)
    readouts = np.zeros(n_steps + 1, dtype=np.float64)
    feedback = np.zeros(n_steps + 1, dtype=np.float64)

    for step in range(n_steps + 1):
        r_value, mean_phase = _order_parameter_with_phase(theta)
        readout = (1.0 - spec.measurement_strength) * r_value + (
            spec.measurement_strength * spec.target_r
        )
        r_values[step] = r_value
        readouts[step] = readout
        feedback[step] = spec.monitor_gain * (spec.target_r - readout)
        if step == n_steps:
            break
        dtheta = _pairwise_velocity(theta, spec.omega, spec.K_nm)
        dtheta += feedback[step] * np.sin(mean_phase - theta)
        theta += dt * dtheta
    return times, r_values, readouts, feedback


def _pt_symmetric_numpy(
    spec: PTSymmetricKuramotoSpec,
    *,
    dt: float,
    n_steps: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    theta0 = np.asarray(_required_theta0(spec.theta0), dtype=np.float64)
    z = np.exp(1j * theta0).astype(np.complex128)
    times, r_values = _empty_trajectory(n_steps, dt)
    pt_norm = np.zeros(n_steps + 1, dtype=np.float64)
    imbalance = np.zeros(n_steps + 1, dtype=np.float64)
    n = z.size

    for step in range(n_steps + 1):
        powers = np.abs(z) ** 2
        norm = float(np.sum(powers))
        r_values[step] = float(abs(np.sum(z)) / np.sqrt(norm * n))
        pt_norm[step] = norm / n
        imbalance[step] = float(np.dot(spec.gain_loss, powers))
        if step == n_steps:
            break
        theta = np.angle(z)
        dtheta = _pairwise_velocity(theta, spec.omega, spec.K_nm)
        z += dt * (spec.gain_loss + 1j * dtheta) * z
        norm_after = float(np.linalg.norm(z))
        if norm_after > 0.0:
            z *= np.sqrt(n) / norm_after
    return times, r_values, pt_norm, imbalance


def _pairwise_velocity(theta: FloatArray, omega: FloatArray, K_nm: FloatArray) -> FloatArray:
    phase_delta = theta[None, :] - theta[:, None]
    return cast(FloatArray, omega + np.sum(K_nm * np.sin(phase_delta), axis=1))


def _order_parameter(theta: FloatArray) -> float:
    return float(abs(np.mean(np.exp(1j * theta))))


def _order_parameter_with_phase(theta: FloatArray) -> tuple[float, float]:
    z = complex(np.mean(np.exp(1j * theta)))
    return float(abs(z)), float(np.angle(z))


def _empty_trajectory(n_steps: int, dt: float) -> tuple[FloatArray, FloatArray]:
    times = np.arange(n_steps + 1, dtype=np.float64) * dt
    return times, np.zeros(n_steps + 1, dtype=np.float64)


def _prepare_theta0(theta0: FloatArray | None, omega: FloatArray) -> FloatArray:
    if theta0 is None:
        theta = np.mod(omega, 2.0 * np.pi).astype(np.float64)
    else:
        theta = _readonly_float_array(theta0, "theta0")
        if theta.shape != omega.shape:
            raise ValueError(f"theta0 must have shape {omega.shape}, got {theta.shape}")
        theta = np.array(theta, dtype=np.float64, copy=True)
    theta.setflags(write=False)
    return theta


def _required_theta0(theta0: FloatArray | None) -> FloatArray:
    if theta0 is None:
        raise ValueError("theta0 was not initialised")
    return theta0


def _validate_hyperedges(hyperedges: IntArray, n_oscillators: int) -> IntArray:
    edges = np.array(hyperedges, dtype=np.int64, copy=True)
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError("hyperedges must have shape (n_edges, 3)")
    if edges.size > 0 and (int(np.min(edges)) < 0 or int(np.max(edges)) >= n_oscillators):
        raise ValueError(f"hyperedge indices must be in [0, {n_oscillators})")
    edges.setflags(write=False)
    return edges


def _readonly_float_array(values: FloatArray, name: str) -> FloatArray:
    arr = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    arr.setflags(write=False)
    return arr


def _validate_metadata(metadata: Mapping[str, JsonScalar]) -> dict[str, JsonScalar]:
    out = dict(metadata)
    for key in out:
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
    try:
        json.dumps(out, sort_keys=True)
    except TypeError as exc:
        raise TypeError("metadata must be JSON-serialisable") from exc
    return out


def validate_variant_kuramoto_inputs(
    K_nm: FloatArray,
    omega: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Validate and copy a finite symmetric Kuramoto problem."""
    K_arr = _readonly_float_array(K_nm, "K_nm")
    omega_arr = _readonly_float_array(omega, "omega")
    if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
        raise ValueError(f"K_nm must be a square matrix, got shape {K_arr.shape}")
    if omega_arr.shape != (K_arr.shape[0],):
        raise ValueError(f"omega must have shape ({K_arr.shape[0]},), got {omega_arr.shape}")
    if not np.allclose(K_arr, K_arr.T, atol=1e-12, rtol=1e-12):
        raise ValueError("K_nm must be symmetric for Kuramoto variant trajectories")
    K_copy = np.array(K_arr, dtype=np.float64, copy=True)
    np.fill_diagonal(K_copy, 0.0)
    K_copy.setflags(write=False)
    return K_copy, omega_arr


def _validate_time_grid(dt: float, n_steps: int) -> None:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")


def _require_range(value: float, lower: float, upper: float, name: str) -> None:
    if not np.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _serialise_diagnostic(value: FloatArray | float | int | str | bool) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "final": float(value[-1]),
        }
    return value
