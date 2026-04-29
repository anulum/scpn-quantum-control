# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Automated Kuramoto witness discovery
"""Automated Kuramoto witness discovery with Bayesian and bandit search."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .sync_witness import WitnessResult, fiedler_witness_from_correlator

FloatArray = NDArray[np.float64]
JsonScalar = str | int | float | bool | None


class WitnessSearchMode(str, Enum):
    """Candidate proposal modes used by the discovery loop."""

    INITIAL = "initial"
    BAYESIAN_UCB = "bayesian_ucb"
    RL_BANDIT = "rl_bandit"


@dataclass(frozen=True)
class WitnessCandidate:
    """Three-parameter Kuramoto witness-search candidate."""

    coupling_scale: float
    omega_scale: float
    phase_bias: float

    def __post_init__(self) -> None:
        _require_non_negative(self.coupling_scale, "coupling_scale")
        _require_non_negative(self.omega_scale, "omega_scale")
        _require_finite(self.phase_bias, "phase_bias")

    def as_array(self) -> FloatArray:
        """Return `[coupling_scale, omega_scale, phase_bias]`."""
        return np.array(
            [self.coupling_scale, self.omega_scale, self.phase_bias],
            dtype=np.float64,
        )

    def to_metadata(self) -> dict[str, float]:
        """Return serialisable candidate parameters."""
        return {
            "coupling_scale": float(self.coupling_scale),
            "omega_scale": float(self.omega_scale),
            "phase_bias": float(self.phase_bias),
        }


@dataclass(frozen=True)
class WitnessDiscoverySpec:
    """Configuration for automated Kuramoto witness discovery."""

    dt: float = 0.02
    n_steps: int = 80
    n_initial: int = 8
    n_iterations: int = 5
    batch_size: int = 3
    pool_size: int = 64
    seed: int = 20260429
    coupling_bounds: tuple[float, float] = (0.0, 2.0)
    omega_bounds: tuple[float, float] = (0.5, 1.5)
    phase_bias_bounds: tuple[float, float] = (-0.5, 0.5)
    correlation_threshold: float = 0.5
    fiedler_threshold: float = 0.4
    final_r_weight: float = 1.0
    correlation_weight: float = 0.35
    fiedler_weight: float = 0.15
    witness_margin_weight: float = 0.25
    novelty_weight: float = 0.03
    ucb_beta: float = 1.5
    rl_epsilon: float = 0.2
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive(self.dt, "dt")
        _require_positive_int(self.n_steps, "n_steps")
        _require_positive_int(self.n_initial, "n_initial")
        _require_positive_int(self.n_iterations, "n_iterations")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.pool_size, "pool_size")
        if self.pool_size < self.batch_size:
            raise ValueError("pool_size must be at least batch_size")
        for name in ("coupling_bounds", "omega_bounds", "phase_bias_bounds"):
            _validate_bounds(cast(tuple[float, float], getattr(self, name)), name)
        _require_range(self.correlation_threshold, -2.0, 2.0, "correlation_threshold")
        _require_non_negative(self.fiedler_threshold, "fiedler_threshold")
        _require_non_negative(self.final_r_weight, "final_r_weight")
        _require_non_negative(self.correlation_weight, "correlation_weight")
        _require_non_negative(self.fiedler_weight, "fiedler_weight")
        _require_non_negative(self.witness_margin_weight, "witness_margin_weight")
        _require_non_negative(self.novelty_weight, "novelty_weight")
        _require_non_negative(self.ucb_beta, "ucb_beta")
        _require_range(self.rl_epsilon, 0.0, 1.0, "rl_epsilon")
        metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class WitnessDiscoveryEvaluation:
    """One scored witness-discovery candidate."""

    candidate: WitnessCandidate
    score: float
    final_r: float
    mean_correlation: float
    fiedler_value: float
    correlation_witness: WitnessResult
    fiedler_witness: WitnessResult
    source: WitnessSearchMode
    acquisition_value: float
    backend: str

    @property
    def witness_margin(self) -> float:
        """Positive margin by which synchronisation witnesses fire."""
        return float(
            max(0.0, -self.correlation_witness.expectation_value)
            + max(0.0, -self.fiedler_witness.expectation_value)
        )

    @property
    def is_synchronised(self) -> bool:
        """Whether either witness fires for this candidate."""
        return bool(
            self.correlation_witness.is_synchronized or self.fiedler_witness.is_synchronized
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return serialisable evaluation metadata."""
        return {
            "candidate": self.candidate.to_metadata(),
            "score": float(self.score),
            "final_r": float(self.final_r),
            "mean_correlation": float(self.mean_correlation),
            "fiedler_value": float(self.fiedler_value),
            "witness_margin": self.witness_margin,
            "is_synchronised": self.is_synchronised,
            "source": self.source.value,
            "acquisition_value": float(self.acquisition_value),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class WitnessDiscoveryResult:
    """Complete automated witness-discovery trace."""

    evaluations: tuple[WitnessDiscoveryEvaluation, ...]
    best: WitnessDiscoveryEvaluation
    spec: WitnessDiscoverySpec
    backend: str
    n_oscillators: int

    def to_metadata(self) -> dict[str, Any]:
        """Return serialisable discovery summary."""
        return {
            "backend": self.backend,
            "n_oscillators": int(self.n_oscillators),
            "n_evaluations": len(self.evaluations),
            "best": self.best.to_metadata(),
            "metadata": dict(self.spec.metadata),
        }

    def ranked(self, limit: int | None = None) -> tuple[WitnessDiscoveryEvaluation, ...]:
        """Return evaluations sorted by descending score."""
        ranked = tuple(sorted(self.evaluations, key=lambda item: item.score, reverse=True))
        return ranked if limit is None else ranked[:limit]

    def to_json(self) -> str:
        """Serialise the discovery trace to compact JSON."""
        payload = self.to_metadata()
        payload["evaluations"] = [item.to_metadata() for item in self.evaluations]
        return json.dumps(payload, sort_keys=True)


def discover_kuramoto_witnesses(
    K_nm: FloatArray,
    omega: FloatArray,
    *,
    theta0: FloatArray | None = None,
    spec: WitnessDiscoverySpec | None = None,
    prefer_rust: bool = True,
) -> WitnessDiscoveryResult:
    """Run automated Kuramoto witness discovery.

    The loop starts with a deterministic Latin-hypercube design, then combines
    an RBF Bayesian upper-confidence-bound acquisition with a bandit-style local
    exploration policy around the current best candidate.
    """
    cfg = spec or WitnessDiscoverySpec()
    K, omega_arr = _validate_problem(K_nm, omega)
    theta = _prepare_theta0(theta0, omega_arr)
    rng = np.random.default_rng(cfg.seed)
    evaluated: list[WitnessDiscoveryEvaluation] = []

    initial_candidates = _latin_hypercube_candidates(cfg, cfg.n_initial, rng)
    evaluated.extend(
        _evaluate_candidate_batch(
            theta,
            omega_arr,
            K,
            initial_candidates,
            cfg,
            WitnessSearchMode.INITIAL,
            np.zeros(initial_candidates.shape[0], dtype=np.float64),
            prefer_rust,
        )
    )

    for _ in range(cfg.n_iterations):
        best = max(evaluated, key=lambda item: item.score)
        pool = _proposal_pool(cfg, best.candidate, rng)
        X_train = _candidate_matrix([item.candidate for item in evaluated])
        y_train = np.array([item.score for item in evaluated], dtype=np.float64)
        acquisition = _bayesian_ucb(X_train, y_train, pool, cfg)
        bayesian_candidates, bayesian_acq = _select_unique_candidates(
            pool,
            acquisition,
            evaluated,
            max(cfg.batch_size - 1, 1),
        )
        rl_candidates = _rl_bandit_candidates(cfg, best.candidate, 1, rng)
        rl_acq = np.full(rl_candidates.shape[0], best.score, dtype=np.float64)

        if bayesian_candidates.size > 0:
            evaluated.extend(
                _evaluate_candidate_batch(
                    theta,
                    omega_arr,
                    K,
                    bayesian_candidates,
                    cfg,
                    WitnessSearchMode.BAYESIAN_UCB,
                    bayesian_acq,
                    prefer_rust,
                )
            )
        evaluated.extend(
            _evaluate_candidate_batch(
                theta,
                omega_arr,
                K,
                rl_candidates,
                cfg,
                WitnessSearchMode.RL_BANDIT,
                rl_acq,
                prefer_rust,
            )
        )

    best = max(evaluated, key=lambda item: item.score)
    backend = (
        "rust:kuramoto_witness_candidate_features"
        if any(item.backend.startswith("rust:") for item in evaluated)
        else "numpy:kuramoto_witness_candidate_features"
    )
    return WitnessDiscoveryResult(
        evaluations=tuple(evaluated),
        best=best,
        spec=cfg,
        backend=backend,
        n_oscillators=int(omega_arr.size),
    )


def score_witness_candidates(
    K_nm: FloatArray,
    omega: FloatArray,
    candidates: Sequence[WitnessCandidate],
    *,
    theta0: FloatArray | None = None,
    spec: WitnessDiscoverySpec | None = None,
    source: WitnessSearchMode = WitnessSearchMode.INITIAL,
    prefer_rust: bool = True,
) -> tuple[WitnessDiscoveryEvaluation, ...]:
    """Score a fixed candidate batch through the witness objective."""
    cfg = spec or WitnessDiscoverySpec(n_iterations=1)
    K, omega_arr = _validate_problem(K_nm, omega)
    theta = _prepare_theta0(theta0, omega_arr)
    candidate_matrix = _candidate_matrix(candidates)
    acquisition = np.zeros(candidate_matrix.shape[0], dtype=np.float64)
    return tuple(
        _evaluate_candidate_batch(
            theta,
            omega_arr,
            K,
            candidate_matrix,
            cfg,
            source,
            acquisition,
            prefer_rust,
        )
    )


def _evaluate_candidate_batch(
    theta0: FloatArray,
    omega: FloatArray,
    K_nm: FloatArray,
    candidates: FloatArray,
    spec: WitnessDiscoverySpec,
    source: WitnessSearchMode,
    acquisition_values: FloatArray,
    prefer_rust: bool,
) -> list[WitnessDiscoveryEvaluation]:
    final_r, mean_corr, final_theta, backend = _candidate_features(
        theta0,
        omega,
        K_nm,
        candidates,
        spec.dt,
        spec.n_steps,
        prefer_rust,
    )
    evaluations: list[WitnessDiscoveryEvaluation] = []
    for index, row in enumerate(candidates):
        theta = final_theta[index]
        corr_matrix = np.cos(np.subtract.outer(theta, theta))
        corr_result = _correlation_witness_from_mean(
            float(mean_corr[index]),
            theta.size,
            spec.correlation_threshold,
        )
        fiedler_result = fiedler_witness_from_correlator(corr_matrix, spec.fiedler_threshold)
        fiedler_value = float(fiedler_result.raw_observable)
        margin = max(0.0, -corr_result.expectation_value) + max(
            0.0, -fiedler_result.expectation_value
        )
        score = (
            spec.final_r_weight * float(final_r[index])
            + spec.correlation_weight * max(0.0, float(mean_corr[index]))
            + spec.fiedler_weight * max(0.0, fiedler_value)
            + spec.witness_margin_weight * margin
        )
        evaluations.append(
            WitnessDiscoveryEvaluation(
                candidate=WitnessCandidate(
                    coupling_scale=float(row[0]),
                    omega_scale=float(row[1]),
                    phase_bias=float(row[2]),
                ),
                score=float(score),
                final_r=float(final_r[index]),
                mean_correlation=float(mean_corr[index]),
                fiedler_value=fiedler_value,
                correlation_witness=corr_result,
                fiedler_witness=fiedler_result,
                source=source,
                acquisition_value=float(acquisition_values[index]),
                backend=backend,
            )
        )
    return evaluations


def _candidate_features(
    theta0: FloatArray,
    omega: FloatArray,
    K_nm: FloatArray,
    candidates: FloatArray,
    dt: float,
    n_steps: int,
    prefer_rust: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, str]:
    if prefer_rust:
        try:
            import scpn_quantum_engine as _engine

            final_r, mean_corr, final_theta = _engine.kuramoto_witness_candidate_features(
                theta0,
                omega,
                K_nm,
                candidates,
                dt,
                n_steps,
            )
            return (
                np.asarray(final_r, dtype=np.float64),
                np.asarray(mean_corr, dtype=np.float64),
                np.asarray(final_theta, dtype=np.float64),
                "rust:kuramoto_witness_candidate_features",
            )
        except (ImportError, AttributeError):
            pass
    final_r, mean_corr, final_theta = _candidate_features_numpy(
        theta0,
        omega,
        K_nm,
        candidates,
        dt,
        n_steps,
    )
    return final_r, mean_corr, final_theta, "numpy:kuramoto_witness_candidate_features"


def _candidate_features_numpy(
    theta0: FloatArray,
    omega: FloatArray,
    K_nm: FloatArray,
    candidates: FloatArray,
    dt: float,
    n_steps: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    final_r = np.zeros(candidates.shape[0], dtype=np.float64)
    mean_corr = np.zeros(candidates.shape[0], dtype=np.float64)
    final_theta = np.zeros((candidates.shape[0], theta0.size), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        coupling_scale, omega_scale, phase_bias = candidate
        theta = np.array(theta0 + phase_bias, dtype=np.float64, copy=True)
        K_scaled = K_nm * coupling_scale
        omega_scaled = omega * omega_scale
        for _ in range(n_steps):
            phase_delta = theta[None, :] - theta[:, None]
            theta += dt * (omega_scaled + np.sum(K_scaled * np.sin(phase_delta), axis=1))
        final_theta[index] = theta
        z = np.exp(1j * theta)
        final_r[index] = float(abs(np.mean(z)))
        if theta.size > 1:
            corr = np.cos(np.subtract.outer(theta, theta))
            mean_corr[index] = float(
                np.sum(np.triu(corr, k=1)) / (theta.size * (theta.size - 1) / 2)
            )
        else:
            mean_corr[index] = 1.0
    return final_r, mean_corr, final_theta


def _correlation_witness_from_mean(
    mean_corr: float,
    n_qubits: int,
    threshold: float,
) -> WitnessResult:
    expectation = threshold - mean_corr
    return WitnessResult(
        witness_name="correlation",
        expectation_value=float(expectation),
        threshold=float(threshold),
        is_synchronized=expectation < 0.0,
        raw_observable=float(mean_corr),
        n_qubits=int(n_qubits),
    )


def _bayesian_ucb(
    X_train: FloatArray,
    y_train: FloatArray,
    X_pool: FloatArray,
    spec: WitnessDiscoverySpec,
) -> FloatArray:
    mean, std = _rbf_surrogate_predict(X_train, y_train, X_pool)
    novelty = _nearest_distance(X_train, X_pool)
    return cast(FloatArray, mean + spec.ucb_beta * std + spec.novelty_weight * novelty)


def _rbf_surrogate_predict(
    X_train: FloatArray,
    y_train: FloatArray,
    X_pool: FloatArray,
    length_scale: float = 0.55,
    noise: float = 1e-8,
) -> tuple[FloatArray, FloatArray]:
    if X_train.shape[0] < 2:
        return (
            np.zeros(X_pool.shape[0], dtype=np.float64),
            np.ones(X_pool.shape[0], dtype=np.float64),
        )
    Xn, lower, span = _normalise_against(X_train, X_train)
    Xpn = (X_pool - lower) / span
    K = _rbf_kernel(Xn, Xn, length_scale) + np.eye(X_train.shape[0]) * noise
    Ks = _rbf_kernel(Xpn, Xn, length_scale)
    centred = y_train - float(np.mean(y_train))
    try:
        alpha = np.linalg.solve(K, centred)
        v = np.linalg.solve(K, Ks.T)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K)
        alpha = K_inv @ centred
        v = K_inv @ Ks.T
    mean = Ks @ alpha + float(np.mean(y_train))
    variance = np.maximum(1.0 - np.sum(Ks * v.T, axis=1), 0.0)
    return cast(FloatArray, mean), cast(FloatArray, np.sqrt(variance))


def _proposal_pool(
    spec: WitnessDiscoverySpec,
    best: WitnessCandidate,
    rng: np.random.Generator,
) -> FloatArray:
    n_random = max(spec.pool_size // 2, spec.batch_size)
    random_pool = _uniform_candidates(spec, n_random, rng)
    local_pool = _rl_bandit_candidates(spec, best, spec.pool_size - n_random, rng)
    return np.vstack([random_pool, local_pool]).astype(np.float64)


def _rl_bandit_candidates(
    spec: WitnessDiscoverySpec,
    best: WitnessCandidate,
    n_candidates: int,
    rng: np.random.Generator,
) -> FloatArray:
    if n_candidates <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    bounds = _bounds_matrix(spec)
    centre = best.as_array()
    span = bounds[:, 1] - bounds[:, 0]
    out = np.zeros((n_candidates, 3), dtype=np.float64)
    for index in range(n_candidates):
        if rng.random() < spec.rl_epsilon:
            out[index] = rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            perturb = rng.normal(0.0, span * 0.12)
            out[index] = np.clip(centre + perturb, bounds[:, 0], bounds[:, 1])
    return out


def _select_unique_candidates(
    pool: FloatArray,
    acquisition: FloatArray,
    evaluated: Sequence[WitnessDiscoveryEvaluation],
    n_select: int,
) -> tuple[FloatArray, FloatArray]:
    seen = {_candidate_key(item.candidate.as_array()) for item in evaluated}
    rows: list[FloatArray] = []
    values: list[float] = []
    for index in np.argsort(acquisition)[::-1]:
        row = pool[int(index)]
        key = _candidate_key(row)
        if key in seen:
            continue
        rows.append(row)
        values.append(float(acquisition[int(index)]))
        seen.add(key)
        if len(rows) >= n_select:
            break
    if not rows:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.vstack(rows).astype(np.float64), np.array(values, dtype=np.float64)


def _latin_hypercube_candidates(
    spec: WitnessDiscoverySpec,
    n_candidates: int,
    rng: np.random.Generator,
) -> FloatArray:
    bounds = _bounds_matrix(spec)
    base = (np.arange(n_candidates, dtype=np.float64) + 0.5) / n_candidates
    out = np.zeros((n_candidates, 3), dtype=np.float64)
    for dim in range(3):
        perm = rng.permutation(n_candidates)
        out[:, dim] = bounds[dim, 0] + base[perm] * (bounds[dim, 1] - bounds[dim, 0])
    return out


def _uniform_candidates(
    spec: WitnessDiscoverySpec,
    n_candidates: int,
    rng: np.random.Generator,
) -> FloatArray:
    bounds = _bounds_matrix(spec)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, 3)).astype(np.float64)


def _candidate_matrix(candidates: Sequence[WitnessCandidate]) -> FloatArray:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return np.vstack([candidate.as_array() for candidate in candidates]).astype(np.float64)


def _nearest_distance(X_train: FloatArray, X_pool: FloatArray) -> FloatArray:
    if X_train.size == 0:
        return np.ones(X_pool.shape[0], dtype=np.float64)
    Xn, lower, span = _normalise_against(X_train, X_train)
    Xpn = (X_pool - lower) / span
    distances = np.linalg.norm(Xpn[:, None, :] - Xn[None, :, :], axis=2)
    return cast(FloatArray, np.min(distances, axis=1))


def _normalise_against(
    reference: FloatArray,
    values: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    lower = np.min(reference, axis=0)
    upper = np.max(reference, axis=0)
    span = np.where(upper > lower, upper - lower, 1.0)
    return cast(FloatArray, (values - lower) / span), lower, span


def _rbf_kernel(X_a: FloatArray, X_b: FloatArray, length_scale: float) -> FloatArray:
    diff = X_a[:, None, :] - X_b[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    return cast(FloatArray, np.exp(-0.5 * sq_dist / (length_scale * length_scale)))


def _validate_problem(K_nm: FloatArray, omega: FloatArray) -> tuple[FloatArray, FloatArray]:
    K = np.array(K_nm, dtype=np.float64, copy=True)
    omega_arr = np.array(omega, dtype=np.float64, copy=True)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K_nm must be a square matrix, got shape {K.shape}")
    if omega_arr.shape != (K.shape[0],):
        raise ValueError(f"omega must have shape ({K.shape[0]},), got {omega_arr.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("K_nm must contain only finite values")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega must contain only finite values")
    if not np.allclose(K, K.T, atol=1e-12, rtol=1e-12):
        raise ValueError("K_nm must be symmetric")
    np.fill_diagonal(K, 0.0)
    K.setflags(write=False)
    omega_arr.setflags(write=False)
    return K, omega_arr


def _prepare_theta0(theta0: FloatArray | None, omega: FloatArray) -> FloatArray:
    if theta0 is None:
        theta = np.mod(omega, 2.0 * np.pi).astype(np.float64)
    else:
        theta = np.array(theta0, dtype=np.float64, copy=True)
        if theta.shape != omega.shape:
            raise ValueError(f"theta0 must have shape {omega.shape}, got {theta.shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta0 must contain only finite values")
    theta.setflags(write=False)
    return theta


def _bounds_matrix(spec: WitnessDiscoverySpec) -> FloatArray:
    return np.array(
        [spec.coupling_bounds, spec.omega_bounds, spec.phase_bias_bounds],
        dtype=np.float64,
    )


def _candidate_key(row: FloatArray) -> tuple[float, float, float]:
    return (float(round(row[0], 10)), float(round(row[1], 10)), float(round(row[2], 10)))


def _validate_bounds(bounds: tuple[float, float], name: str) -> None:
    low, high = bounds
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"{name} must be finite increasing bounds")


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


def _require_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_range(value: float, lower: float, upper: float, name: str) -> None:
    if not np.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
