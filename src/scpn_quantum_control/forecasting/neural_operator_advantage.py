# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Neural-operator surrogate vs direct simulation: an honest advantage study
r"""Does the Kuramoto neural-operator surrogate genuinely beat direct simulation?

A trained surrogate that merely *reduces its loss* has not been shown to be useful. This study asks
the sharper question — for a concrete network, does the DeepONet surrogate of
:mod:`.kuramoto_neural_operator` forecast unseen initial conditions accurately, and is it cheaper than
integrating the Kuramoto equations directly? — and answers it along three axes, keeping the
host-independent claims separate from the host-dependent ones.

* **Fidelity (host-independent).** On held-out initial conditions the surrogate's mean wrapped
  angular error over the horizon is compared against the naive *persistence* baseline (hold the
  initial phase). The surrogate is required to beat persistence; the error-versus-horizon curve shows
  where it does so.
* **Operation count (host-independent).** The :mod:`.neural_operator_cost_model` gives the exact,
  model-free number of right-hand-side evaluations direct RK4 needs to reach the horizon (which the
  surrogate replaces with one forward pass) and, under a stated FLOP model, the per-query FLOP ratio
  and the amortised break-even query count.
* **Wall clock (host-dependent, boundary-guarded).** Millisecond timings for a direct trajectory and a
  single-query forecast are recorded *only* as advisory evidence, captured alongside the host
  provenance, and explicitly excluded from the reproducible set — a Rust-accelerated twenty-step
  integrator can be faster than a DeepONet forward pass at small ``N``, and saying otherwise would be
  a host-specific overclaim. The reproducible efficiency statement is the operation-count model, not
  the milliseconds.

The honest conclusion this study is built to report is that the surrogate's advantage is *structural*
— random access to any query time in one pass, and amortisation of the one-time training cost over
many inferences — realised at scale, and it is not a single-query wall-clock speedup at small ``N``.

This capability requires PyTorch (``scpn-quantum-control[torch]``); the training and forecasting are
delegated to :mod:`.kuramoto_neural_operator`, so this module imports no framework directly and stays
importable without it, raising a clear installation hint only when the study is actually run.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.accel.diff_kuramoto_rk4 import (
    kuramoto_rk4_trajectory,
    last_kuramoto_rk4_trajectory_tier_used,
)
from scpn_quantum_control.accel.tier_benchmark import capture_provenance

from .kuramoto_neural_operator import simulate_operator_dataset, train_kuramoto_neural_operator
from .neural_operator_cost_model import SurrogateCostModel, build_cost_model

#: The artefact schema tag (bump on any breaking field change).
SCHEMA = "scpn-quantum-control.neural-operator-advantage.v1"

#: This study does not license a production performance claim; the reproducible set is fidelity and
#: the operation-count model, never the wall-clock milliseconds.
PRODUCTION_CLAIM_ALLOWED = False

#: What the numbers do and do not license.
CLAIM_BOUNDARY = (
    "The reproducible quantities are the held-out forecast fidelity and the host-independent "
    "operation-count model. The wall-clock milliseconds are advisory host-bounded evidence, captured "
    "under the recorded load and governor, and are excluded from any performance claim. A "
    "Rust-accelerated fixed-step integrator can outrun a DeepONet forward pass at small N; the "
    "surrogate's advantage is the structural random-access and amortisation captured by the "
    "operation-count model, not a single-query millisecond margin."
)

#: How determinism is asserted.
DETERMINISM = (
    "Fidelity is deterministic on a fixed host for fixed dataset, training and evaluation seeds "
    "(NumPy default_rng plus a torch manual seed on CPU); reproducibility is asserted on the content "
    "(loss curve, forecast, cost model), never on timings. The payload digest covers only the "
    "host-independent cost model and configuration, which are bit-exact everywhere."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _wrapped_angle_error(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    diff = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.mean(np.abs(diff)))


def _median_ms(fn: Callable[[], object], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1.0e3)
    samples.sort()
    return samples[len(samples) // 2]


@dataclass(frozen=True)
class HeldOutFidelity:
    """Forecast accuracy on held-out initial conditions against the persistence baseline.

    Attributes
    ----------
    n_eval : int
        The number of held-out initial conditions.
    horizon : float
        The forecast horizon ``T``.
    surrogate_mean_error : float
        The surrogate's mean wrapped angular error over the horizon (radians).
    persistence_mean_error : float
        The persistence baseline's mean wrapped angular error over the horizon (radians).
    surrogate_terminal_error : float
        The surrogate's wrapped angular error at the horizon (radians).
    persistence_terminal_error : float
        The persistence baseline's error at the horizon (radians).
    beats_persistence : bool
        Whether the surrogate's mean error is strictly below persistence.
    error_vs_horizon : tuple
        Per-query-time ``(time, surrogate_error, persistence_error)`` triples (radians).
    """

    n_eval: int
    horizon: float
    surrogate_mean_error: float
    persistence_mean_error: float
    surrogate_terminal_error: float
    persistence_terminal_error: float
    beats_persistence: bool
    error_vs_horizon: tuple[tuple[float, float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the fidelity block."""
        return {
            "n_eval": self.n_eval,
            "horizon": self.horizon,
            "surrogate_mean_error": self.surrogate_mean_error,
            "persistence_mean_error": self.persistence_mean_error,
            "surrogate_terminal_error": self.surrogate_terminal_error,
            "persistence_terminal_error": self.persistence_terminal_error,
            "beats_persistence": self.beats_persistence,
            "error_vs_horizon": [
                {"time": t, "surrogate_error": s, "persistence_error": p}
                for t, s, p in self.error_vs_horizon
            ],
        }


@dataclass(frozen=True)
class NeuralOperatorAdvantage:
    """The full surrogate-versus-direct-simulation picture for one network and configuration.

    Attributes
    ----------
    schema : str
        The artefact schema tag.
    generated_utc : str
        The ISO-8601 UTC timestamp of the run.
    n_oscillators : int
        The number of oscillators ``N``.
    dt : float
        The RK4 step.
    n_steps : int
        The number of RK4 steps to the horizon.
    horizon : float
        The horizon ``T = dt · n_steps``.
    rk4_tier : str or None
        The dispatched RK4 tier used for the ground-truth trajectories (``rust`` / ``python`` / ``julia``).
    loss_start : float
        The first-epoch training loss.
    loss_final : float
        The final-epoch training loss.
    fidelity : HeldOutFidelity
        The held-out forecast fidelity.
    cost_model : SurrogateCostModel
        The host-independent operation-count model.
    wall_clock_ms : dict or None
        Advisory host-bounded timings, or ``None`` when timing was not measured.
    provenance : dict
        The host, toolchain and revision provenance.
    claim_boundary : str
        What the numbers license.
    determinism : str
        How determinism is asserted.
    production_claim_allowed : bool
        Always ``False``.
    payload_sha256 : str
        A digest of the host-independent cost model and configuration.
    """

    schema: str
    generated_utc: str
    n_oscillators: int
    dt: float
    n_steps: int
    horizon: float
    rk4_tier: str | None
    loss_start: float
    loss_final: float
    fidelity: HeldOutFidelity
    cost_model: SurrogateCostModel
    wall_clock_ms: dict[str, float] | None
    provenance: dict[str, Any]
    claim_boundary: str
    determinism: str
    production_claim_allowed: bool
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the full study."""
        return {
            "schema": self.schema,
            "generated_utc": self.generated_utc,
            "n_oscillators": self.n_oscillators,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "horizon": self.horizon,
            "rk4_tier": self.rk4_tier,
            "loss_start": self.loss_start,
            "loss_final": self.loss_final,
            "fidelity": self.fidelity.to_dict(),
            "cost_model": self.cost_model.to_dict(),
            "wall_clock_ms": self.wall_clock_ms,
            "provenance": self.provenance,
            "claim_boundary": self.claim_boundary,
            "determinism": self.determinism,
            "production_claim_allowed": self.production_claim_allowed,
            "payload_sha256": self.payload_sha256,
        }


def _payload_digest(cost_model: SurrogateCostModel, config: dict[str, Any]) -> str:
    payload = {"schema": SCHEMA, "config": config, "cost_model": cost_model.to_dict()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _held_out_fidelity(
    operator: Any,
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    dt: float,
    n_steps: int,
    n_eval: int,
    eval_seed: int,
) -> tuple[HeldOutFidelity, str | None]:
    rng = np.random.default_rng(eval_seed)
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    count = omega.size
    surrogate_mean = np.zeros(n_eval, dtype=np.float64)
    persistence_mean = np.zeros(n_eval, dtype=np.float64)
    surrogate_terminal = np.zeros(n_eval, dtype=np.float64)
    persistence_terminal = np.zeros(n_eval, dtype=np.float64)
    per_time_surrogate = np.zeros(n_steps + 1, dtype=np.float64)
    per_time_persistence = np.zeros(n_steps + 1, dtype=np.float64)
    tier: str | None = None
    for index in range(n_eval):
        theta0 = rng.uniform(0.0, 2.0 * np.pi, size=count)
        truth = kuramoto_rk4_trajectory(theta0, omega, coupling, dt, n_steps)
        tier = last_kuramoto_rk4_trajectory_tier_used()
        forecast = operator.forecast(theta0, times)
        persistence = np.tile(truth[0], (n_steps + 1, 1))
        surrogate_mean[index] = _wrapped_angle_error(forecast, truth)
        persistence_mean[index] = _wrapped_angle_error(persistence, truth)
        surrogate_terminal[index] = _wrapped_angle_error(forecast[-1], truth[-1])
        persistence_terminal[index] = _wrapped_angle_error(persistence[-1], truth[-1])
        for step in range(n_steps + 1):
            per_time_surrogate[step] += _wrapped_angle_error(forecast[step], truth[step])
            per_time_persistence[step] += _wrapped_angle_error(persistence[step], truth[step])
    per_time_surrogate /= n_eval
    per_time_persistence /= n_eval
    surrogate_error = float(np.mean(surrogate_mean))
    persistence_error = float(np.mean(persistence_mean))
    curve = tuple(
        (float(times[step]), float(per_time_surrogate[step]), float(per_time_persistence[step]))
        for step in range(n_steps + 1)
    )
    fidelity = HeldOutFidelity(
        n_eval=n_eval,
        horizon=float(times[-1]),
        surrogate_mean_error=surrogate_error,
        persistence_mean_error=persistence_error,
        surrogate_terminal_error=float(np.mean(surrogate_terminal)),
        persistence_terminal_error=float(np.mean(persistence_terminal)),
        beats_persistence=surrogate_error < persistence_error,
        error_vs_horizon=curve,
    )
    return fidelity, tier


def evaluate_neural_operator_advantage(
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    dt: float,
    n_steps: int,
    n_trajectories: int,
    n_eval: int,
    latent_dim: int = 32,
    hidden_dim: int = 96,
    epochs: int = 300,
    learning_rate: float = 3e-3,
    dataset_seed: int = 3,
    train_seed: int = 0,
    eval_seed: int = 9999,
    measure_wall_clock: bool = True,
    clock: Callable[[], datetime] = _utc_now,
) -> NeuralOperatorAdvantage:
    r"""Run the surrogate-versus-direct-simulation advantage study for a network (requires PyTorch).

    Trains the DeepONet surrogate on RK4 rollouts of the given network, measures its held-out forecast
    fidelity against the persistence baseline, assembles the host-independent operation-count model,
    and optionally records advisory wall-clock timings.

    Parameters
    ----------
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N ≥ 2``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step (``> 0``).
    n_steps : int
        The number of RK4 steps to the horizon (``≥ 1``).
    n_trajectories : int
        The number of training trajectories (``≥ 1``).
    n_eval : int
        The number of held-out evaluation initial conditions (``≥ 1``).
    latent_dim : int
        The DeepONet latent width ``p``.
    hidden_dim : int
        The DeepONet hidden width.
    epochs : int
        The number of full-batch training epochs.
    learning_rate : float
        The Adam learning rate.
    dataset_seed : int
        The seed for the training initial conditions.
    train_seed : int
        The torch manual seed for training.
    eval_seed : int
        The seed for the held-out evaluation initial conditions.
    measure_wall_clock : bool
        Whether to record advisory host-bounded timings (disable for deterministic output).
    clock : Callable[[], datetime]
        The UTC clock (injectable for deterministic tests).

    Returns
    -------
    NeuralOperatorAdvantage
        The full fidelity / operation-count / advisory-timing study.

    Raises
    ------
    ValueError
        If ``n_eval`` is not positive (other bounds are enforced by the delegated builders).
    ImportError
        If PyTorch is not installed.
    """
    if n_eval < 1:
        raise ValueError(f"n_eval must be positive, got {n_eval}")
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    dataset = simulate_operator_dataset(
        frequencies,
        matrix,
        n_trajectories=n_trajectories,
        dt=dt,
        n_steps=n_steps,
        seed=dataset_seed,
    )
    operator = train_kuramoto_neural_operator(
        dataset,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=train_seed,
    )
    fidelity, tier = _held_out_fidelity(
        operator,
        frequencies,
        matrix,
        dt=dt,
        n_steps=n_steps,
        n_eval=n_eval,
        eval_seed=eval_seed,
    )
    cost_model = build_cost_model(
        frequencies.size,
        n_steps=n_steps,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_trajectories=n_trajectories,
        epochs=epochs,
    )
    wall_clock: dict[str, float] | None = None
    if measure_wall_clock:
        horizon = dt * n_steps
        probe_rng = np.random.default_rng(eval_seed + 1)
        theta0 = probe_rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
        terminal = np.array([horizon], dtype=np.float64)
        wall_clock = {
            "direct_full_trajectory_ms": _median_ms(
                lambda: kuramoto_rk4_trajectory(theta0, frequencies, matrix, dt, n_steps),
                warmup=3,
                repeats=21,
            ),
            "surrogate_single_query_ms": _median_ms(
                lambda: operator.forecast(theta0, terminal), warmup=3, repeats=21
            ),
        }
    config = {
        "n_oscillators": int(frequencies.size),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "n_trajectories": int(n_trajectories),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
    }
    return NeuralOperatorAdvantage(
        schema=SCHEMA,
        generated_utc=clock().isoformat(),
        n_oscillators=int(frequencies.size),
        dt=float(dt),
        n_steps=int(n_steps),
        horizon=float(dt * n_steps),
        rk4_tier=tier,
        loss_start=float(operator.loss_history[0]),
        loss_final=float(operator.loss_history[-1]),
        fidelity=fidelity,
        cost_model=cost_model,
        wall_clock_ms=wall_clock,
        provenance=capture_provenance().to_dict(),
        claim_boundary=CLAIM_BOUNDARY,
        determinism=DETERMINISM,
        production_claim_allowed=PRODUCTION_CLAIM_ALLOWED,
        payload_sha256=_payload_digest(cost_model, config),
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "DETERMINISM",
    "PRODUCTION_CLAIM_ALLOWED",
    "SCHEMA",
    "HeldOutFidelity",
    "NeuralOperatorAdvantage",
    "evaluate_neural_operator_advantage",
]
