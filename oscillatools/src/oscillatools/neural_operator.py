# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Neural-operator surrogate for Kuramoto network dynamics
r"""A DeepONet neural-operator surrogate for Kuramoto network dynamics.

The differentiable Kuramoto integrator is the natural *training-data substrate* for a learned
surrogate: it generates exact trajectories cheaply, and a neural operator distilled from them then
forecasts the evolution at a fraction of the cost and generalises to initial conditions it never saw
(Lu et al., DeepONet, 2021). The operator learned here is the solution map

.. math::

    \mathcal G : \bigl(\theta(0),\, t\bigr) \mapsto \theta(t),

realised as a DeepONet: a *branch* network encodes the initial condition (embedded as
``(\cos\theta_0, \sin\theta_0)`` to respect the phase circle) and a *trunk* network encodes the query
time, their inner product giving each output channel. Training minimises the mean-squared error against
trajectories produced by the toolkit's own Runge–Kutta Kuramoto integrator.

This module is an *optional* capability — it requires PyTorch (``oscillatools[torch]``) and so
lives outside the NumPy/Rust accelerator core, which stays importable without it. The dataset builder
is pure NumPy; only training and forecasting touch PyTorch, behind a lazy import that raises a clear
installation hint when the framework is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .accel.networked_kuramoto import networked_kuramoto_force


@dataclass(frozen=True)
class KuramotoOperatorDataset:
    """Branch / trunk / target tensors sampled from true Kuramoto trajectories.

    Attributes
    ----------
    branch_inputs : numpy.ndarray
        The ``(M, 2N)`` initial-condition embeddings ``(cos θ_0, sin θ_0)``.
    trunk_inputs : numpy.ndarray
        The ``(M, 1)`` query times, normalised by the horizon.
    targets : numpy.ndarray
        The ``(M, 2N)`` target embeddings ``(cos θ(t), sin θ(t))``.
    n_oscillators : int
        The number of oscillators ``N``.
    horizon : float
        The trajectory horizon ``T`` (the un-normalised time of the final sample).
    """

    branch_inputs: NDArray[np.float64]
    trunk_inputs: NDArray[np.float64]
    targets: NDArray[np.float64]
    n_oscillators: int
    horizon: float


@dataclass(frozen=True)
class TrainedKuramotoOperator:
    """A trained Kuramoto DeepONet surrogate.

    Attributes
    ----------
    model : Any
        The trained PyTorch module.
    loss_history : numpy.ndarray
        The ``(epochs,)`` training-loss curve.
    n_oscillators : int
        The number of oscillators ``N``.
    horizon : float
        The horizon ``T`` the trunk times were normalised by.
    """

    model: Any
    loss_history: NDArray[np.float64]
    n_oscillators: int
    horizon: float

    def forecast(
        self, initial_phases: NDArray[np.float64], times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Forecast ``θ(t)`` at the requested times for a new initial condition.

        Parameters
        ----------
        initial_phases : numpy.ndarray
            The initial phases ``θ(0)`` (length ``N``).
        times : numpy.ndarray
            The query times (length ``Q``); normalised internally by the horizon.

        Returns
        -------
        numpy.ndarray
            The ``(Q, N)`` forecast phases, reconstructed from the predicted embedding.
        """
        torch = _require_torch()
        phases = np.ascontiguousarray(initial_phases, dtype=np.float64)
        query = np.ascontiguousarray(times, dtype=np.float64)
        if phases.shape != (self.n_oscillators,):
            raise ValueError(f"initial_phases must have length {self.n_oscillators}")
        if query.ndim != 1 or query.size < 1:
            raise ValueError("times must be a non-empty one-dimensional array")
        embedding = np.concatenate([np.cos(phases), np.sin(phases)])
        branch = torch.tensor(
            np.repeat(embedding[None, :], query.size, axis=0), dtype=torch.float32
        )
        trunk = torch.tensor((query / self.horizon)[:, None], dtype=torch.float32)
        with torch.no_grad():
            predicted = self.model(branch, trunk).cpu().numpy()
        cosines = predicted[:, : self.n_oscillators]
        sines = predicted[:, self.n_oscillators :]
        return np.asarray(np.arctan2(sines, cosines), dtype=np.float64)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without the optional extra
        raise ImportError(
            "the Kuramoto neural operator requires PyTorch; install oscillatools[torch]"
        ) from error
    return torch


def simulate_operator_dataset(
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    n_trajectories: int,
    dt: float,
    n_steps: int,
    seed: int,
) -> KuramotoOperatorDataset:
    r"""Sample a DeepONet training set from true Kuramoto trajectories (pure NumPy, no PyTorch).

    Integrates the network from ``n_trajectories`` random initial conditions by RK4 and records, for
    every sampled time, the embedded ``(θ_0, t, θ(t))`` triples.

    Parameters
    ----------
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    n_trajectories : int
        The number of random initial conditions (``≥ 1``).
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The number of steps per trajectory (``≥ 1``); the horizon is ``dt · n_steps``.
    seed : int
        The random seed for the initial conditions.

    Returns
    -------
    KuramotoOperatorDataset
        The branch / trunk / target tensors.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("omega must be a one-dimensional array of length at least two")
    count = frequencies.size
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if not (np.all(np.isfinite(frequencies)) and np.all(np.isfinite(matrix))):
        raise ValueError("omega and coupling must be finite")
    if n_trajectories < 1:
        raise ValueError(f"n_trajectories must be positive, got {n_trajectories}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    rng = np.random.default_rng(seed)
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    horizon = float(times[-1])
    branch: list[NDArray[np.float64]] = []
    trunk: list[float] = []
    targets: list[NDArray[np.float64]] = []
    for _ in range(n_trajectories):
        initial = rng.uniform(0.0, 2.0 * np.pi, size=count)
        embedding = np.concatenate([np.cos(initial), np.sin(initial)])
        state = initial
        for step in range(n_steps + 1):
            branch.append(embedding)
            trunk.append(times[step] / horizon)
            targets.append(np.concatenate([np.cos(state), np.sin(state)]))
            if step < n_steps:
                state = _rk4_step(state, frequencies, matrix, dt)
    return KuramotoOperatorDataset(
        branch_inputs=np.asarray(branch, dtype=np.float64),
        trunk_inputs=np.asarray(trunk, dtype=np.float64)[:, None],
        targets=np.asarray(targets, dtype=np.float64),
        n_oscillators=count,
        horizon=horizon,
    )


def _rk4_step(
    state: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    def field(angle: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(omega + networked_kuramoto_force(angle, coupling), dtype=np.float64)

    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return np.asarray(state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)


def _build_deeponet(n_oscillators: int, latent_dim: int, hidden_dim: int) -> Any:
    torch = _require_torch()
    channels = 2 * n_oscillators

    class _DeepONet(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.branch = torch.nn.Sequential(
                torch.nn.Linear(channels, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, channels * latent_dim),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(1, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, latent_dim),
            )
            self.bias = torch.nn.Parameter(torch.zeros(channels))

        def forward(self, branch_input: Any, trunk_input: Any) -> Any:
            basis = self.branch(branch_input).view(-1, channels, latent_dim)
            weights = self.trunk(trunk_input)
            return torch.einsum("bcp,bp->bc", basis, weights) + self.bias

    return _DeepONet()


def train_kuramoto_neural_operator(
    dataset: KuramotoOperatorDataset,
    *,
    latent_dim: int = 24,
    hidden_dim: int = 64,
    epochs: int = 250,
    learning_rate: float = 3e-3,
    seed: int = 0,
) -> TrainedKuramotoOperator:
    r"""Train a Kuramoto DeepONet surrogate on a dataset (requires PyTorch).

    Parameters
    ----------
    dataset : KuramotoOperatorDataset
        The branch / trunk / target tensors from :func:`simulate_operator_dataset`.
    latent_dim : int
        The shared branch/trunk latent width ``p`` (``≥ 1``).
    hidden_dim : int
        The hidden width of the branch and trunk networks (``≥ 1``).
    epochs : int
        The number of full-batch Adam epochs (``≥ 1``).
    learning_rate : float
        The Adam learning rate (``> 0``).
    seed : int
        The PyTorch manual seed.

    Returns
    -------
    TrainedKuramotoOperator
        The trained model and its loss history.

    Raises
    ------
    ValueError
        If any hyper-parameter falls outside its documented bound.
    ImportError
        If PyTorch is not installed.
    """
    if latent_dim < 1 or hidden_dim < 1:
        raise ValueError("latent_dim and hidden_dim must be positive")
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    torch = _require_torch()
    torch.manual_seed(seed)
    model = _build_deeponet(dataset.n_oscillators, latent_dim, hidden_dim)
    branch = torch.tensor(dataset.branch_inputs, dtype=torch.float32)
    trunk = torch.tensor(dataset.trunk_inputs, dtype=torch.float32)
    targets = torch.tensor(dataset.targets, dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()
    history = np.empty(epochs, dtype=np.float64)
    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = loss_function(model(branch, trunk), targets)
        loss.backward()
        optimiser.step()
        history[epoch] = float(loss.item())
    return TrainedKuramotoOperator(
        model=model,
        loss_history=history,
        n_oscillators=dataset.n_oscillators,
        horizon=dataset.horizon,
    )


__all__ = [
    "KuramotoOperatorDataset",
    "TrainedKuramotoOperator",
    "simulate_operator_dataset",
    "train_kuramoto_neural_operator",
]
