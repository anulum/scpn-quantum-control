# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Host-independent cost model: direct RK4 vs neural-operator surrogate
r"""Host-independent operation-count model for the Kuramoto neural-operator surrogate.

A wall-clock timing depends on the host, the governor, the BLAS backend and the accelerator tier, so
it is never a portable claim. The *operation count*, by contrast, is fixed by the algorithm alone.
This module counts the arithmetic of the two ways to answer "what is :math:`\theta(t)` for this
network and this initial condition?":

* **Direct simulation** integrates the networked Kuramoto right-hand side by fixed-step RK4. Reaching a
  horizon of ``n_steps`` steps costs ``4 · n_steps`` right-hand-side evaluations (RK4 has four stages),
  and each right-hand-side evaluation of the dense all-to-all force
  :math:`f_i = \omega_i + \sum_j K_{ij}\,\sin(\theta_j - \theta_i)` is :math:`O(N^2)`.
* **The DeepONet surrogate** (:mod:`.kuramoto_neural_operator`) maps :math:`(\theta_0, t) \mapsto
  \theta(t)` in a *single* forward pass for *any* query time — it does not step. Its cost is
  :math:`O(N \cdot \text{hidden} \cdot \text{latent})`, independent of how far into the horizon the
  query lands.

Two consequences are host-independent and exact (no timing, no FLOP model):

* **Random access.** Direct simulation must traverse every intermediate step to reach a far time; the
  surrogate answers in one pass. So a single query at the horizon replaces ``4 · n_steps``
  right-hand-side evaluations with one forward pass — :func:`rk4_right_hand_side_evaluations`.
* **Asymptotic scaling.** The per-query direct cost grows like ``n_steps · N²`` while the surrogate's
  grows like ``N · hidden · latent``; for a large enough ``n_steps · N`` the surrogate's per-query
  arithmetic is strictly smaller.

A concrete crossover requires a FLOP model, and every modelling choice here is stated explicitly so
the arithmetic is reproducible: a matrix multiply of an ``in``-vector to an ``out``-vector is counted
as ``2 · in · out`` floating-point operations (one multiply and one add per multiply–accumulate); a
transcendental (``sin``) and each elementary ``+``/``-``/``*`` count as one operation; a ``tanh``
activation counts as one operation per element. These are the conventional operation-count
assumptions used to compare neural surrogates against numerical integrators; they are a transparent
model, not a measurement, and the module labels them as such.

The surrogate pays a **one-time training cost** — the RK4 rollouts that build the dataset plus the
Adam epochs — before any inference. :func:`amortised_break_even_queries` reports the number of
inference queries beyond which the trained surrogate's total operation count (training included) falls
below repeated direct simulation, or reports that no such crossover exists for the given
configuration. This is the honest efficiency statement: the surrogate wins by amortisation and by
random access at scale, not by a single-query wall-clock margin at small ``N``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# --- Operation-count model constants (stated so the arithmetic is reproducible) ---

#: RK4 evaluates the right-hand side four times per step (the four stage slopes).
RK4_STAGES_PER_STEP = 4

#: Floating-point operations charged per multiply–accumulate in a matrix multiply (one ``*`` + one ``+``).
MATMUL_FLOPS_PER_MULTIPLY_ACCUMULATE = 2

#: Operations charged per oscillator pair in one dense force evaluation: one phase difference (``-``),
#: one ``sin``, one multiply by the coupling weight, and one accumulation into the row sum.
FORCE_FLOPS_PER_OSCILLATOR_PAIR = 4

#: Operations charged per oscillator for the RK4 stage combination — building the three intermediate
#: states ``x + c·dt·k`` and the final ``x + dt/6·(k1 + 2k2 + 2k3 + k4)`` (a documented modelling
#: constant; the dominant RK4 cost is the four :math:`O(N^2)` force evaluations, not this :math:`O(N)`
#: term).
RK4_COMBINATION_FLOPS_PER_OSCILLATOR = 14

#: A backward pass costs roughly twice a forward pass, so one training step (forward + backward) is
#: charged three forward passes (a standard operation-count convention for gradient training).
TRAINING_FORWARD_BACKWARD_FACTOR = 3


def _require_positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def rk4_right_hand_side_evaluations(n_steps: int) -> int:
    """Return the number of right-hand-side evaluations to integrate one initial condition to ``n_steps``.

    This is the model-free, exact operation the surrogate eliminates: fixed-step RK4 evaluates the
    Kuramoto force ``RK4_STAGES_PER_STEP`` times per step, so reaching the horizon costs
    ``4 · n_steps`` force evaluations, whereas the surrogate reaches any query time in a single forward
    pass (zero force evaluations).

    Parameters
    ----------
    n_steps : int
        The number of RK4 steps to the horizon (``≥ 1``).

    Returns
    -------
    int
        ``RK4_STAGES_PER_STEP · n_steps``.

    Raises
    ------
    ValueError
        If ``n_steps`` is not positive.
    """
    return RK4_STAGES_PER_STEP * _require_positive("n_steps", n_steps)


def networked_force_flops(n_oscillators: int) -> int:
    r"""Return the modelled FLOP count of one dense all-to-all Kuramoto force evaluation.

    The force :math:`f_i = \omega_i + \sum_j K_{ij}\,\sin(\theta_j - \theta_i)` charges
    ``FORCE_FLOPS_PER_OSCILLATOR_PAIR`` operations for each of the :math:`N^2` ordered oscillator pairs
    (difference, ``sin``, coupling multiply, accumulation) plus one addition per oscillator for the
    natural frequency.

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ 1``).

    Returns
    -------
    int
        ``FORCE_FLOPS_PER_OSCILLATOR_PAIR · N² + N``.

    Raises
    ------
    ValueError
        If ``n_oscillators`` is not positive.
    """
    count = _require_positive("n_oscillators", n_oscillators)
    return FORCE_FLOPS_PER_OSCILLATOR_PAIR * count * count + count


def rk4_step_flops(n_oscillators: int) -> int:
    """Return the modelled FLOP count of one RK4 step of the networked Kuramoto system.

    One step is four force evaluations plus the :math:`O(N)` stage combination.

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ 1``).

    Returns
    -------
    int
        ``RK4_STAGES_PER_STEP · networked_force_flops(N) + RK4_COMBINATION_FLOPS_PER_OSCILLATOR · N``.

    Raises
    ------
    ValueError
        If ``n_oscillators`` is not positive.
    """
    count = _require_positive("n_oscillators", n_oscillators)
    return (
        RK4_STAGES_PER_STEP * networked_force_flops(count)
        + RK4_COMBINATION_FLOPS_PER_OSCILLATOR * count
    )


def direct_simulation_flops(n_oscillators: int, n_steps: int) -> int:
    """Return the modelled FLOP count to integrate one initial condition to the horizon by RK4.

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ 1``).
    n_steps : int
        The number of RK4 steps to the horizon (``≥ 1``).

    Returns
    -------
    int
        ``n_steps · rk4_step_flops(N)``.

    Raises
    ------
    ValueError
        If either argument is not positive.
    """
    steps = _require_positive("n_steps", n_steps)
    return steps * rk4_step_flops(n_oscillators)


def deeponet_forward_flops(n_oscillators: int, latent_dim: int, hidden_dim: int) -> int:
    r"""Return the modelled FLOP count of one DeepONet forward pass (a single-time forecast).

    The surrogate embeds the initial condition on the phase circle as ``channels = 2N`` values. The
    branch network is ``Linear(2N, hidden) → tanh → Linear(hidden, 2N·latent)``, the trunk network is
    ``Linear(1, hidden) → tanh → Linear(hidden, latent)``, their outputs are contracted by an
    ``einsum`` over the latent index, and a per-channel bias is added. Matrix multiplies are charged
    ``MATMUL_FLOPS_PER_MULTIPLY_ACCUMULATE · in · out`` and activations one operation per element.

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ 1``).
    latent_dim : int
        The shared branch/trunk latent width ``p`` (``≥ 1``).
    hidden_dim : int
        The hidden width of the branch and trunk networks (``≥ 1``).

    Returns
    -------
    int
        The summed forward-pass FLOP count.

    Raises
    ------
    ValueError
        If any argument is not positive.
    """
    count = _require_positive("n_oscillators", n_oscillators)
    latent = _require_positive("latent_dim", latent_dim)
    hidden = _require_positive("hidden_dim", hidden_dim)
    channels = 2 * count
    mac = MATMUL_FLOPS_PER_MULTIPLY_ACCUMULATE
    branch = mac * channels * hidden + hidden + mac * hidden * channels * latent
    trunk = mac * 1 * hidden + hidden + mac * hidden * latent
    contraction = mac * channels * latent
    bias = channels
    return branch + trunk + contraction + bias


def training_flops(
    n_oscillators: int,
    *,
    n_steps: int,
    n_trajectories: int,
    epochs: int,
    latent_dim: int,
    hidden_dim: int,
) -> int:
    """Return the modelled one-time training FLOP count of the surrogate.

    Training first builds the dataset by integrating ``n_trajectories`` initial conditions to the
    horizon (``n_trajectories · n_steps`` RK4 steps), then runs ``epochs`` full-batch Adam passes over
    all ``n_trajectories · (n_steps + 1)`` samples, each pass charged
    ``TRAINING_FORWARD_BACKWARD_FACTOR`` forward passes for the forward-plus-backward gradient step.

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ 1``).
    n_steps : int
        The number of RK4 steps to the horizon (``≥ 1``).
    n_trajectories : int
        The number of training trajectories (``≥ 1``).
    epochs : int
        The number of full-batch training epochs (``≥ 1``).
    latent_dim : int
        The latent width ``p`` (``≥ 1``).
    hidden_dim : int
        The hidden width (``≥ 1``).

    Returns
    -------
    int
        The dataset-generation FLOPs plus the optimisation FLOPs.

    Raises
    ------
    ValueError
        If any argument is not positive.
    """
    steps = _require_positive("n_steps", n_steps)
    trajectories = _require_positive("n_trajectories", n_trajectories)
    passes = _require_positive("epochs", epochs)
    dataset_flops = trajectories * steps * rk4_step_flops(n_oscillators)
    samples = trajectories * (steps + 1)
    forward = deeponet_forward_flops(n_oscillators, latent_dim, hidden_dim)
    optimisation_flops = passes * samples * TRAINING_FORWARD_BACKWARD_FACTOR * forward
    return dataset_flops + optimisation_flops


def amortised_break_even_queries(
    training_flops_total: int,
    direct_flops_per_query: int,
    surrogate_flops_per_query: int,
) -> int | None:
    """Return the query count beyond which the trained surrogate is cheaper than direct simulation.

    Each direct query costs ``direct_flops_per_query``; each surrogate query costs
    ``surrogate_flops_per_query`` after the one-time ``training_flops_total``. The surrogate's total
    over ``B`` queries is ``training_flops_total + B · surrogate_flops_per_query`` and beats direct
    simulation's ``B · direct_flops_per_query`` once ``B`` exceeds
    ``training_flops_total / (direct_flops_per_query - surrogate_flops_per_query)``.

    Parameters
    ----------
    training_flops_total : int
        The one-time training FLOP count (``≥ 0``).
    direct_flops_per_query : int
        The per-query FLOP count of direct simulation (``≥ 0``).
    surrogate_flops_per_query : int
        The per-query FLOP count of the surrogate (``≥ 0``).

    Returns
    -------
    int or None
        The smallest positive query count at which the surrogate's cumulative cost is strictly lower,
        or ``None`` when the surrogate is not cheaper per query (no crossover exists).

    Raises
    ------
    ValueError
        If any argument is negative.
    """
    if training_flops_total < 0:
        raise ValueError("training_flops_total must be non-negative")
    if direct_flops_per_query < 0 or surrogate_flops_per_query < 0:
        raise ValueError("per-query FLOP counts must be non-negative")
    margin = direct_flops_per_query - surrogate_flops_per_query
    if margin <= 0:
        return None
    return int(math.floor(training_flops_total / margin)) + 1


@dataclass(frozen=True)
class SurrogateCostModel:
    """The full host-independent cost picture for one surrogate-vs-direct configuration.

    Attributes
    ----------
    n_oscillators : int
        The number of oscillators ``N``.
    n_steps : int
        The number of RK4 steps to the horizon.
    latent_dim : int
        The DeepONet latent width ``p``.
    hidden_dim : int
        The DeepONet hidden width.
    n_trajectories : int
        The number of training trajectories.
    epochs : int
        The number of training epochs.
    rk4_right_hand_side_evaluations : int
        Force evaluations the surrogate eliminates per query (model-free).
    direct_flops_per_query : int
        Modelled FLOPs to integrate one initial condition to the horizon.
    surrogate_flops_per_query : int
        Modelled FLOPs of one DeepONet forward pass.
    per_query_flop_ratio : float
        ``direct_flops_per_query / surrogate_flops_per_query`` — the per-query operation-count factor.
    training_flops : int
        The one-time training FLOP count.
    break_even_queries : int or None
        Queries beyond which the trained surrogate is cheaper overall, or ``None`` if never.
    """

    n_oscillators: int
    n_steps: int
    latent_dim: int
    hidden_dim: int
    n_trajectories: int
    epochs: int
    rk4_right_hand_side_evaluations: int
    direct_flops_per_query: int
    surrogate_flops_per_query: int
    per_query_flop_ratio: float
    training_flops: int
    break_even_queries: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the cost model."""
        return {
            "n_oscillators": self.n_oscillators,
            "n_steps": self.n_steps,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_trajectories": self.n_trajectories,
            "epochs": self.epochs,
            "rk4_right_hand_side_evaluations": self.rk4_right_hand_side_evaluations,
            "direct_flops_per_query": self.direct_flops_per_query,
            "surrogate_flops_per_query": self.surrogate_flops_per_query,
            "per_query_flop_ratio": self.per_query_flop_ratio,
            "training_flops": self.training_flops,
            "break_even_queries": self.break_even_queries,
        }


def build_cost_model(
    n_oscillators: int,
    *,
    n_steps: int,
    latent_dim: int,
    hidden_dim: int,
    n_trajectories: int,
    epochs: int,
) -> SurrogateCostModel:
    """Assemble the :class:`SurrogateCostModel` for one configuration.

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ 1``).
    n_steps : int
        The number of RK4 steps to the horizon (``≥ 1``).
    latent_dim : int
        The DeepONet latent width ``p`` (``≥ 1``).
    hidden_dim : int
        The DeepONet hidden width (``≥ 1``).
    n_trajectories : int
        The number of training trajectories (``≥ 1``).
    epochs : int
        The number of training epochs (``≥ 1``).

    Returns
    -------
    SurrogateCostModel
        Every count for this configuration, including the amortised break-even.

    Raises
    ------
    ValueError
        If any argument is not positive.
    """
    direct = direct_simulation_flops(n_oscillators, n_steps)
    surrogate = deeponet_forward_flops(n_oscillators, latent_dim, hidden_dim)
    total_training = training_flops(
        n_oscillators,
        n_steps=n_steps,
        n_trajectories=n_trajectories,
        epochs=epochs,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    return SurrogateCostModel(
        n_oscillators=n_oscillators,
        n_steps=n_steps,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_trajectories=n_trajectories,
        epochs=epochs,
        rk4_right_hand_side_evaluations=rk4_right_hand_side_evaluations(n_steps),
        direct_flops_per_query=direct,
        surrogate_flops_per_query=surrogate,
        per_query_flop_ratio=direct / surrogate,
        training_flops=total_training,
        break_even_queries=amortised_break_even_queries(total_training, direct, surrogate),
    )


__all__ = [
    "FORCE_FLOPS_PER_OSCILLATOR_PAIR",
    "MATMUL_FLOPS_PER_MULTIPLY_ACCUMULATE",
    "RK4_COMBINATION_FLOPS_PER_OSCILLATOR",
    "RK4_STAGES_PER_STEP",
    "TRAINING_FORWARD_BACKWARD_FACTOR",
    "SurrogateCostModel",
    "amortised_break_even_queries",
    "build_cost_model",
    "deeponet_forward_flops",
    "direct_simulation_flops",
    "networked_force_flops",
    "rk4_right_hand_side_evaluations",
    "rk4_step_flops",
    "training_flops",
]
