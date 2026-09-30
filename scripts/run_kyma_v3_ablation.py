# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 dynamics ablation runner
"""Run the pre-registered KYMA v3 dynamics ablation and write the artefact.

The KYMA v3 probe (PASS) could not separate the oscillator dynamics from the other
architectural priors of the substrate. This runner trains four variants of the
frozen substrate on the frozen split with the frozen training schedule and reports
their held-out accuracy:

``full``
    the frozen substrate, unchanged (control; must reproduce the v3 result);
``phasor``
    no dynamics: each write stage sets the destination phase directly to the stable
    fixed point of the frozen stage dynamics, ``θ_i = arg(Σ_j K e^{i(x_j+α)} +
    Σ_p T e^{i(x_{p1}+x_{p2}+β)})``, with the same 108 gates;
``pairwise``
    the frozen dynamics without the triadic term (54 gates; the triadic gates are
    held at zero and not trained);
``short``
    the frozen dynamics integrated for 5 RK4 steps per stage instead of 60.

Contract, interpretation rules and seeds are frozen in
``docs/campaigns/kyma_v3_dynamics_ablation_prereg_2026-09-30.md``. The frozen v3
modules are imported, never modified. 0 QPU.

Usage::

    python scripts/run_kyma_v3_ablation.py --commit SHA [--out PATH] [--variants ...]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks.kyma_v2.models import _adam_descent
from scpn_quantum_control.benchmarks.kyma_v3 import probe, substrate
from scpn_quantum_control.benchmarks.kyma_v3.task import (
    NO_OPERATION,
    SymbolicDataset,
    all_states,
    build_dataset,
    configurations,
    run_configuration,
)

VARIANTS = ("full", "phasor", "pairwise", "short")
SHORT_STEPS = 5
ATTRIBUTION_MARGIN = 0.10
_DEFAULT_OUT = Path("data/kyma_v3_symbolic_composition/kyma_v3_dynamics_ablation.json")
_DEFAULT_V3 = Path("data/kyma_v3_symbolic_composition/kyma_v3_symbolic_composition.json")
_PREREGISTRATION = "docs/campaigns/kyma_v3_dynamics_ablation_prereg_2026-09-30.md"
_TRIADIC = ("triadic", "triadic_lag")

_Params = dict[str, jax.Array]
_Stage = Callable[[_Params, jax.Array, jax.Array], jax.Array]


def integrate_stage(
    gates: _Params, operation: jax.Array, source: jax.Array, steps: int
) -> jax.Array:
    """Integrate one write stage of the frozen dynamics for ``steps`` RK4 steps.

    The right-hand side, step size and reset phase are those of the frozen
    substrate; only the number of steps is a parameter.

    Parameters
    ----------
    gates
        Gate tensors ``coupling``, ``lag``, ``triadic``, ``triadic_lag``.
    operation
        ``(n,)`` operation index per item.
    source
        ``(n, 3)`` phases of the held source bank.
    steps
        Number of RK4 steps of size ``substrate.DT``.

    Returns
    -------
    jax.Array
        ``(n, 3)`` destination phases.

    """
    coupling = gates["coupling"][operation]
    lag = gates["lag"][operation]
    triadic = gates["triadic"][operation]
    triadic_lag = gates["triadic_lag"][operation]
    pair_sum = jnp.stack([source[:, p] + source[:, q] for p, q in substrate.PAIRS], axis=1)

    def rhs(theta: jax.Array) -> jax.Array:
        pairwise = coupling * jnp.sin(source[:, None, :] - theta[:, :, None] + lag)
        three = triadic * jnp.sin(pair_sum[:, None, :] - theta[:, :, None] + triadic_lag)
        return jnp.sum(pairwise, axis=2) + jnp.sum(three, axis=2)

    def step(theta: jax.Array, _: None) -> tuple[jax.Array, None]:
        dt = substrate.DT
        k1 = rhs(theta)
        k2 = rhs(theta + 0.5 * dt * k1)
        k3 = rhs(theta + 0.5 * dt * k2)
        k4 = rhs(theta + dt * k3)
        return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    start = jnp.full_like(source, substrate.RESET_PHASE)
    final, _ = jax.lax.scan(step, start, None, length=steps)
    return final


def phasor_stage(gates: _Params, operation: jax.Array, source: jax.Array) -> jax.Array:
    """Set each destination phase to the stable fixed point of the stage dynamics.

    The frozen right-hand side equals ``|Z| sin(arg Z − θ)`` with
    ``Z = Σ_j K e^{i(x_j+α)} + Σ_p T e^{i(x_{p1}+x_{p2}+β)}``, whose stable fixed point is
    ``θ = arg Z``. No integration takes place.

    Parameters
    ----------
    gates
        Gate tensors.
    operation
        ``(n,)`` operation index per item.
    source
        ``(n, 3)`` phases of the held source bank.

    Returns
    -------
    jax.Array
        ``(n, 3)`` destination phases in ``(−π, π]``.

    """
    coupling = gates["coupling"][operation]
    lag = gates["lag"][operation]
    triadic = gates["triadic"][operation]
    triadic_lag = gates["triadic_lag"][operation]
    pair_sum = jnp.stack([source[:, p] + source[:, q] for p, q in substrate.PAIRS], axis=1)
    pair_angle = source[:, None, :] + lag
    three_angle = pair_sum[:, None, :] + triadic_lag
    real = jnp.sum(coupling * jnp.cos(pair_angle), axis=2) + jnp.sum(
        triadic * jnp.cos(three_angle), axis=2
    )
    imag = jnp.sum(coupling * jnp.sin(pair_angle), axis=2) + jnp.sum(
        triadic * jnp.sin(three_angle), axis=2
    )
    return jnp.arctan2(imag, real)


def _stage_for(variant: str) -> _Stage:
    if variant == "phasor":
        return phasor_stage
    if variant == "short":
        return lambda g, o, s: integrate_stage(g, o, s, SHORT_STEPS)
    if variant in ("full", "pairwise"):
        # The frozen v3 stage itself, so the control is the substrate that passed.
        return substrate._write_stage
    raise ValueError(f"unknown variant {variant!r}")


def _complete(variant: str, trainable: _Params) -> _Params:
    """Add the fixed zero triadic gates of the ``pairwise`` variant."""
    if variant != "pairwise":
        return trainable
    zeros = jnp.zeros_like(trainable["coupling"])
    return {**trainable, "triadic": zeros, "triadic_lag": zeros}


def final_phases(
    variant: str,
    gates: _Params,
    states: jax.Array,
    first_op: jax.Array,
    second_op: jax.Array,
) -> jax.Array:
    """Run each item's program under ``variant`` and return the final phases.

    Parameters
    ----------
    variant
        One of :data:`VARIANTS`.
    gates
        Trainable gate tensors of the variant.
    states
        ``(n, 3)`` initial register values.
    first_op, second_op
        ``(n,)`` operation indices; ``second_op == NO_OPERATION`` skips stage two.

    Returns
    -------
    jax.Array
        ``(n, 3)`` final phases.

    """
    stage = _stage_for(variant)
    full_gates = _complete(variant, gates)
    start = states.astype(jnp.float32) * substrate.PHASE_STEP
    after_first = stage(full_gates, first_op, start)
    second = jnp.where(second_op == NO_OPERATION, 0, second_op)
    after_second = stage(full_gates, second, after_first)
    return jnp.where((second_op == NO_OPERATION)[:, None], after_first, after_second)


def init_gates(variant: str, seed: int) -> _Params:
    """Draw the frozen initial gates, dropping the triadic ones for ``pairwise``.

    Parameters
    ----------
    variant
        One of :data:`VARIANTS`.
    seed
        Initialisation seed.

    Returns
    -------
    dict
        Trainable gate tensors.

    """
    gates = substrate.init_gates(seed)
    if variant == "pairwise":
        return {name: value for name, value in gates.items() if name not in _TRIADIC}
    return gates


def param_count(variant: str) -> int:
    """Return the number of trainable parameters of ``variant``.

    Parameters
    ----------
    variant
        One of :data:`VARIANTS`.

    Returns
    -------
    int
        108, or 54 for ``pairwise``.

    """
    return sum(int(value.size) for value in init_gates(variant, 0).values())


def _queried(phases: jax.Array, query: jax.Array) -> jax.Array:
    return jnp.take_along_axis(phases, query[:, None], axis=1)[:, 0]


def predict(
    variant: str, gates: _Params, dataset: SymbolicDataset, mask: NDArray[np.bool_]
) -> NDArray[np.int64]:
    """Predict labels of the masked items under ``variant``.

    Parameters
    ----------
    variant
        One of :data:`VARIANTS`.
    gates
        Trainable gate tensors.
    dataset
        The frozen split.
    mask
        Items to predict.

    Returns
    -------
    numpy.ndarray
        Predicted labels.

    """
    phases = final_phases(
        variant,
        gates,
        jnp.asarray(dataset.states[mask]),
        jnp.asarray(dataset.first_op[mask]),
        jnp.asarray(dataset.second_op[mask]),
    )
    labels = substrate.phase_to_label(_queried(phases, jnp.asarray(dataset.query[mask])))
    return np.asarray(labels, dtype=np.int64)


def train(
    variant: str, dataset: SymbolicDataset, seed: int, epochs: int, learning_rate: float
) -> _Params:
    """Train ``variant`` with the frozen loss and full-batch Adam schedule.

    Parameters
    ----------
    variant
        One of :data:`VARIANTS`.
    dataset
        The frozen split; only ``~is_test`` items are used.
    seed
        Initialisation seed.
    epochs
        Adam steps.
    learning_rate
        Adam step size.

    Returns
    -------
    dict
        Trained gate tensors.

    """
    train_mask = ~dataset.is_test
    states = jnp.asarray(dataset.states[train_mask])
    first_op = jnp.asarray(dataset.first_op[train_mask])
    second_op = jnp.asarray(dataset.second_op[train_mask])
    query = jnp.asarray(dataset.query[train_mask])
    target = jnp.asarray(dataset.label[train_mask]).astype(jnp.float32) * substrate.PHASE_STEP
    params = init_gates(variant, seed)

    def loss(p: _Params) -> jax.Array:
        phases = _queried(final_phases(variant, p, states, first_op, second_op), query)
        return jnp.mean(1.0 - jnp.cos(phases - target))

    @jax.jit  # type: ignore[untyped-decorator]  # jax.jit is untyped upstream
    def optimise() -> _Params:
        return _adam_descent(params, loss, lr=learning_rate, epochs=epochs)

    return cast(_Params, optimise())


def realisability_accuracy(variant: str) -> float:
    """Accuracy of the hand-set v3 gates under ``variant`` on all 2,304 items.

    Parameters
    ----------
    variant
        One of :data:`VARIANTS`.

    Returns
    -------
    float
        Fraction of the 12 × 64 × 3 items labelled exactly (the ``pairwise`` variant
        uses the hand-set pairwise gates only).

    """
    rows = [
        (state, configuration) for configuration, state in product(configurations(), all_states())
    ]
    states = jnp.asarray(np.array([row[0] for row in rows]))
    first = jnp.asarray(np.array([row[1][0] for row in rows]))
    second = jnp.asarray(
        np.array([row[1][1] if len(row[1]) == 2 else NO_OPERATION for row in rows])
    )
    gates = substrate.hand_gates()
    if variant == "pairwise":
        gates = {name: value for name, value in gates.items() if name not in _TRIADIC}
    phases = final_phases(variant, gates, states, first, second)
    predicted = np.asarray(substrate.phase_to_label(phases))
    truth = np.array([run_configuration(row[0], row[1]) for row in rows])
    return float(np.mean(predicted == truth))


def run_seed(
    dataset: SymbolicDataset, variant: str, seed: int, epochs: int
) -> dict[str, float | int]:
    """Train and evaluate one variant for one seed.

    Parameters
    ----------
    dataset
        The frozen split.
    variant
        One of :data:`VARIANTS`.
    seed
        Initialisation seed.
    epochs
        Adam steps.

    Returns
    -------
    dict
        ``seed``, ``train_accuracy``, ``test_accuracy`` and ``train_seconds``.

    """
    start = time.perf_counter()
    gates = train(variant, dataset, seed, epochs, probe.SUBSTRATE_LEARNING_RATE)
    train_mask = ~dataset.is_test
    train_accuracy = float(
        np.mean(predict(variant, gates, dataset, train_mask) == dataset.label[train_mask])
    )
    test_accuracy = float(
        np.mean(
            predict(variant, gates, dataset, dataset.is_test) == dataset.label[dataset.is_test]
        )
    )
    return {
        "seed": seed,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_seconds": time.perf_counter() - start,
    }


def summarise(
    per_variant: dict[str, list[dict[str, float | int]]],
    v3_substrate_per_seed: dict[int, float] | None,
) -> dict[str, Any]:
    """Apply the frozen interpretation rules to the per-seed results.

    Parameters
    ----------
    per_variant
        Per-seed rows of every variant that was run; must include ``full``.
    v3_substrate_per_seed
        Held-out accuracy per seed of the substrate in the v3 artefact, or ``None``.

    Returns
    -------
    dict
        Mean and population SD per variant, the gap of every variant below
        ``full``, whether it lies within the 10-point attribution margin, and the
        reproduction check of ``full`` against v3.

    Raises
    ------
    ValueError
        If ``full`` was not run.

    """
    if "full" not in per_variant:
        raise ValueError("the full control variant is required")
    stats: dict[str, dict[str, float]] = {}
    for variant, rows in per_variant.items():
        test = [float(row["test_accuracy"]) for row in rows]
        train_values = [float(row["train_accuracy"]) for row in rows]
        stats[variant] = {
            "test_mean": mean(test),
            "test_sd": pstdev(test),
            "train_mean": mean(train_values),
        }
    full_mean = stats["full"]["test_mean"]
    comparisons = {
        variant: {
            "gap_below_full": full_mean - values["test_mean"],
            "within_attribution_margin": full_mean - values["test_mean"] <= ATTRIBUTION_MARGIN,
        }
        for variant, values in stats.items()
        if variant != "full"
    }
    reproduction: dict[str, Any] = {"checked": v3_substrate_per_seed is not None}
    if v3_substrate_per_seed is not None:
        mismatches = [
            int(row["seed"])
            for row in per_variant["full"]
            if v3_substrate_per_seed.get(int(row["seed"])) != row["test_accuracy"]
        ]
        reproduction["mismatched_seeds"] = mismatches
        reproduction["reproduced"] = not mismatches
    return {"stats": stats, "comparisons": comparisons, "reproduction_of_v3": reproduction}


def _v3_per_seed(path: Path) -> dict[int, float] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["seed"]): float(row["substrate_test_accuracy"])
        for row in data["result"]["per_seed"]
    }


def run_ablation(
    variants: Sequence[str], seeds: Sequence[int], epochs: int, v3_artefact: Path
) -> dict[str, Any]:
    """Run every variant over every seed and apply the interpretation rules.

    Parameters
    ----------
    variants
        Variants to run; must include ``full``.
    seeds
        Seeds.
    epochs
        Adam steps per training run.
    v3_artefact
        Path of the v3 artefact used for the reproduction check.

    Returns
    -------
    dict
        Result with schema ``kyma_v3_ablation_result_v1``.

    """
    dataset = build_dataset()
    per_variant = {
        variant: [run_seed(dataset, variant, seed, epochs) for seed in seeds]
        for variant in variants
    }
    return {
        "schema": "kyma_v3_ablation_result_v1",
        "variants": list(variants),
        "seeds": list(seeds),
        "epochs": epochs,
        "learning_rate": probe.SUBSTRATE_LEARNING_RATE,
        "short_steps": SHORT_STEPS,
        "param_count": {variant: param_count(variant) for variant in variants},
        "realisability_accuracy": {
            variant: realisability_accuracy(variant) for variant in variants
        },
        "per_variant": per_variant,
        "summary": summarise(per_variant, _v3_per_seed(v3_artefact)),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ablation and write the artefact.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0).

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--v3-artefact", type=Path, default=_DEFAULT_V3)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(probe.SEEDS))
    parser.add_argument("--epochs", type=int, default=probe.EPOCHS)
    args = parser.parse_args(argv)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = run_ablation(args.variants, args.seeds, args.epochs, args.v3_artefact)
    artefact = {
        "probe": "kyma_v3_dynamics_ablation",
        "pre_registration": _PREREGISTRATION,
        "source_commit": args.commit,
        "host": {
            "node": platform.node(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artefact, indent=2, default=float) + "\n", encoding="utf-8")
    for variant, values in result["summary"]["stats"].items():
        print(f"{variant}: held-out {values['test_mean']:.3f}±{values['test_sd']:.3f}")
    print(f"artefact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
