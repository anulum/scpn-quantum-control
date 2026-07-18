# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA probe orchestration
"""Train, evaluate, and aggregate the KYMA composition probe over seeds.

Implements the frozen pre-registration pass/fail contract: held-out-conjunction
accuracy PASSES iff it is ≥ 70 %, at least 25 percentage points above the
parameter-matched MLP baseline, and clearly above the measured chance floor,
reproduced over five seeds (mean ± sd). No post-hoc metric shopping.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import models
from .task import ProbeConfig, TrialBatch, build_trials, success_mask

# Frozen pre-registration thresholds.
PASS_ACCURACY = 0.70
PASS_MARGIN_PP = 0.25
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
# Training budget (both models trained to their plateau on this toy task).
SUBSTRATE_EPOCHS = 1000
MLP_EPOCHS = 1500
# Energy proxy: nominal package power for the wall-time → joules conversion.
# Stated assumption, not a measured wattmeter reading.
NOMINAL_POWER_W = 15.0


@dataclass
class SeedResult:
    """One seed's measured outcome."""

    seed: int
    substrate_accuracy: float
    mlp_accuracy: float
    chance_accuracy: float
    substrate_params: int
    mlp_params: int
    mlp_hidden: int
    substrate_j_per_task: float
    mlp_j_per_task: float
    substrate_steps: int


def _test_slice(batch: TrialBatch) -> TrialBatch:
    m = batch.is_test
    return TrialBatch(
        batch.theta0[m], batch.code[m], batch.r1_pair[m], batch.r2_pair[m], batch.is_test[m]
    )


def _substrate_test_accuracy(
    params: dict[str, jax.Array], batch: TrialBatch, config: ProbeConfig
) -> float:
    test = _test_slice(batch)
    r1m, r2m, _, _ = models._member_tables(test)
    r1, r2 = models.substrate_readout(
        params, jnp.asarray(test.theta0), jnp.asarray(test.code), r1m, r2m, config
    )
    return float(np.mean(success_mask(np.asarray(r1), np.asarray(r2), config.epsilon)))


def _mlp_test_accuracy(
    params: dict[str, jax.Array], batch: TrialBatch, config: ProbeConfig
) -> float:
    test = _test_slice(batch)
    feats = models._mlp_features(test.theta0, test.code)
    pred = np.asarray(models.mlp_forward(params, feats))
    return float(np.mean(success_mask(pred[:, 0], pred[:, 1], config.epsilon)))


def _measure_j_per_task_substrate(
    params: dict[str, jax.Array], batch: TrialBatch, config: ProbeConfig
) -> float:
    """steps × measured per-step wall-cost + energy proxy for one substrate task."""
    test = _test_slice(batch)
    one = TrialBatch(
        test.theta0[:1], test.code[:1], test.r1_pair[:1], test.r2_pair[:1], test.is_test[:1]
    )
    r1m, r2m, _, _ = models._member_tables(one)
    t0 = jnp.asarray(one.theta0)
    code = jnp.asarray(one.code)

    def run() -> None:
        r1, _ = models.substrate_readout(params, t0, code, r1m, r2m, config)
        r1.block_until_ready()

    run()  # warm up JIT
    reps = 20
    start = time.perf_counter()
    for _ in range(reps):
        run()
    wall = (time.perf_counter() - start) / reps
    return wall * NOMINAL_POWER_W


def _measure_j_per_task_mlp(params: dict[str, jax.Array], batch: TrialBatch) -> float:
    test = _test_slice(batch)
    feats = models._mlp_features(test.theta0[:1], test.code[:1])

    def run() -> None:
        models.mlp_forward(params, feats).block_until_ready()

    run()
    reps = 50
    start = time.perf_counter()
    for _ in range(reps):
        run()
    wall = (time.perf_counter() - start) / reps
    return wall * NOMINAL_POWER_W


def run_seed(seed: int, config: ProbeConfig | None = None) -> SeedResult:
    """Train + evaluate substrate, MLP, and chance floor for one seed."""
    config = config or ProbeConfig()
    batch = build_trials(config, seed)

    substrate = models.train_substrate(batch, config, seed, epochs=SUBSTRATE_EPOCHS)
    sub_acc = _substrate_test_accuracy(substrate, batch, config)

    n_sub = models.substrate_param_count()
    hidden = models.mlp_hidden_for_match(n_sub)
    mlp = models.train_mlp(batch, hidden, seed, epochs=MLP_EPOCHS)
    mlp_acc = _mlp_test_accuracy(mlp, batch, config)

    chance = models.chance_floor_accuracy(batch, config.epsilon, seed)

    return SeedResult(
        seed=seed,
        substrate_accuracy=sub_acc,
        mlp_accuracy=mlp_acc,
        chance_accuracy=chance,
        substrate_params=n_sub,
        mlp_params=models.mlp_param_count(hidden),
        mlp_hidden=hidden,
        substrate_j_per_task=_measure_j_per_task_substrate(substrate, batch, config),
        mlp_j_per_task=_measure_j_per_task_mlp(mlp, batch),
        substrate_steps=config.steps,
    )


def run_probe(
    seeds: tuple[int, ...] = DEFAULT_SEEDS, config: ProbeConfig | None = None
) -> dict[str, Any]:
    """Run all seeds and evaluate the frozen pass/fail contract."""
    config = config or ProbeConfig()
    results = [run_seed(s, config) for s in seeds]

    def stats(values: list[float]) -> dict[str, float]:
        return {"mean": mean(values), "sd": pstdev(values)}

    sub = stats([r.substrate_accuracy for r in results])
    mlp = stats([r.mlp_accuracy for r in results])
    chance = stats([r.chance_accuracy for r in results])
    margin = sub["mean"] - mlp["mean"]

    passed = (
        sub["mean"] >= PASS_ACCURACY and margin >= PASS_MARGIN_PP and sub["mean"] > chance["mean"]
    )
    return {
        "config": asdict(config),
        "seeds": list(seeds),
        "substrate_accuracy": sub,
        "mlp_accuracy": mlp,
        "chance_accuracy": chance,
        "margin_over_mlp_pp": margin,
        "thresholds": {
            "pass_accuracy": PASS_ACCURACY,
            "pass_margin_pp": PASS_MARGIN_PP,
        },
        "verdict": "PASS" if passed else "NEGATIVE",
        "substrate_j_per_task": stats([r.substrate_j_per_task for r in results]),
        "mlp_j_per_task": stats([r.mlp_j_per_task for r in results]),
        "energy_proxy_watts": NOMINAL_POWER_W,
        "param_counts": {
            "substrate": results[0].substrate_params,
            "mlp": results[0].mlp_params,
            "mlp_hidden": results[0].mlp_hidden,
        },
        "per_seed": [asdict(r) for r in results],
    }
