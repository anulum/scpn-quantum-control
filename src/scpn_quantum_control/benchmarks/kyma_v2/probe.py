# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 probe orchestration
"""Train, evaluate, and aggregate the KYMA v2 composition probe over seeds.

Implements the frozen v2 pre-registration pass/fail contract: held-out-conjunction
**classification accuracy** PASSES iff the gated student is ≥ 60 %, at least 20
percentage points above the parameter-matched MLP baseline, and clearly above the
measured chance floor, reproduced over five seeds (mean ± sd). No post-hoc metric
shopping — the design constants come from :mod:`.design` (teacher-only), not from
any model's test accuracy.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import models, teacher
from .coupling import base_coupling_matrix, partners_for
from .task import ProbeConfigV2, TrialBatchV2, build_trials

# Frozen v2 pre-registration thresholds.
PASS_ACCURACY = 0.60
PASS_MARGIN_PP = 0.20
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
STUDENT_EPOCHS = 1500
MLP_EPOCHS = 2000
# Energy proxy: nominal package power for the wall-time → joules conversion.
NOMINAL_POWER_W = 15.0


@dataclass
class SeedResultV2:
    """One seed's measured outcome."""

    seed: int
    student_accuracy: float
    mlp_accuracy: float
    chance_accuracy: float
    student_params: int
    mlp_params: int
    mlp_hidden: int
    student_j_per_task: float
    mlp_j_per_task: float
    steps: int


def _test_slice(batch: TrialBatchV2) -> TrialBatchV2:
    m = batch.is_test
    return TrialBatchV2(
        batch.theta0[m], batch.code[m], batch.r1_pair[m], batch.r2_pair[m], batch.is_test[m]
    )


def _measure_j_per_task_student(
    params: dict[str, jax.Array], batch: TrialBatchV2, config: ProbeConfigV2
) -> float:
    """steps × measured per-task inference wall-cost × nominal power (one trial)."""
    test = _test_slice(batch)
    one = TrialBatchV2(
        test.theta0[:1], test.code[:1], test.r1_pair[:1], test.r2_pair[:1], test.is_test[:1]
    )
    base = jnp.asarray(
        base_coupling_matrix(
            config.k_ambient, config.k_bridge, partners_for(config.held_out, config.bridge_mode)
        )
    )
    t0 = jnp.asarray(one.theta0)
    code = jnp.asarray(one.code)

    def run() -> None:
        models.student_final_phases(params, t0, code, base, config).block_until_ready()

    run()  # warm up JIT
    reps = 20
    start = time.perf_counter()
    for _ in range(reps):
        run()
    wall = (time.perf_counter() - start) / reps
    return wall * NOMINAL_POWER_W


def _measure_j_per_task_mlp(params: dict[str, jax.Array], batch: TrialBatchV2) -> float:
    test = _test_slice(batch)
    feats = models._mlp_features(test.theta0[:1], test.code[:1])

    def run() -> None:
        models.mlp_logits(params, feats).block_until_ready()

    run()
    reps = 50
    start = time.perf_counter()
    for _ in range(reps):
        run()
    wall = (time.perf_counter() - start) / reps
    return wall * NOMINAL_POWER_W


def run_seed(seed: int, config: ProbeConfigV2) -> SeedResultV2:
    """Train + evaluate student, MLP, and chance floor for one seed."""
    batch = build_trials(config, seed)
    teacher_finals = np.asarray(teacher.teacher_final_phases(batch.theta0, batch.code, config))
    labels = teacher.teacher_labels(batch.theta0, batch.code, config)

    test = _test_slice(batch)

    student = models.train_student(batch, teacher_finals, config, seed, epochs=STUDENT_EPOCHS)
    student_pred = models.student_predict(student, test, config)
    student_acc = float(np.mean(student_pred == labels[batch.is_test]))

    n_sub = models.substrate_param_count()
    hidden = models.mlp_hidden_for_match(n_sub, config.n_bins)
    mlp = models.train_mlp(batch, labels, hidden, config, seed, epochs=MLP_EPOCHS)
    mlp_pred = models.mlp_predict(mlp, test)
    mlp_acc = float(np.mean(mlp_pred == labels[batch.is_test]))

    chance = models.chance_floor_accuracy(labels, batch.is_test, config.n_bins)

    return SeedResultV2(
        seed=seed,
        student_accuracy=student_acc,
        mlp_accuracy=mlp_acc,
        chance_accuracy=chance,
        student_params=n_sub,
        mlp_params=models.mlp_param_count(hidden, config.n_bins),
        mlp_hidden=hidden,
        student_j_per_task=_measure_j_per_task_student(student, batch, config),
        mlp_j_per_task=_measure_j_per_task_mlp(mlp, batch),
        steps=config.steps,
    )


def run_probe(seeds: tuple[int, ...], config: ProbeConfigV2) -> dict[str, Any]:
    """Run all seeds and evaluate the frozen pass/fail contract."""
    results = [run_seed(s, config) for s in seeds]

    def stats(values: list[float]) -> dict[str, float]:
        return {"mean": mean(values), "sd": pstdev(values)}

    student = stats([r.student_accuracy for r in results])
    mlp = stats([r.mlp_accuracy for r in results])
    chance = stats([r.chance_accuracy for r in results])
    margin = student["mean"] - mlp["mean"]

    passed = (
        student["mean"] >= PASS_ACCURACY
        and margin >= PASS_MARGIN_PP
        and student["mean"] > chance["mean"]
    )
    return {
        "config": asdict(config),
        "seeds": list(seeds),
        "student_accuracy": student,
        "mlp_accuracy": mlp,
        "chance_accuracy": chance,
        "margin_over_mlp_pp": margin,
        "thresholds": {"pass_accuracy": PASS_ACCURACY, "pass_margin_pp": PASS_MARGIN_PP},
        "verdict": "PASS" if passed else "NEGATIVE",
        "student_j_per_task": stats([r.student_j_per_task for r in results]),
        "mlp_j_per_task": stats([r.mlp_j_per_task for r in results]),
        "energy_proxy_watts": NOMINAL_POWER_W,
        "param_counts": {
            "student": results[0].student_params,
            "mlp": results[0].mlp_params,
            "mlp_hidden": results[0].mlp_hidden,
        },
        "per_seed": [asdict(r) for r in results],
    }
