# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2.1 supplementary-rigor orchestration
"""Orchestrate the four v2.1 supplementary analyses (ablations, stronger
baselines, MLP convergence, leave-one-out) against the frozen v2 task.

Every prediction is fixed in
``KYMA_V2_1_SUPPLEMENTARY_RIGOR_PREREGISTRATION_7f6b_2026-07-21.md``; this module
reports the ACTUAL numbers and the pre-committed pass/fail for each.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import mean, pstdev
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import ablations, baselines, design, models, teacher
from .task import ProbeConfigV2, TrialBatchV2, build_trials, disjoint_conjunctions

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
LOO_SEEDS = (0, 1, 2)  # fewer seeds per split keeps the 6-split sweep tractable
SUBSTRATE_EPOCHS = 1500
MLP_EPOCHS = 2000


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "sd": pstdev(values) if len(values) > 1 else 0.0}


def _teacher_data(
    config: ProbeConfigV2, seed: int
) -> tuple[TrialBatchV2, NDArray[np.float64], NDArray[np.int64]]:
    batch = build_trials(config, seed)
    finals = np.asarray(teacher.teacher_final_phases(batch.theta0, batch.code, config))
    labels = teacher.label_batch(batch, config)
    return batch, finals, labels


def _test_slice(batch: TrialBatchV2) -> TrialBatchV2:
    m = batch.is_test
    return TrialBatchV2(
        batch.theta0[m], batch.code[m], batch.r1_pair[m], batch.r2_pair[m], batch.is_test[m]
    )


def _acc(pred: NDArray[np.int64], labels: NDArray[np.int64], is_test: NDArray[np.bool_]) -> float:
    return float(np.mean(pred == labels[is_test]))


def _substrate_acc(
    config: ProbeConfigV2,
    seed: int,
    batch: TrialBatchV2,
    finals: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    params = models.train_student(batch, finals, config, seed, epochs=SUBSTRATE_EPOCHS)
    pred = models.student_predict(params, _test_slice(batch), config)
    return _acc(pred, labels, batch.is_test)


def _mlp_acc(
    config: ProbeConfigV2,
    seed: int,
    batch: TrialBatchV2,
    labels: NDArray[np.int64],
    hidden: int,
    epochs: int = MLP_EPOCHS,
) -> float:
    params = models.train_mlp(batch, labels, hidden, config, seed, epochs=epochs)
    pred = models.mlp_predict(params, _test_slice(batch))
    return _acc(pred, labels, batch.is_test)


def _matched_hidden(config: ProbeConfigV2) -> int:
    return models.mlp_hidden_for_match(models.substrate_param_count(), config.n_bins)


# --------------------------------------------------------------------------- #
# #1 Ablations                                                                 #
# --------------------------------------------------------------------------- #
def run_ablations(config: ProbeConfigV2, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, Any]:
    """A1 (no gating) and A2 (separable readout); both predict the advantage vanishes."""
    hidden = _matched_hidden(config)

    a1_shared, a1_mlp = [], []
    for s in seeds:
        batch, finals, labels = _teacher_data(config, s)
        shared = ablations.train_shared(batch, finals, config, s, epochs=SUBSTRATE_EPOCHS)
        a1_shared.append(
            _acc(
                ablations.shared_predict(shared, _test_slice(batch), config), labels, batch.is_test
            )
        )
        a1_mlp.append(_mlp_acc(config, s, batch, labels, hidden))

    sep = replace(config, bridge_mode="r1_only")
    a2_sub, a2_mlp = [], []
    for s in seeds:
        batch, finals, labels = _teacher_data(sep, s)
        a2_sub.append(_substrate_acc(sep, s, batch, finals, labels))
        a2_mlp.append(_mlp_acc(sep, s, batch, labels, hidden))

    a1_margin = _stats(a1_shared)["mean"] - _stats(a1_mlp)["mean"]
    a2_margin = _stats(a2_sub)["mean"] - _stats(a2_mlp)["mean"]
    return {
        "A1_no_gating": {
            "shared_substrate": _stats(a1_shared),
            "mlp": _stats(a1_mlp),
            "margin_pp": a1_margin,
            "prediction": "shared <= 0.55 AND margin < 0.10",
            "meets_prediction": _stats(a1_shared)["mean"] <= 0.55 and a1_margin < 0.10,
        },
        "A2_separable_readout": {
            "substrate": _stats(a2_sub),
            "mlp": _stats(a2_mlp),
            "margin_pp": a2_margin,
            "prediction": "margin < 0.10",
            "meets_prediction": a2_margin < 0.10,
        },
    }


# --------------------------------------------------------------------------- #
# #2 Stronger baselines                                                        #
# --------------------------------------------------------------------------- #
def run_stronger_baselines(
    config: ProbeConfigV2, seeds: tuple[int, ...] = DEFAULT_SEEDS, gnn_hidden: int = 8
) -> dict[str, Any]:
    """Over-parameterised MLP, deep MLP, and a code-conditioned GNN vs the substrate."""
    over_hidden = models.mlp_hidden_for_match(4 * models.substrate_param_count(), config.n_bins)
    deep_hidden = 24

    sub, over, deep, gnn = [], [], [], []
    for s in seeds:
        batch, finals, labels = _teacher_data(config, s)
        sub.append(_substrate_acc(config, s, batch, finals, labels))
        over.append(_mlp_acc(config, s, batch, labels, over_hidden))
        dparams = baselines.train_deep_mlp(
            batch, labels, deep_hidden, config, s, epochs=MLP_EPOCHS
        )
        deep.append(
            _acc(baselines.deep_mlp_predict(dparams, _test_slice(batch)), labels, batch.is_test)
        )
        gparams = baselines.train_gnn(batch, labels, gnn_hidden, config, s, epochs=MLP_EPOCHS)
        gnn.append(
            _acc(baselines.gnn_predict(gparams, _test_slice(batch), config), labels, batch.is_test)
        )

    sub_m = _stats(sub)["mean"]
    over_m, deep_m, gnn_m = _stats(over)["mean"], _stats(deep)["mean"], _stats(gnn)["mean"]
    return {
        "substrate": _stats(sub),
        "over_param_mlp": {
            "accuracy": _stats(over),
            "hidden": over_hidden,
            "params": models.mlp_param_count(over_hidden, config.n_bins),
            "substrate_minus": sub_m - over_m,
            "prediction": "over < 0.55 AND substrate-over >= 0.20",
            "meets_prediction": over_m < 0.55 and (sub_m - over_m) >= 0.20,
        },
        "deep_mlp": {
            "accuracy": _stats(deep),
            "hidden": deep_hidden,
            "params": baselines.deep_mlp_param_count(deep_hidden, config.n_bins),
            "substrate_minus": sub_m - deep_m,
        },
        "gnn": {
            "accuracy": _stats(gnn),
            "hidden": gnn_hidden,
            "params": baselines.gnn_param_count(gnn_hidden, config.n_bins),
            "substrate_minus": sub_m - gnn_m,
            "prediction": "substrate-gnn >= 0.15",
            "meets_prediction": (sub_m - gnn_m) >= 0.15,
        },
    }


# --------------------------------------------------------------------------- #
# #3 MLP convergence                                                           #
# --------------------------------------------------------------------------- #
def mlp_convergence(config: ProbeConfigV2, seed: int = 0, probes: int = 10) -> dict[str, Any]:
    """Record the MLP training-loss trajectory + training accuracy (rules out undertraining).

    The training loop is fully jitted, so each checkpoint is a fresh run to a
    different epoch budget; the cross-entropy on the training trials at each
    checkpoint gives the loss trajectory.
    """
    hidden = _matched_hidden(config)
    batch, _, labels = _teacher_data(config, seed)
    tr = ~batch.is_test
    feats = models._mlp_features(batch.theta0[tr], batch.code[tr])
    train_labels = jnp.asarray(labels[tr])

    step = max(1, MLP_EPOCHS // probes)
    epochs = list(range(step, MLP_EPOCHS + 1, step))
    traj: list[float] = []
    for e in epochs:
        p = models.train_mlp(batch, labels, hidden, config, seed, epochs=e)
        traj.append(
            float(
                baselines._softmax_cross_entropy(
                    models.mlp_logits(p, feats), train_labels, config.n_bins
                )
            )
        )

    final = models.train_mlp(batch, labels, hidden, config, seed, epochs=MLP_EPOCHS)
    train_pred = models.mlp_predict(
        final,
        TrialBatchV2(
            batch.theta0[tr],
            batch.code[tr],
            batch.r1_pair[tr],
            batch.r2_pair[tr],
            batch.is_test[tr],
        ),
    )
    train_acc = float(np.mean(train_pred == labels[tr]))

    total_drop = traj[0] - traj[-1] if len(traj) > 1 else 1.0
    tail_drop = traj[-2] - traj[-1] if len(traj) > 1 else 0.0
    plateaued = total_drop <= 0 or (tail_drop / total_drop) < 0.05
    return {
        "loss_trajectory": traj,
        "checkpoint_epochs": epochs,
        "final_train_loss": traj[-1],
        "train_accuracy": train_acc,
        "plateaued": plateaued,
        "prediction": "plateaued AND train_accuracy >= 0.70",
        "meets_prediction": plateaued and train_acc >= 0.70,
    }


# --------------------------------------------------------------------------- #
# #4 Leave-one-out over held-out conjunctions                                 #
# --------------------------------------------------------------------------- #
def run_leave_one_out(
    base_config: ProbeConfigV2 | None = None, seeds: tuple[int, ...] = LOO_SEEDS
) -> dict[str, Any]:
    """Hold out each disjoint conjunction in turn; design re-derived teacher-only per split."""
    base_config = base_config or ProbeConfigV2()
    per_split: list[dict[str, Any]] = []
    for r1_pair, r2_pair in disjoint_conjunctions():
        seed_cfg = replace(base_config, held_out=(r1_pair, r2_pair))
        selection = design.select_config(seed=0, base=seed_cfg)
        cfg = selection["config"]
        hidden = _matched_hidden(cfg)
        sub, mlp = [], []
        for s in seeds:
            batch, finals, labels = _teacher_data(cfg, s)
            sub.append(_substrate_acc(cfg, s, batch, finals, labels))
            mlp.append(_mlp_acc(cfg, s, batch, labels, hidden))
        per_split.append(
            {
                "held_out": [r1_pair, r2_pair],
                "k_bridge": selection["k_bridge"],
                "non_separability": selection["non_separability_rate"],
                "substrate": _stats(sub),
                "mlp": _stats(mlp),
                "margin_pp": _stats(sub)["mean"] - _stats(mlp)["mean"],
            }
        )

    sub_means = [p["substrate"]["mean"] for p in per_split]
    mlp_means = [p["mlp"]["mean"] for p in per_split]
    wins = sum(1 for p in per_split if p["substrate"]["mean"] > p["mlp"]["mean"])
    mean_margin = mean(sub_means) - mean(mlp_means)
    return {
        "per_split": per_split,
        "seeds_per_split": list(seeds),
        "substrate_mean": mean(sub_means),
        "mlp_mean": mean(mlp_means),
        "mean_margin_pp": mean_margin,
        "substrate_wins": f"{wins}/{len(per_split)}",
        "prediction": "substrate_mean >= 0.70 AND mean_margin >= 0.20 AND wins >= 5/6",
        "meets_prediction": mean(sub_means) >= 0.70 and mean_margin >= 0.20 and wins >= 5,
    }
