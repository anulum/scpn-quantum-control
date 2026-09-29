# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 probe orchestration and frozen contract
"""Run the KYMA v3 probe over seeds and apply the frozen pass/fail contract.

The contract is frozen in the pre-registration
``docs/campaigns/kyma_v3_symbolic_composition_prereg_2026-09-29.md``: the
substrate PASSES iff its mean held-out accuracy is at least 10 percentage points
above the best contract baseline (parameter-matched MLP, staged GNN,
transformer) and its mean minus one standard deviation exceeds the measured
chance floor, over seeds 0–4. Diagnostic baselines are reported with a
pre-declared attribution rule and never change the verdict.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import product
from statistics import mean, pstdev
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import baselines, substrate
from .task import (
    NO_OPERATION,
    SymbolicDataset,
    all_states,
    build_dataset,
    configurations,
    design_report,
    run_configuration,
    training_marginal_accuracy,
)

SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 3000
SUBSTRATE_LEARNING_RATE = 0.05
LEARNING_RATES = {"mlp": 0.02, "gnn": 0.01, "transformer": 0.01, "sequential": 0.02}
CONTRACT_BASELINES = ("mlp", "gnn", "transformer")
LARGE_DIAGNOSTIC_WIDTHS = {"mlp_large": 64, "gnn_large": 16, "transformer_large": 16}
PASS_MARGIN = 0.10
ATTRIBUTION_MARGIN = 0.10

_COUNTS = {
    "mlp": baselines.mlp_param_count,
    "gnn": baselines.gnn_param_count,
    "transformer": baselines.transformer_param_count,
    "sequential": baselines.sequential_param_count,
}


def baseline_specs() -> tuple[baselines.BaselineSpec, ...]:
    """Contract and diagnostic baselines with their widths and parameter counts.

    Returns
    -------
    tuple of BaselineSpec
        Matched contract baselines, the matched sequential diagnostic, and the
        large-capacity diagnostics.

    """
    target = substrate.substrate_param_count()
    specs = []
    for name in (*CONTRACT_BASELINES, "sequential"):
        width = baselines.closest_width(_COUNTS[name], target)
        specs.append(
            baselines.BaselineSpec(name, width, _COUNTS[name](width), name in CONTRACT_BASELINES)
        )
    for name, large_width in LARGE_DIAGNOSTIC_WIDTHS.items():
        base = name.removesuffix("_large")
        specs.append(baselines.BaselineSpec(name, large_width, _COUNTS[base](large_width), False))
    return tuple(specs)


def realisability_accuracy() -> float:
    """Accuracy of the hand-set gates on every configuration, state and query.

    This is the teacher-free realisability check: it includes the held-out pair
    on all three queries and trains nothing.

    Returns
    -------
    float
        Fraction of the 12 × 64 × 3 items the hand-set substrate labels exactly.

    """
    rows = [
        (state, configuration) for configuration, state in product(configurations(), all_states())
    ]
    states = jnp.asarray(np.array([row[0] for row in rows]))
    first = jnp.asarray(np.array([row[1][0] for row in rows]))
    second = jnp.asarray(
        np.array([row[1][1] if len(row[1]) == 2 else NO_OPERATION for row in rows])
    )
    predicted = np.asarray(
        substrate.phase_to_label(
            substrate.final_phases(substrate.hand_gates(), states, first, second)
        )
    )
    truth = np.array([run_configuration(row[0], row[1]) for row in rows])
    return float(np.mean(predicted == truth))


def _accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predicted == labels))


def _timed_substrate_inference(gates: dict[str, jax.Array], dataset: SymbolicDataset) -> float:
    substrate.predict(gates, dataset, dataset.is_test)  # compile once
    start = time.perf_counter()
    substrate.predict(gates, dataset, dataset.is_test)
    return (time.perf_counter() - start) / int(np.sum(dataset.is_test))


@dataclass
class SeedResult:
    """Measured outcome of one seed."""

    seed: int
    substrate_train_accuracy: float
    substrate_test_accuracy: float
    substrate_seconds_per_item: float
    baseline_train_accuracy: dict[str, float]
    baseline_test_accuracy: dict[str, float]
    baseline_seconds_per_item: dict[str, float]


def run_seed(dataset: SymbolicDataset, seed: int, epochs: int = EPOCHS) -> SeedResult:
    """Train and evaluate the substrate and every baseline for one seed.

    Parameters
    ----------
    dataset
        The frozen split.
    seed
        Initialisation seed.
    epochs
        Adam steps for every model.

    Returns
    -------
    SeedResult
        Training and held-out accuracies and the substrate's inference time.

    """
    train_mask = ~dataset.is_test
    gates = substrate.train(dataset, seed, epochs, SUBSTRATE_LEARNING_RATE)
    train_accuracy: dict[str, float] = {}
    test_accuracy: dict[str, float] = {}
    seconds: dict[str, float] = {}
    for spec in baseline_specs():
        base = spec.name.removesuffix("_large")
        predicted_train, predicted_test, seconds[spec.name] = baselines.train_and_predict(
            base, spec.width, dataset, seed, epochs, LEARNING_RATES[base]
        )
        train_accuracy[spec.name] = _accuracy(predicted_train, dataset.label[train_mask])
        test_accuracy[spec.name] = _accuracy(predicted_test, dataset.label[dataset.is_test])
    return SeedResult(
        seed=seed,
        substrate_train_accuracy=_accuracy(
            substrate.predict(gates, dataset, train_mask), dataset.label[train_mask]
        ),
        substrate_test_accuracy=_accuracy(
            substrate.predict(gates, dataset, dataset.is_test), dataset.label[dataset.is_test]
        ),
        substrate_seconds_per_item=_timed_substrate_inference(gates, dataset),
        baseline_train_accuracy=train_accuracy,
        baseline_test_accuracy=test_accuracy,
        baseline_seconds_per_item=seconds,
    )


def verdict(results: list[SeedResult], chance: float) -> dict[str, Any]:
    """Apply the frozen contract and the attribution rule.

    Parameters
    ----------
    results
        One result per seed.
    chance
        Measured chance floor.

    Returns
    -------
    dict
        Means and standard deviations, the best contract baseline, the PASS flag
        and the attribution flags.

    """
    substrate_scores = [result.substrate_test_accuracy for result in results]
    names = list(results[0].baseline_test_accuracy)
    means = {name: mean(r.baseline_test_accuracy[name] for r in results) for name in names}
    sds = {name: pstdev(r.baseline_test_accuracy[name] for r in results) for name in names}
    best_contract = max(CONTRACT_BASELINES, key=lambda name: means[name])
    substrate_mean, substrate_sd = mean(substrate_scores), pstdev(substrate_scores)
    passed = (
        substrate_mean >= means[best_contract] + PASS_MARGIN
        and substrate_mean - substrate_sd > chance
    )
    diagnostics = [name for name in names if name not in CONTRACT_BASELINES]
    staged_within = {
        name: means[name] >= substrate_mean - ATTRIBUTION_MARGIN for name in ("gnn", "sequential")
    }
    return {
        "substrate_mean": substrate_mean,
        "substrate_sd": substrate_sd,
        "baseline_mean": means,
        "baseline_sd": sds,
        "best_contract_baseline": best_contract,
        "margin_over_best_contract": substrate_mean - means[best_contract],
        "chance_floor": chance,
        "pass": passed,
        "staged_model_within_attribution_margin": staged_within,
        "best_diagnostic": max(diagnostics, key=lambda name: means[name]),
    }


def run_probe(
    seeds: tuple[int, ...] = SEEDS, epochs: int = EPOCHS, nominal_power_w: float | None = None
) -> dict[str, Any]:
    """Run the design checks, every seed and the contract.

    Parameters
    ----------
    seeds
        Seeds to run (the contract uses 0–4).
    epochs
        Adam steps for every model.
    nominal_power_w
        Declared nominal package power of the host for the energy proxy; ``None``
        leaves the proxy absent.

    Returns
    -------
    dict
        The artefact: design report, realisability, parameter counts, per-seed
        results, verdict and energy proxy.

    """
    dataset = build_dataset()
    design = design_report()
    realisability = realisability_accuracy()
    results = [run_seed(dataset, seed, epochs) for seed in seeds]
    seconds = {"substrate": mean(result.substrate_seconds_per_item for result in results)}
    for name in results[0].baseline_seconds_per_item:
        seconds[name] = mean(result.baseline_seconds_per_item[name] for result in results)
    energy = (
        None
        if nominal_power_w is None
        else {
            "kind": "ENERGY PROXY (nominal package power × measured wall time), not a measurement",
            "nominal_power_w": nominal_power_w,
            "joules_per_item": {name: nominal_power_w * value for name, value in seconds.items()},
        }
    )
    return {
        "schema": "kyma_v3_probe_result_v1",
        "design": asdict(design),
        "realisability_accuracy": realisability,
        "substrate_params": substrate.substrate_param_count(),
        "baselines": [asdict(spec) for spec in baseline_specs()],
        "epochs": epochs,
        "seeds": list(seeds),
        "per_seed": [asdict(result) for result in results],
        "verdict": verdict(results, training_marginal_accuracy(dataset)),
        "seconds_per_item": seconds,
        "energy_proxy": energy,
    }
