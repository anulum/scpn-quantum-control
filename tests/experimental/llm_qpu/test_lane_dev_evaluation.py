# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental dev evaluation of a quantum arm
"""Quantum-arm features per sample, the shared readout and a validated paired dev report."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from _llm_qpu_dev_chain import _REVISIONS, _dev_design, _planned, _spec

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    estimate_simulated_features,
    validate_evaluation_report,
)
from scpn_quantum_control.experimental.llm_qpu.data.digital_simulation import (
    simulate_exact,
    simulate_sampled,
)
from scpn_quantum_control.experimental.llm_qpu.learning.classical_baselines import (
    LabelledSplit,
    run_baseline,
    run_feature_readout,
)
from scpn_quantum_control.experimental.llm_qpu.learning.dev_evaluation import (
    aligned_features,
    evaluate_on_dev,
    exact_arm_features,
    sampled_arm_features,
)
from scpn_quantum_control.experimental.llm_qpu.learning.feature_maps import trig_product_basis

_ARM = "xy"


def _split(rows: int, tag: str, seed: int) -> LabelledSplit:
    """Return a synthetic split whose labels depend on a product of angle functions."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-math.pi, math.pi, size=(rows, 4))
    labels = (np.sin(angles[:, 0]) * np.cos(angles[:, 1]) > 0).astype(np.int64)
    return LabelledSplit(
        tuple(f"{tag}-{index:03d}" for index in range(rows)),
        tuple(f"{tag}-group-{index // 2:03d}" for index in range(rows)),
        rng.normal(size=(rows, 6)),
        angles,
        labels,
    )


_TRAIN, _DEV = _split(64, "train", 1), _split(24, "dev", 2)


def test_sampled_and_exact_arm_features_concatenate_bases_per_sample() -> None:
    """Each sample of an arm gets X, Y and Z values in order; exact and sampled agree in shape."""
    planned = _planned()
    counts = simulate_sampled(
        planned.experiment, planned.plan_set, seed_hex="5eed" * 16, revisions=_REVISIONS
    )
    batches = tuple(
        estimate_simulated_features(
            counts, planned.experiment, basis_id=basis, revisions=_REVISIONS
        )
        for basis in "XYZ"
    )
    sampled = sampled_arm_features(batches, planned.experiment, _ARM)
    exact = exact_arm_features(
        simulate_exact(planned.plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS), _ARM
    )
    assert sorted(sampled) == sorted(exact) and len(sampled) == 2
    exact_set = simulate_exact(planned.plan_set, 4, bases=("X", "Y", "Z"), revisions=_REVISIONS)
    for row in exact_set.rows:
        if row.arm_id == _ARM:
            by_basis = dict(row.expectations)
            assert exact[row.sample_id] == by_basis["X"] + by_basis["Y"] + by_basis["Z"]
    for sample, values in sampled.items():
        assert len(values) == len(exact[sample]) == 21
        assert all(-1.0 <= value <= 1.0 for value in values)
    x_row = next(
        row
        for row in batches[0].rows
        if planned.experiment.cells[row.cell_ordinal].key.arm_id == _ARM
    )
    first_sample = planned.experiment.cells[x_row.cell_ordinal].key.source_sample_id
    assert sampled[first_sample][:7] == x_row.values
    with pytest.raises(ValueError, match="^sampled features need exactly one batch per basis$"):
        sampled_arm_features(batches[:2], planned.experiment, _ARM)
    with pytest.raises(ValueError, match="^the batches hold no data cell of this arm$"):
        sampled_arm_features(batches, planned.experiment, "none")
    with pytest.raises(ValueError, match="^the feature set has no row of this arm$"):
        exact_arm_features(
            simulate_exact(planned.plan_set, 4, bases=("Z",), revisions=_REVISIONS), "none"
        )
    with pytest.raises(ValueError, match="^a quantum feature row needs all three bases$"):
        exact_arm_features(
            simulate_exact(planned.plan_set, 4, bases=("Z",), revisions=_REVISIONS), _ARM
        )
    z_only = _planned(spec=_spec(bases=("Z",)))
    z_counts = simulate_sampled(
        z_only.experiment, z_only.plan_set, seed_hex="5eed" * 16, revisions=_REVISIONS
    )
    z_batch = estimate_simulated_features(
        z_counts, z_only.experiment, basis_id="Z", revisions=_REVISIONS
    )
    with pytest.raises(ValueError, match="^sampled features need exactly one batch per basis$"):
        sampled_arm_features((z_batch,), z_only.experiment, _ARM)
    with pytest.raises(
        ValueError, match="^a batch row is not a cell of its basis in this experiment$"
    ):
        sampled_arm_features((*batches[:2], z_batch), planned.experiment, _ARM)


def test_aligned_features_follow_the_split_and_refuse_gaps() -> None:
    """Rows come back in split order; a sample without features is refused."""
    rows: dict[str, tuple[float, ...]] = {
        sample: (float(index), 1.0) for index, sample in enumerate(reversed(_DEV.sample_ids))
    }
    aligned = aligned_features(_DEV, rows)
    assert aligned.shape == (24, 2)
    assert aligned[0, 0] == float(len(_DEV.sample_ids) - 1)
    with pytest.raises(ValueError, match="^no quantum features for 1 split samples$"):
        aligned_features(_DEV, {sample: rows[sample] for sample in _DEV.sample_ids[1:]})


def test_the_feature_readout_equals_the_baseline_on_the_same_features() -> None:
    """Given the product basis as features, the readout chooses and scores exactly as that family."""
    family = run_baseline("trig_product_basis", _TRAIN, _DEV, classes=2, max_fits=80, seed=0)
    given = run_feature_readout(
        "given",
        trig_product_basis(_TRAIN.angles),
        trig_product_basis(_DEV.angles),
        _TRAIN,
        _DEV,
        classes=2,
        max_fits=80,
    )
    assert (given.l2, given.fits, given.feature_dim) == (
        family.l2,
        family.fits,
        family.feature_dim,
    )
    assert [o.log_loss for o in given.outcomes] == [o.log_loss for o in family.outcomes]
    for train_features, dev_features, reason in (
        (
            np.zeros((63, 3)),
            np.zeros((24, 3)),
            "^given features must be one finite row per sample$",
        ),
        (
            np.zeros((64, 3)),
            np.full((24, 3), np.nan),
            "^given features must be one finite row per sample$",
        ),
        (np.zeros(64), np.zeros((24, 3)), "^given features must be one finite row per sample$"),
        (np.zeros((64, 3)), np.zeros((24, 2)), "^train and dev features must have one width$"),
    ):
        with pytest.raises(ValueError, match=reason):
            run_feature_readout(
                "given", train_features, dev_features, _TRAIN, _DEV, classes=2, max_fits=80
            )
    with pytest.raises(ValueError, match="^the tuning grid exceeds the declared fit budget$"):
        run_feature_readout(
            "given", np.zeros((64, 3)), np.zeros((24, 3)), _TRAIN, _DEV, classes=2, max_fits=24
        )


def test_a_dev_report_pairs_the_arms_by_source_group_and_validates() -> None:
    """The report passes the contract's own validation; relevance and costs follow the definition."""
    split, protocol = _dev_design(_DEV.groups)
    classical = run_baseline("compressed_logistic", _TRAIN, _DEV, classes=2, max_fits=80)
    quantum = run_feature_readout(
        "quantum",
        trig_product_basis(_TRAIN.angles),
        trig_product_basis(_DEV.angles),
        _TRAIN,
        _DEV,
        classes=2,
        max_fits=80,
    )
    report = evaluate_on_dev(
        protocol,
        split,
        classical,
        quantum,
        execution_class="sampled_simulation",
        interval_seed=11,
        currency="EUR",
        revisions=_REVISIONS,
    )
    validate_evaluation_report(report, protocol, split, ())
    assert report.arms == ("classical", "quantum") and len(report.outcomes) == 48
    assert dict(report.arm_metrics) == pytest.approx(
        {"classical": classical.mean_log_loss, "quantum": quantum.mean_log_loss}
    )
    assert report.mean_delta == pytest.approx(classical.mean_log_loss - quantum.mean_log_loss)
    assert report.relevant == (report.interval[0] > 0.02)
    assert [name for name, _ in report.costs] == [
        "provider_resource_units",
        "price_minor_units",
        "classical_compute_seconds",
    ]
    assert [count.value for _, count in report.costs[:2]] == [0, 0]
    assert report.costs[2][1].value == math.ceil(
        classical.fit_seconds
        + classical.predict_seconds
        + quantum.fit_seconds
        + quantum.predict_seconds
    )
    swapped = replace(quantum, outcomes=quantum.outcomes[::-1])
    with pytest.raises(
        ValueError, match="^both arms must score the same dev samples in the same order$"
    ):
        evaluate_on_dev(
            protocol,
            split,
            classical,
            swapped,
            execution_class="sampled_simulation",
            interval_seed=11,
            currency="EUR",
            revisions=_REVISIONS,
        )
    _, accuracy = _dev_design(_DEV.groups, metric="accuracy")
    with pytest.raises(ValueError, match="^dev readouts report log-loss"):
        evaluate_on_dev(
            accuracy,
            split,
            classical,
            quantum,
            execution_class="sampled_simulation",
            interval_seed=11,
            currency="EUR",
            revisions=_REVISIONS,
        )
