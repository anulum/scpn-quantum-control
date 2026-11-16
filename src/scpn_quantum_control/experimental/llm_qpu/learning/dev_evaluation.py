# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU dev evaluation
"""Quantum-arm features per sample and a paired dev evaluation report against a classical arm."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from ..contracts.cost import QuotedCount
from ..contracts.evaluation import EvaluationReport, SampleOutcome, paired_statistics
from ..contracts.exact import ExactFeatureSet
from ..contracts.experiment import PlannedExperiment
from ..contracts.protocol import ExperimentProtocol
from ..contracts.simulated_features import SimulatedFeatureBatch
from ..contracts.task import SplitManifest
from ..contracts.wire import EVALUATION_REPORT_SCHEMA, ArtifactHeader, canonical_bytes, f64
from .classical_baselines import BaselineResult, LabelledSplit

_FloatArray = np.ndarray[Any, np.dtype[np.float64]]
_BASIS_ORDER = ("X", "Y", "Z")


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_arm_features(features: ExactFeatureSet, arm: str) -> dict[str, tuple[float, ...]]:
    """Return each sample's exact expectations of one arm, bases in X, Y, Z order."""
    rows = {}
    for row in features.rows:
        if row.arm_id != arm:
            continue
        by_basis = dict(row.expectations)
        if tuple(sorted(by_basis)) != _BASIS_ORDER:
            raise ValueError("a quantum feature row needs all three bases")
        rows[row.sample_id] = tuple(value for basis in _BASIS_ORDER for value in by_basis[basis])
    if not rows:
        raise ValueError("the feature set has no row of this arm")
    return rows


def sampled_arm_features(
    batches: tuple[SimulatedFeatureBatch, ...], experiment: PlannedExperiment, arm: str
) -> dict[str, tuple[float, ...]]:
    """Return each sample's sampled means of one arm from one batch per basis, bases in X, Y, Z order."""
    if tuple(sorted(batch.basis_id for batch in batches)) != _BASIS_ORDER:
        raise ValueError("sampled features need exactly one batch per basis")
    values: dict[str, dict[str, tuple[float, ...]]] = {}
    for batch in batches:
        for row in batch.rows:
            if experiment.measurements[row.cell_ordinal].basis_id != batch.basis_id:
                raise ValueError("a batch row is not a cell of its basis in this experiment")
            key = experiment.cells[row.cell_ordinal].key
            if experiment.cells[row.cell_ordinal].role == "data" and key.arm_id == arm:
                values.setdefault(key.source_sample_id, {})[batch.basis_id] = row.values or ()
    if not values:
        raise ValueError("the batches hold no data cell of this arm")
    return {
        sample: tuple(v for basis in _BASIS_ORDER for v in by_basis[basis])
        for sample, by_basis in values.items()
    }


def aligned_features(split: LabelledSplit, rows: dict[str, tuple[float, ...]]) -> _FloatArray:
    """Return the rows in the split's sample order, refusing any sample without features."""
    missing = [sample for sample in split.sample_ids if sample not in rows]
    if missing:
        raise ValueError(f"no quantum features for {len(missing)} split samples")
    return np.array([rows[sample] for sample in split.sample_ids], dtype=np.float64)


def _outcomes(
    results: tuple[BaselineResult, ...], names: tuple[str, ...]
) -> tuple[SampleOutcome, ...]:
    """Return every dev log-loss as an outcome of its registered arm, ordered by arm then sample."""
    outcomes = [
        SampleOutcome(outcome.sample_id, outcome.source_group, name, outcome.log_loss)
        for result, name in zip(results, names, strict=True)
        for outcome in result.outcomes
    ]
    return tuple(sorted(outcomes, key=lambda item: (item.arm, item.sample_id)))


def evaluate_on_dev(
    protocol: ExperimentProtocol,
    split: SplitManifest,
    classical: BaselineResult,
    quantum: BaselineResult,
    *,
    execution_class: str,
    interval_seed: int,
    currency: str,
    revisions: tuple[str, str],
) -> EvaluationReport:
    """Return the paired dev report of the protocol's classical and quantum arms from their dev log-losses."""
    if protocol.primary_metric != "mean_log_loss":
        raise ValueError("dev readouts report log-loss; the protocol must lock mean log-loss")
    if [o.sample_id for o in classical.outcomes] != [o.sample_id for o in quantum.outcomes]:
        raise ValueError("both arms must score the same dev samples in the same order")
    outcomes = _outcomes((classical, quantum), (protocol.classical_arm, protocol.quantum_arm))
    arm_metrics, group_deltas, mean_delta, interval = paired_statistics(
        outcomes, protocol, interval_seed
    )
    seconds = (
        classical.fit_seconds
        + classical.predict_seconds
        + quantum.fit_seconds
        + quantum.predict_seconds
    )
    costs = (
        ("provider_resource_units", QuotedCount("known", 0, None)),
        ("price_minor_units", QuotedCount("known", 0, None)),
        ("classical_compute_seconds", QuotedCount("known", math.ceil(seconds), None)),
    )
    fields: dict[str, Any] = {
        "protocol_digest": _sha(protocol.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "evaluated_split": "dev",
        "execution_class": execution_class,
        "provider": None,
        "receipt_digests": (),
        "metric": "mean_log_loss",
        "arms": protocol.arms,
        "outcomes": outcomes,
        "arm_metrics": arm_metrics,
        "group_deltas": group_deltas,
        "mean_delta": mean_delta,
        "interval_method": "paired_group_bootstrap_percentile_v1",
        "interval_seed": interval_seed,
        "interval": interval,
        "relevant": interval[0] > protocol.minimum_relevant_delta,
        "costs": costs,
        "currency": currency,
    }
    return EvaluationReport(**fields, header=_header(fields, protocol, revisions))


def _header(
    fields: dict[str, Any], protocol: ExperimentProtocol, revisions: tuple[str, str]
) -> ArtifactHeader:
    """Return the report header hashing its content, with the protocol and split as parents."""
    content = {
        "schema": EVALUATION_REPORT_SCHEMA,
        "object_kind": "evaluation_report",
        **fields,
        "receipt_digests": [],
        "arms": list(fields["arms"]),
        "outcomes": [outcome.to_wire() for outcome in fields["outcomes"]],
        "arm_metrics": [[name, f64(value)] for name, value in fields["arm_metrics"]],
        "group_deltas": [[name, f64(value)] for name, value in fields["group_deltas"]],
        "mean_delta": f64(fields["mean_delta"]),
        "interval": [f64(bound) for bound in fields["interval"]],
        "costs": [[name, count.to_wire()] for name, count in fields["costs"]],
    }
    parents = tuple(sorted((fields["protocol_digest"], fields["split_digest"])))
    return ArtifactHeader(
        "evaluation_report",
        _sha(content),
        parents,
        revisions[0],
        revisions[1],
        "offline_design",
        protocol.header.data_origin,
        "design_only",
    )
