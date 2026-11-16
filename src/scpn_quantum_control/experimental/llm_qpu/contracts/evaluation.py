# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Paired evaluation reports locked to a frozen protocol, one split and one provider."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass

from .cost import QuotedCount
from .protocol import ExperimentProtocol
from .receipt import ProviderBatchReceipt
from .task import SplitManifest
from .wire import (
    EVALUATION_REPORT_SCHEMA,
    ArtifactHeader,
    _digest,
    _strict_json,
    _text,
    canonical_bytes,
    f64,
)

_DECOMPOSABLE = ("accuracy", "mean_log_loss", "mse", "mae")
_SPLITS = ("dev", "test")
_EXECUTION = ("hardware", "sampled_simulation", "exact_simulation", "classical_only")
_COST_ITEMS = ("provider_resource_units", "price_minor_units", "classical_compute_seconds")
_INTERVAL = "paired_group_bootstrap_percentile_v1"
_RESAMPLES = 2000
_CURRENCY = re.compile(r"[A-Z]{3}\Z")


def _finite(value: object, *, name: str) -> float:
    """Return one finite float, refusing integers and booleans."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _mean(values: list[float]) -> float:
    """Return the left-to-right mean, the one order both validators use."""
    return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    """One arm's per-sample contribution to the primary metric."""

    sample_id: str
    source_group: str
    arm: str
    value: float

    def __post_init__(self) -> None:
        """Require named sample, group and arm and a finite non-negative contribution."""
        for name in ("sample_id", "source_group", "arm"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if _finite(self.value, name="sample outcome") < 0.0:
            raise ValueError("a per-sample metric contribution cannot be negative")

    def to_wire(self) -> dict[str, object]:
        """Return the exact outcome."""
        return {
            "sample_id": self.sample_id,
            "source_group": self.source_group,
            "arm": self.arm,
            "value": f64(self.value),
        }

    @classmethod
    def from_wire(cls, value: object) -> SampleOutcome:
        """Reject a record with missing or extra fields."""
        if type(value) is not dict or set(value) != {"sample_id", "source_group", "arm", "value"}:
            raise ValueError("SampleOutcome fields mismatch")
        number = value["value"]
        if type(number) is not dict or set(number) != {"$f64"} or type(number["$f64"]) is not str:
            raise ValueError("sample outcome must be an f64 tag on wire")
        return cls(
            value["sample_id"], value["source_group"], value["arm"], float.fromhex(number["$f64"])
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Design-only paired comparison of the protocol's arms on one locked split."""

    protocol_digest: str
    split_digest: str
    evaluated_split: str
    execution_class: str
    provider: str | None
    receipt_digests: tuple[str, ...]
    metric: str
    arms: tuple[str, ...]
    outcomes: tuple[SampleOutcome, ...]
    arm_metrics: tuple[tuple[str, float], ...]
    group_deltas: tuple[tuple[str, float], ...]
    mean_delta: float
    interval_method: str
    interval_seed: int
    interval: tuple[float, float]
    relevant: bool
    costs: tuple[tuple[str, QuotedCount], ...]
    currency: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse undecomposable metrics, mixed execution and an unordered inventory."""
        for name in ("protocol_digest", "split_digest"):
            _digest(getattr(self, name), name=name)
        if self.evaluated_split not in _SPLITS:
            raise ValueError("an evaluation reads the dev or the test split, never train")
        self._check_execution()
        if self.metric in ("balanced_accuracy", "f1"):
            raise ValueError("v1 reports per-sample decomposable metrics only")
        if self.metric not in _DECOMPOSABLE:
            raise ValueError("unknown evaluation metric")
        self._check_inventory()
        self._check_statistics()
        self._check_costs()
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "evaluation_report"
        ):
            raise ValueError("EvaluationReport requires its exact artifact header")
        if self.header.data_origin not in ("provider_result", "synthetic_classical"):
            raise ValueError("an evaluation derives from provider results or a fixture")
        parents = (self.protocol_digest, self.split_digest, *self.receipt_digests)
        self.header.validate_content(self._scientific_wire(), parents=parents)

    def _check_execution(self) -> None:
        """Hardware names one provider and its receipts; every other class names none."""
        if self.execution_class not in _EXECUTION:
            raise ValueError("unknown execution class")
        receipts = self.receipt_digests
        if type(receipts) is not tuple or receipts != tuple(sorted(set(receipts))):
            raise ValueError("receipt digests must be a sorted unique tuple")
        for digest in receipts:
            _digest(digest, name="receipt digest")
        if self.execution_class == "hardware":
            if self.provider not in ("ibm", "iqm") or not receipts:
                raise ValueError("a hardware evaluation names one provider and its receipts")
        elif self.provider is not None or receipts:
            raise ValueError("only a hardware evaluation names a provider or receipts")

    def _check_inventory(self) -> None:
        """Require sorted arms and outcomes ordered by arm then sample."""
        arms = self.arms
        if type(arms) is not tuple or not 2 <= len(arms) <= 32:
            raise ValueError("arms must be a sorted unique tuple of two to 32")
        if tuple(_text(arm, name="arm") for arm in arms) != tuple(sorted(set(arms))):
            raise ValueError("arms must be a sorted unique tuple of two to 32")
        outcomes = self.outcomes
        if type(outcomes) is not tuple or not outcomes:
            raise ValueError("outcomes must be a nonempty tuple of SampleOutcome records")
        if any(type(outcome) is not SampleOutcome for outcome in outcomes):
            raise ValueError("outcomes must be a nonempty tuple of SampleOutcome records")
        keys = [(outcome.arm, outcome.sample_id) for outcome in outcomes]
        if keys != sorted(set(keys)):
            raise ValueError("outcomes must be ordered by arm, then unique sample")
        if any(outcome.arm not in arms for outcome in outcomes):
            raise ValueError("an outcome names an unregistered arm")
        if self.metric == "accuracy" and any(o.value not in (0.0, 1.0) for o in outcomes):
            raise ValueError("an accuracy contribution is 0 or 1 per sample")

    def _check_statistics(self) -> None:
        """Require finite tagged statistics, a fixed interval method and ordered bounds."""
        for name in ("arm_metrics", "group_deltas"):
            pairs = getattr(self, name)
            if type(pairs) is not tuple or not pairs:
                raise ValueError(f"{name} must be a nonempty tuple of named values")
            for pair in pairs:
                if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str:
                    raise ValueError(f"{name} must be a nonempty tuple of named values")
                _finite(pair[1], name=name)
        _finite(self.mean_delta, name="mean delta")
        if self.interval_method != _INTERVAL:
            raise ValueError("unknown interval method")
        if type(self.interval_seed) is not int or not 0 <= self.interval_seed < 2**63:
            raise ValueError("interval seed must be a non-negative 63-bit integer")
        bounds = self.interval
        if type(bounds) is not tuple or len(bounds) != 2:
            raise ValueError("interval must be a lower and an upper bound")
        lower, upper = (_finite(bound, name="interval bound") for bound in bounds)
        if lower > upper:
            raise ValueError("interval lower bound exceeds the upper bound")
        if type(self.relevant) is not bool:
            raise ValueError("relevance must be a boolean")

    def _check_costs(self) -> None:
        """Require every cost item once, each known or unknown with its blocker."""
        costs = self.costs
        if type(costs) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in costs
        ):
            raise ValueError("costs must be a tuple of (name, QuotedCount) pairs")
        if tuple(name for name, _ in costs) != _COST_ITEMS:
            raise ValueError("costs must name each cost item once, in order")
        if any(type(count) is not QuotedCount for _, count in costs):
            raise ValueError("each cost must be a QuotedCount")
        if type(self.currency) is not str or _CURRENCY.fullmatch(self.currency) is None:
            raise ValueError("currency must be an ISO 4217 code")

    def _scientific_wire(self) -> dict[str, object]:
        """Return the report content hashed by its header."""
        return {
            "schema": EVALUATION_REPORT_SCHEMA,
            "object_kind": "evaluation_report",
            "protocol_digest": self.protocol_digest,
            "split_digest": self.split_digest,
            "evaluated_split": self.evaluated_split,
            "execution_class": self.execution_class,
            "provider": self.provider,
            "receipt_digests": list(self.receipt_digests),
            "metric": self.metric,
            "arms": list(self.arms),
            "outcomes": [outcome.to_wire() for outcome in self.outcomes],
            "arm_metrics": [[name, f64(value)] for name, value in self.arm_metrics],
            "group_deltas": [[name, f64(value)] for name, value in self.group_deltas],
            "mean_delta": f64(self.mean_delta),
            "interval_method": self.interval_method,
            "interval_seed": self.interval_seed,
            "interval": [f64(bound) for bound in self.interval],
            "relevant": self.relevant,
            "costs": [[name, count.to_wire()] for name, count in self.costs],
            "currency": self.currency,
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned report."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> EvaluationReport:
        """Reject unknown fields and restore the frozen report."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("EvaluationReport fields mismatch")
        if (
            value["schema"] != EVALUATION_REPORT_SCHEMA
            or value["object_kind"] != "evaluation_report"
        ):
            raise ValueError("unknown EvaluationReport schema")
        lists = (
            "receipt_digests",
            "arms",
            "outcomes",
            "arm_metrics",
            "group_deltas",
            "interval",
            "costs",
        )
        if any(type(value[name]) is not list for name in lists):
            raise ValueError("report sequences must be lists on wire")
        return cls(
            protocol_digest=value["protocol_digest"],
            split_digest=value["split_digest"],
            evaluated_split=value["evaluated_split"],
            execution_class=value["execution_class"],
            provider=value["provider"],
            receipt_digests=tuple(value["receipt_digests"]),
            metric=value["metric"],
            arms=tuple(value["arms"]),
            outcomes=tuple(SampleOutcome.from_wire(item) for item in value["outcomes"]),
            arm_metrics=_named(value["arm_metrics"]),
            group_deltas=_named(value["group_deltas"]),
            mean_delta=_tagged(value["mean_delta"]),
            interval_method=value["interval_method"],
            interval_seed=value["interval_seed"],
            interval=_bounds(value["interval"]),
            relevant=value["relevant"],
            costs=tuple(_cost(item) for item in value["costs"]),
            currency=value["currency"],
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _tagged(value: object) -> float:
    """Decode one f64 wire tag."""
    if type(value) is not dict or set(value) != {"$f64"} or type(value["$f64"]) is not str:
        raise ValueError("report numbers must be f64 tags on wire")
    return float.fromhex(value["$f64"])


def _bounds(value: list[object]) -> tuple[float, float]:
    """Decode the ``[lower, upper]`` interval."""
    if len(value) != 2:
        raise ValueError("interval must be a lower and an upper bound on wire")
    return _tagged(value[0]), _tagged(value[1])


def _named(value: list[object]) -> tuple[tuple[str, float], ...]:
    """Decode a list of ``[name, f64]`` pairs."""
    pairs: list[tuple[str, float]] = []
    for item in value:
        if type(item) is not list or len(item) != 2 or type(item[0]) is not str:
            raise ValueError("named values must be [name, f64] pairs on wire")
        pairs.append((item[0], _tagged(item[1])))
    return tuple(pairs)


def _cost(value: object) -> tuple[str, QuotedCount]:
    """Decode one ``[name, QuotedCount]`` pair."""
    if type(value) is not list or len(value) != 2 or type(value[0]) is not str:
        raise ValueError("costs must be [name, QuotedCount] pairs on wire")
    return value[0], QuotedCount.from_wire(value[1])


def paired_statistics(
    outcomes: tuple[SampleOutcome, ...], protocol: ExperimentProtocol, seed: int
) -> tuple[
    tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], float, tuple[float, float]
]:
    """Return arm metrics, group deltas, their mean and the seeded group-bootstrap interval.

    Each per-sample delta is classical minus quantum for a lower-is-better metric and
    quantum minus classical otherwise, so a positive delta always favours the quantum arm.
    """
    by_arm: dict[str, dict[str, SampleOutcome]] = {}
    for outcome in outcomes:
        by_arm.setdefault(outcome.arm, {})[outcome.sample_id] = outcome
    arm_metrics = tuple(
        (arm, _mean([by_arm[arm][sample].value for sample in sorted(by_arm[arm])]))
        for arm in sorted(by_arm)
    )
    classical = by_arm.get(protocol.classical_arm, {})
    quantum = by_arm.get(protocol.quantum_arm, {})
    if not quantum or set(classical) != set(quantum):
        raise ValueError("the primary arms must hold paired outcomes for the same samples")
    sign = 1.0 if protocol.contrast_direction == "quantum_minus_classical" else -1.0
    groups: dict[str, list[float]] = {}
    for sample in sorted(quantum):
        delta = sign * (quantum[sample].value - classical[sample].value)
        groups.setdefault(quantum[sample].source_group, []).append(delta)
    group_deltas = tuple((group, _mean(groups[group])) for group in sorted(groups))
    deltas = [delta for _, delta in group_deltas]
    rng = random.Random(seed)
    means = sorted(
        _mean([deltas[rng.randrange(len(deltas))] for _ in deltas]) for _ in range(_RESAMPLES)
    )
    return arm_metrics, group_deltas, _mean(deltas), (means[50], means[1949])


def _check_pairing(report: EvaluationReport, split: SplitManifest) -> None:
    """Require one split's groups and a paired outcome in every arm for every sample."""
    allowed = set(split.dev_groups if report.evaluated_split == "dev" else split.test_groups)
    if any(outcome.source_group not in allowed for outcome in report.outcomes):
        raise ValueError(
            "an evaluation cannot mix splits; every group belongs to the evaluated one"
        )
    samples: dict[str, set[str]] = {}
    group_of: dict[str, str] = {}
    for outcome in report.outcomes:
        samples.setdefault(outcome.arm, set()).add(outcome.sample_id)
        if group_of.setdefault(outcome.sample_id, outcome.source_group) != outcome.source_group:
            raise ValueError("one sample belongs to one source group in every arm")
    expected = set(group_of)
    if set(samples) != set(report.arms) or any(found != expected for found in samples.values()):
        raise ValueError("every evaluated sample needs a paired outcome in every arm")


def _check_receipts(report: EvaluationReport, receipts: tuple[ProviderBatchReceipt, ...]) -> None:
    """Require the named receipts, all from the report's one provider."""
    if type(receipts) is not tuple or any(type(r) is not ProviderBatchReceipt for r in receipts):
        raise ValueError("receipts must be a tuple of ProviderBatchReceipt records")
    digests = sorted(hashlib.sha256(canonical_bytes(r.to_wire())).hexdigest() for r in receipts)
    if tuple(digests) != report.receipt_digests:
        raise ValueError("report receipts differ from the receipts given")
    if any(receipt.provider != report.provider for receipt in receipts):
        raise ValueError("an evaluation cannot mix providers")
    if any(receipt.header.data_origin != "provider_metadata" for receipt in receipts):
        raise ValueError("a hardware evaluation needs provider receipts, not fixtures")


def validate_evaluation_report(
    report: EvaluationReport,
    protocol: ExperimentProtocol,
    split: SplitManifest,
    receipts: tuple[ProviderBatchReceipt, ...],
) -> None:
    """Refuse an unlocked metric, mixed splits or providers, or any altered statistic."""
    if type(report) is not EvaluationReport:
        raise ValueError("report validation needs an EvaluationReport")
    if report.protocol_digest != hashlib.sha256(canonical_bytes(protocol.to_wire())).hexdigest():
        raise ValueError("the report names a different frozen protocol")
    split_digest = hashlib.sha256(canonical_bytes(split.to_wire())).hexdigest()
    if report.split_digest != split_digest or protocol.split_digest != split_digest:
        raise ValueError("report, protocol and split name different splits")
    if report.metric != protocol.primary_metric:
        raise ValueError("the reported metric is not the protocol's locked primary metric")
    if report.arms != protocol.arms:
        raise ValueError("the report's arms differ from the protocol's")
    _check_pairing(report, split)
    _check_receipts(report, receipts)
    if (report.header.data_origin == "provider_result") != (report.execution_class == "hardware"):
        raise ValueError("only a hardware evaluation derives from provider results")
    arm_metrics, group_deltas, mean_delta, interval = paired_statistics(
        report.outcomes, protocol, report.interval_seed
    )
    if (report.arm_metrics, report.group_deltas, report.mean_delta) != (
        arm_metrics,
        group_deltas,
        mean_delta,
    ):
        raise ValueError("reported metrics or paired deltas differ from the outcomes")
    if report.interval != interval:
        raise ValueError("reported interval differs from the seeded group bootstrap")
    if report.relevant != (interval[0] > protocol.minimum_relevant_delta):
        raise ValueError("relevance must be the interval's lower bound above the protocol delta")


def decode_evaluation_report(raw: bytes) -> EvaluationReport:
    """Decode one evaluation report from strict canonical wire bytes."""
    return EvaluationReport.from_wire(_strict_json(raw))
