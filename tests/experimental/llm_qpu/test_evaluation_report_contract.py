# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental evaluation report contract
"""Exercise paired evaluation reports against their protocol, split, receipts and worker."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    EVALUATION_REPORT_SCHEMA,
    PROVIDER_BATCH_RECEIPT_SCHEMA,
    ArtifactHeader,
    EvaluationReport,
    ExperimentProtocol,
    ProviderBatchReceipt,
    QuotedCount,
    SampleOutcome,
    SplitManifest,
    SubmittedJob,
    TaskSpec,
    canonical_bytes,
    decode_contract,
    decode_evaluation_report,
    f64,
    paired_statistics,
    validate_evaluation_report,
)

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_SEED = 20261116
_JOB_IDS = ("019f8639-96be-77d1-8f70-109a70e6fb33", "019f8639-980e-7c21-bb38-f26396a883e3")
_LOSSES = {
    "classical": (0.70, 0.66, 0.72, 0.61, 0.69, 0.64),
    "quantum": (0.62, 0.65, 0.60, 0.63, 0.58, 0.66),
}
_GROUPS = ("dev-a", "dev-a", "dev-b", "dev-b", "dev-c", "dev-c")
_DENSE_GROUPS = tuple(f"dev-{letter}" for letter in "abcdefghijkl")
_DENSE = {
    "classical": tuple(0.5 + 0.0173 * k + 0.0011 * k * k for k in range(12)),
    "quantum": tuple(0.5 + 0.0029 * k * k for k in range(12)),
}


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header bound to the content and parents."""
    return ArtifactHeader(
        kind,
        _sha(content),
        tuple(sorted(parents)),
        _BASE,
        _REVISION,
        "offline_design",
        origin,
        "design_only",
    )


def _design(metric: str = "mean_log_loss") -> tuple[TaskSpec, SplitManifest, ExperimentProtocol]:
    """Return a frozen task, its split and a two-arm protocol for the metric."""
    task_fields: dict[str, Any] = {
        "task_id": "fixture-label",
        "objective": "Predict held-out generated labels",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": metric,
        "group_definition": "One generated scene per group",
    }
    task_content = {
        "schema": "scpn.experimental.llm_qpu.task_spec.v2",
        "object_kind": "task_spec",
        **task_fields,
    }
    task = TaskSpec(
        **task_fields,
        header=_header("task_spec", task_content, ("a" * 64,), "synthetic_classical"),
    )
    split_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "dataset_digest": "b" * 64,
        "train_groups": ("train",),
        "dev_groups": _DENSE_GROUPS,
        "test_groups": ("test-a", "test-b"),
        "seed": 7,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split_content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{k: list(v) if k.endswith("_groups") else v for k, v in split_fields.items()},
    }
    split_parents = (split_fields["task_digest"], split_fields["dataset_digest"])
    split = SplitManifest(
        **split_fields,
        header=_header("split_manifest", split_content, split_parents, "synthetic_classical"),
    )
    lower_is_better = metric != "accuracy"
    fields: dict[str, Any] = {
        "experiment_id": "fixture-protocol",
        "evaluation_id": "heldout-v1",
        "task_digest": _sha(task.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "model_digest": "c" * 64,
        "compressor_digest": "d" * 64,
        "kernel_plan_digest": "e" * 64,
        "measurement_plan_digest": "f" * 64,
        "analysis_plan_digest": "0" * 64,
        "mode": "contextual_latent_transform",
        "prediction_mode": "teacher_forced_trace",
        "arms": ("classical", "quantum"),
        "classical_arm": "classical",
        "quantum_arm": "quantum",
        "primary_metric": metric,
        "contrast_direction": "classical_minus_quantum"
        if lower_is_better
        else "quantum_minus_classical",
        "minimum_relevant_delta": 0.02,
        "max_dev_fits_per_arm": 8,
        "max_qpu_evaluations": 128,
        "stopping_rule": "fixed_split_no_optional_stopping",
        "failure_policy": "retain_all_ids_report_missing_and_sensitivity",
        "inference_unit": "source_group_paired",
        "analysis_scope": "exploratory",
    }
    content = {
        "schema": "scpn.experimental.llm_qpu.experiment_protocol.v2",
        "object_kind": "experiment_protocol",
        **{
            k: list(v) if k == "arms" else f64(v) if k == "minimum_relevant_delta" else v
            for k, v in fields.items()
        },
    }
    names = (
        "task_digest",
        "split_digest",
        "model_digest",
        "compressor_digest",
        "kernel_plan_digest",
        "measurement_plan_digest",
        "analysis_plan_digest",
    )
    parents = tuple(fields[name] for name in names)
    protocol = ExperimentProtocol(
        **fields,
        header=_header("experiment_protocol", content, parents, "synthetic_classical"),
    )
    return task, split, protocol


def _receipt(
    provider: str = "iqm", origin: str = "provider_metadata", tag: str = "1"
) -> ProviderBatchReceipt:
    """Return a receipt of one provider job; reports read only its digest, provider and origin."""
    job_id = {"iqm": _JOB_IDS[int(tag) - 1], "ibm": f"d86g7h1789is738vkre{tag}"}[provider]
    job = SubmittedJob(
        ordinal=0,
        name="phase_0",
        payload_sha256=tag * 64,
        cell_ordinals=(0,),
        state="submitted",
        provider_job_id=job_id,
        id_source="submit_response",
        boundary_at_utc="2026-11-16T09:00:00Z",
        provider_created_at_utc=None,
        blocker=None,
    )
    fields: dict[str, Any] = {
        "attempt_id": "6f1c2d3e-4b5a-4c6d-8e7f-901a2b3c4d5e",
        "compiled_bundle_digest": "3" * 64,
        "bundle_payload_digest": "4" * 64,
        "owner_approval_digest": "5" * 64,
        "provider": provider,
        "account_route": "fixture-route",
        "device_name": "fixture-device",
    }
    content = {
        "schema": PROVIDER_BATCH_RECEIPT_SCHEMA,
        "object_kind": "provider_batch_receipt",
        **fields,
        "jobs": [job.to_wire()],
    }
    header = _header("provider_batch_receipt", content, ("3" * 64, "5" * 64), origin)
    return ProviderBatchReceipt(**fields, jobs=(job,), header=header)


def _outcomes(
    losses: Mapping[str, tuple[float, ...]] = _LOSSES, groups: tuple[str, ...] = _GROUPS
) -> tuple[SampleOutcome, ...]:
    """Return outcomes ordered by arm, then sample."""
    return tuple(
        SampleOutcome(f"s{index:02d}", groups[index], arm, value)
        for arm in sorted(losses)
        for index, value in enumerate(losses[arm])
    )


def _oracle(
    outcomes: tuple[SampleOutcome, ...], sign: float, seed: int
) -> tuple[
    tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], float, tuple[float, float]
]:
    """Independently compute arm means, group deltas, their mean and the bootstrap interval."""
    table: dict[tuple[str, str], SampleOutcome] = {(o.arm, o.sample_id): o for o in outcomes}
    samples = sorted({o.sample_id for o in outcomes})
    arms = sorted({o.arm for o in outcomes})
    means = tuple(
        (arm, sum(o.value for o in outcomes if o.arm == arm) / sum(o.arm == arm for o in outcomes))
        for arm in arms
    )
    per_group: dict[str, list[float]] = {}
    paired = [s for s in samples if ("quantum", s) in table and ("classical", s) in table]
    for sample in paired:
        quantum, classical = table["quantum", sample], table["classical", sample]
        per_group.setdefault(quantum.source_group, []).append(
            sign * (quantum.value - classical.value)
        )
    deltas = tuple((g, sum(per_group[g]) / len(per_group[g])) for g in sorted(per_group))
    values = [d for _, d in deltas]
    rng = random.Random(seed)
    boot = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(2000)
    )
    return means, deltas, sum(values) / len(values), (boot[50], boot[1949])


def _costs(*, known: bool = True) -> tuple[tuple[str, QuotedCount], ...]:
    """Return the cost breakdown; the classical compute time is known or blocked."""
    compute = (
        QuotedCount("known", 12, None) if known else QuotedCount("unknown", None, "not timed")
    )
    return (
        ("provider_resource_units", QuotedCount("known", 0, None)),
        ("price_minor_units", QuotedCount("known", 0, None)),
        ("classical_compute_seconds", compute),
    )


def _report(
    protocol: ExperimentProtocol,
    split: SplitManifest,
    *,
    receipts: tuple[ProviderBatchReceipt, ...] = (),
    outcomes: tuple[SampleOutcome, ...] | None = None,
    origin: str | None = None,
    **changes: Any,
) -> EvaluationReport:
    """Build a report whose statistics follow the outcomes unless overridden."""
    chosen = outcomes or _outcomes()
    sign = 1.0 if protocol.contrast_direction == "quantum_minus_classical" else -1.0
    seed = changes.get("interval_seed", _SEED)
    arm_metrics, group_deltas, mean_delta, interval = _oracle(chosen, sign, seed)
    hardware = bool(receipts)
    fields: dict[str, Any] = {
        "protocol_digest": _sha(protocol.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "evaluated_split": "dev",
        "execution_class": "hardware" if hardware else "classical_only",
        "provider": receipts[0].provider if hardware else None,
        "receipt_digests": tuple(sorted(_sha(r.to_wire()) for r in receipts)),
        "metric": protocol.primary_metric,
        "arms": protocol.arms,
        "outcomes": chosen,
        "arm_metrics": arm_metrics,
        "group_deltas": group_deltas,
        "mean_delta": mean_delta,
        "interval_method": "paired_group_bootstrap_percentile_v1",
        "interval_seed": seed,
        "interval": interval,
        "relevant": interval[0] > protocol.minimum_relevant_delta,
        "costs": _costs(),
        "currency": "EUR",
        **changes,
    }
    content = {
        "schema": EVALUATION_REPORT_SCHEMA,
        "object_kind": "evaluation_report",
        **fields,
        "receipt_digests": list(fields["receipt_digests"]),
        "arms": list(fields["arms"]),
        "outcomes": [o.to_wire() for o in fields["outcomes"]],
        "arm_metrics": [[n, f64(v)] for n, v in fields["arm_metrics"]],
        "group_deltas": [[n, f64(v)] for n, v in fields["group_deltas"]],
        "mean_delta": f64(fields["mean_delta"]),
        "interval": [f64(b) for b in fields["interval"]],
        "costs": [[n, c.to_wire()] for n, c in fields["costs"]],
    }
    data_origin = origin or ("provider_result" if hardware else "synthetic_classical")
    parents = (fields["protocol_digest"], fields["split_digest"], *fields["receipt_digests"])
    header = _header("evaluation_report", content, parents, data_origin)
    return EvaluationReport(**fields, header=header)


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=30,
        check=False,
    )


def _request(
    design: tuple[TaskSpec, SplitManifest, ExperimentProtocol],
    report: object,
    receipts: tuple[ProviderBatchReceipt, ...] = (),
) -> dict[str, object]:
    """Return one worker evaluation request."""
    task, split, protocol = design
    return {
        "op": "roundtrip_evaluation_report",
        "task": task.to_wire(),
        "split": split.to_wire(),
        "protocol": protocol.to_wire(),
        "receipts": [receipt.to_wire() for receipt in receipts],
        "report": report,
    }


def test_reports_validate_on_dev_test_hardware_and_accuracy(tmp_path: Path) -> None:
    """Classical, simulated, hardware and accuracy reports validate in both validators."""
    design = _design()
    _, split, protocol = design
    accuracy = _design("accuracy")
    test_groups = ("test-a", "test-a", "test-a", "test-b", "test-b", "test-b")
    hits = {"classical": (1.0, 0.0, 1.0, 0.0, 1.0, 0.0), "quantum": (1.0, 1.0, 1.0, 0.0, 1.0, 1.0)}
    receipts = (_receipt(tag="1"), _receipt(tag="2"))
    cases = (
        (design, _report(protocol, split), ()),
        (
            design,
            _report(
                protocol,
                split,
                outcomes=_outcomes(groups=test_groups),
                evaluated_split="test",
                execution_class="exact_simulation",
            ),
            (),
        ),
        (design, _report(protocol, split, receipts=receipts), receipts),
        (accuracy, _report(accuracy[2], accuracy[1], outcomes=_outcomes(hits)), ()),
        (design, _report(protocol, split, costs=_costs(known=False)), ()),
        (design, _report(protocol, split, outcomes=_outcomes(_DENSE, _DENSE_GROUPS)), ()),
    )
    for used, report, given in cases:
        validate_evaluation_report(report, used[2], used[1], given)
        assert decode_evaluation_report(canonical_bytes(report.to_wire())) == report
        response = _worker(_request(used, report.to_wire(), given), tmp_path)
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["report_sha256"] == _sha(report.to_wire())
    baseline = cases[0][1]
    assert baseline.arm_metrics == (
        ("classical", sum(_LOSSES["classical"]) / 6),
        ("quantum", sum(_LOSSES["quantum"]) / 6),
    )
    assert baseline.group_deltas[0] == ("dev-a", ((0.70 - 0.62) + (0.66 - 0.65)) / 2)
    assert baseline.mean_delta > 0.0
    assert baseline.relevant is (baseline.interval[0] > 0.02)
    flat = {"classical": _LOSSES["quantum"], "quantum": _LOSSES["quantum"]}
    null = _report(protocol, split, outcomes=_outcomes(flat))
    assert null.interval == (0.0, 0.0) and null.relevant is False
    validate_evaluation_report(null, protocol, split, ())
    assert cases[3][1].group_deltas == (("dev-a", 0.5), ("dev-b", 0.0), ("dev-c", 0.5))
    assert paired_statistics(baseline.outcomes, protocol, _SEED) == (
        baseline.arm_metrics,
        baseline.group_deltas,
        baseline.mean_delta,
        baseline.interval,
    )
    strong = {
        "classical": tuple(v + 0.5 for v in _LOSSES["quantum"]),
        "quantum": _LOSSES["quantum"],
    }
    assert _report(protocol, split, outcomes=_outcomes(strong)).relevant is True


def test_sample_outcome_and_report_fields_refuse_unlocked_or_malformed_records() -> None:
    """Records are exact: decomposable metrics, one execution class, ordered inventories."""
    for args, reason in (
        (("", "dev-a", "quantum", 0.5), "sample_id must be bounded nonempty text"),
        (
            ("s0", "dev-a", "quantum", -0.1),
            "^a per-sample metric contribution cannot be negative$",
        ),
        (("s00", "dev-a", "quantum", 1), "^sample outcome must be a finite float$"),
        (("s00", "dev-a", "quantum", float("inf")), "^sample outcome must be a finite float$"),
    ):
        with pytest.raises(ValueError, match=reason):
            SampleOutcome(*args)
    _, split, protocol = _design()
    valid = _report(protocol, split)
    outcomes = valid.outcomes
    stray: Any = (*outcomes[:-1], "outcome")
    for changes, reason in (
        (
            {"evaluated_split": "train"},
            "^an evaluation reads the dev or the test split, never train$",
        ),
        ({"execution_class": "emulator"}, "^unknown execution class$"),
        ({"execution_class": "hardware"}, "^a hardware evaluation names one provider and its"),
        ({"provider": "iqm"}, "^only a hardware evaluation names a provider or receipts$"),
        (
            {"receipt_digests": ("a" * 64,)},
            "^only a hardware evaluation names a provider or receipts",
        ),
        (
            {"receipt_digests": ("a" * 64, "a" * 64)},
            "^receipt digests must be a sorted unique tuple$",
        ),
        (
            {"execution_class": "hardware", "provider": "iqm"},
            "^a hardware evaluation names one provider and its receipts$",
        ),
        (
            {"receipt_digests": ("b" * 64, "a" * 64)},
            "^receipt digests must be a sorted unique tuple$",
        ),
        ({"receipt_digests": ["a" * 64]}, "^receipt digests must be a sorted unique tuple$"),
        ({"receipt_digests": ("A" * 64,)}, "^receipt digest must be a lowercase SHA-256 digest$"),
        ({"metric": "f1"}, "^v1 reports per-sample decomposable metrics only$"),
        ({"metric": "balanced_accuracy"}, "^v1 reports per-sample decomposable metrics only$"),
        ({"metric": "auc"}, "^unknown evaluation metric$"),
        ({"metric": "accuracy"}, "^an accuracy contribution is 0 or 1 per sample$"),
        ({"arms": ("quantum",)}, "^arms must be a sorted unique tuple of two to 32$"),
        ({"arms": ["classical", "quantum"]}, "^arms must be a sorted unique tuple of two to 32$"),
        ({"arms": ("quantum", "classical")}, "^arms must be a sorted unique tuple of two to 32$"),
        ({"arms": ("classical", 7)}, "arm must be text"),
        ({"outcomes": ()}, "^outcomes must be a nonempty tuple of SampleOutcome records$"),
        (
            {"outcomes": list(outcomes)},
            "^outcomes must be a nonempty tuple of SampleOutcome records$",
        ),
        ({"outcomes": stray}, "^outcomes must be a nonempty tuple of SampleOutcome records$"),
        ({"outcomes": outcomes[::-1]}, "^outcomes must be ordered by arm, then unique sample$"),
        ({"outcomes": (outcomes[0], *outcomes)}, "^outcomes must be ordered by arm, then unique"),
        ({"arms": ("baseline", "quantum")}, "^an outcome names an unregistered arm$"),
        ({"arm_metrics": ()}, "^arm_metrics must be a nonempty tuple of named values$"),
        (
            {"group_deltas": (("dev-a",),)},
            "^group_deltas must be a nonempty tuple of named values$",
        ),
        ({"group_deltas": ((1, 0.0),)}, "^group_deltas must be a nonempty tuple of named values$"),
        ({"arm_metrics": (("classical", 1),)}, "^arm_metrics must be a finite float$"),
        ({"mean_delta": float("nan")}, "^mean delta must be a finite float$"),
        ({"interval_method": "t_interval"}, "^unknown interval method$"),
        ({"interval_seed": -1}, "^interval seed must be a non-negative 63-bit integer$"),
        ({"interval_seed": True}, "^interval seed must be a non-negative 63-bit integer$"),
        ({"interval": (0.0,)}, "^interval must be a lower and an upper bound$"),
        ({"interval": (0.2, 0.1)}, "^interval lower bound exceeds the upper bound$"),
        ({"interval": (0.0, 1)}, "^interval bound must be a finite float$"),
        ({"relevant": 0}, "^relevance must be a boolean$"),
        ({"costs": _costs()[::-1]}, "^costs must name each cost item once, in order$"),
        ({"costs": _costs()[:2]}, "^costs must name each cost item once, in order$"),
        ({"costs": list(_costs())}, "^costs must be a tuple of \\(name, QuotedCount\\) pairs$"),
        ({"costs": (*_costs()[:2], ("classical_compute_seconds",))}, "^costs must be a tuple of"),
        ({"costs": (*_costs()[:2], ("classical_compute_seconds", 12))}, "^each cost must be a"),
        ({"currency": "eur"}, "^currency must be an ISO 4217 code$"),
        ({"currency": "EU1"}, "^currency must be an ISO 4217 code$"),
        ({"protocol_digest": "E" * 64}, "^protocol_digest must be a lowercase SHA-256 digest$"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    with pytest.raises(ValueError, match="^EvaluationReport requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="feature_batch"))
    with pytest.raises(ValueError, match="^an evaluation derives from provider results or a"):
        replace(valid, header=replace(valid.header, data_origin="provider_metadata"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, currency="USD")


def test_report_validation_refuses_unlocked_metrics_mixed_splits_and_providers() -> None:
    """The protocol locks metric and arms; one split, one provider, paired samples."""
    _, split, protocol = _design()
    _, other_split, other_protocol = _design("mse")
    valid = _report(protocol, split)
    leaked = ("dev-a", "dev-a", "dev-b", "test-a", "dev-c", "dev-c")
    trained = ("dev-a", "dev-a", "dev-b", "train", "dev-c", "dev-c")
    unpaired = _outcomes()[:-1]
    regrouped = (*_outcomes()[:6], replace(_outcomes()[6], source_group="dev-b"), *_outcomes()[7:])
    triple = _outcomes({**_LOSSES, "baseline": _LOSSES["classical"]})
    for used, report, reason in (
        (
            protocol,
            _report(other_protocol, split),
            "^the report names a different frozen protocol$",
        ),
        (
            protocol,
            _report(protocol, other_split),
            "^report, protocol and split name different splits$",
        ),
        (
            protocol,
            _report(protocol, split, metric="mse"),
            "^the reported metric is not the protocol's",
        ),
        (
            protocol,
            _report(protocol, split, outcomes=_outcomes(groups=leaked)),
            "^an evaluation cannot mix splits",
        ),
        (
            protocol,
            _report(protocol, split, outcomes=_outcomes(groups=trained)),
            "^an evaluation cannot mix splits",
        ),
        (
            protocol,
            _report(protocol, split, outcomes=regrouped),
            "^one sample belongs to one source group",
        ),
        (
            protocol,
            _report(protocol, split, outcomes=unpaired),
            "^every evaluated sample needs a paired",
        ),
        (
            protocol,
            _report(protocol, split, outcomes=triple, arms=("baseline", "classical", "quantum")),
            "^the report's arms differ from the protocol's$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_evaluation_report(report, used, split, ())
    with pytest.raises(ValueError, match="^report, protocol and split name different splits$"):
        validate_evaluation_report(valid, protocol, other_split, ())
    iqm, ibm = _receipt(tag="1"), _receipt("ibm", tag="2")
    hardware = _report(protocol, split, receipts=(iqm,))
    fixture = _receipt(origin="synthetic_classical", tag="2")
    stray: Any = ("receipt",)
    receipt_cases: tuple[tuple[EvaluationReport, Any, str], ...] = (
        (hardware, (), "^report receipts differ from the receipts given$"),
        (hardware, stray, "^receipts must be a tuple of ProviderBatchReceipt records$"),
        (hardware, [iqm], "^receipts must be a tuple of ProviderBatchReceipt records$"),
        (
            _report(protocol, split, receipts=(iqm, ibm)),
            (iqm, ibm),
            "^an evaluation cannot mix providers$",
        ),
        (
            _report(protocol, split, receipts=(fixture,)),
            (fixture,),
            "^a hardware evaluation needs provider",
        ),
        (
            _report(protocol, split, origin="provider_result"),
            (),
            "^only a hardware evaluation derives",
        ),
        (
            _report(protocol, split, receipts=(iqm,), origin="synthetic_classical"),
            (iqm,),
            "^only a hardware",
        ),
    )
    for report, receipts, reason in receipt_cases:
        with pytest.raises(ValueError, match=reason):
            validate_evaluation_report(report, protocol, split, receipts)
    not_a_report: Any = "report"
    with pytest.raises(ValueError, match="^report validation needs an EvaluationReport$"):
        validate_evaluation_report(not_a_report, protocol, split, ())


def test_report_validation_recomputes_every_statistic_and_the_relevance_claim() -> None:
    """Arm metrics, deltas, mean, interval and relevance must equal the recomputation."""
    _, split, protocol = _design()
    valid = _report(protocol, split)
    lower, upper = valid.interval
    for changes, reason in (
        (
            {"arm_metrics": (("classical", 0.5), valid.arm_metrics[1])},
            "^reported metrics or paired",
        ),
        (
            {"group_deltas": (("dev-a", 0.0), *valid.group_deltas[1:])},
            "^reported metrics or paired",
        ),
        ({"mean_delta": valid.mean_delta + 2**-40}, "^reported metrics or paired deltas differ"),
        (
            {"interval": (lower - 2**-40, upper)},
            "^reported interval differs from the seeded group",
        ),
        ({"relevant": not valid.relevant}, "^relevance must be the interval's lower bound above"),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_evaluation_report(_report(protocol, split, **changes), protocol, split, ())
    spread = _outcomes(groups=("dev-a", "dev-b", "dev-c", "dev-d", "dev-e", "dev-f"))
    first = _report(protocol, split, outcomes=spread)
    validate_evaluation_report(first, protocol, split, ())
    edge = {"classical": (0.02,) * 6, "quantum": (0.0,) * 6}
    boundary = _report(protocol, split, outcomes=_outcomes(edge))
    assert boundary.interval == (0.02, 0.02) and boundary.relevant is False
    validate_evaluation_report(boundary, protocol, split, ())
    assert (
        _report(protocol, split, outcomes=spread, interval_seed=_SEED + 1).interval
        != first.interval
    )
    reseeded = _report(
        protocol, split, outcomes=spread, interval_seed=_SEED + 1, interval=first.interval
    )
    with pytest.raises(ValueError, match="^reported interval differs from the seeded group"):
        validate_evaluation_report(reseeded, protocol, split, ())
    with pytest.raises(ValueError, match="^the primary arms must hold paired outcomes"):
        paired_statistics(_outcomes()[:-1], protocol, _SEED)
    with pytest.raises(ValueError, match="^the primary arms must hold paired outcomes"):
        paired_statistics(_outcomes({"classical": _LOSSES["classical"]}), protocol, _SEED)


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


_WORKER_FORGERIES: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
    (("unexpected",), True, "EvaluationReport fields mismatch"),
    (("schema",), "scpn.experimental.llm_qpu.evaluation_report.v2", "unknown EvaluationReport"),
    (("protocol_digest",), "E" * 64, "invalid protocol_digest"),
    (("evaluated_split",), "train", "never train"),
    (("execution_class",), "emulator", "unknown execution class"),
    (("execution_class",), "hardware", "names one provider and its receipts"),
    (("provider",), "iqm", "only a hardware evaluation names a provider"),
    (("receipt_digests",), ["b" * 64, "a" * 64], "receipt digests must be a sorted unique list"),
    (("receipt_digests",), ["A" * 64], "invalid receipt digest"),
    (("metric",), "f1", "per-sample decomposable metrics only"),
    (("arms",), ["quantum"], "arms must be a sorted unique list of two to 32"),
    (("outcomes",), [], "outcomes must be a nonempty list"),
    (("outcomes", 0, "unexpected"), True, "SampleOutcome fields mismatch"),
    (("outcomes", 0, "sample_id"), "", "sample_id must be bounded normalized text"),
    (("outcomes", 0, "value"), {"$f64": "-0x1.0p-1"}, "a per-sample contribution is non-negative"),
    (("outcomes", 0, "value"), 0.5, "feature numbers must be f64 tags"),
    (("outcomes", 1, "sample_id"), "s00", "outcomes must be ordered by arm and unique sample"),
    (("outcomes", 0, "arm"), "baseline", "outcomes must be ordered by arm and unique sample"),
    (("arm_metrics",), [], "arm_metrics must be a nonempty list of named values"),
    (("group_deltas", 0), ["dev-a"], "group_deltas must be a nonempty list of named values"),
    (("mean_delta",), 0.1, "feature numbers must be f64 tags"),
    (("interval_method",), "t_interval", "unknown interval method"),
    (("interval_seed",), -1, "interval seed must be a non-negative 63-bit integer"),
    (("interval",), [{"$f64": "0x0.0p+0"}], "interval must be a lower and an upper bound"),
    (("interval", 0), {"$f64": "0x1.0p+0"}, "interval lower bound exceeds the upper bound"),
    (("relevant",), 0, "relevance must be a boolean"),
    (("costs",), [], "costs must name each cost item once, in order"),
    (("costs", 2), ["classical_compute_seconds"], "costs must be [name, QuotedCount] pairs"),
    (("costs", 2, 1), {"state": "unknown", "value": 0, "blocker": "x"}, "zero is not unknown"),
    (("currency",), "eur", "currency must be an ISO 4217 code"),
    (("header", "data_origin"), "provider_metadata", "derives from provider results or a"),
    (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
)


def test_report_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    design = _design()
    _, split, protocol = design
    report = _report(protocol, split)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(report.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^EvaluationReport fields mismatch$"),
        (
            ("schema",),
            "scpn.experimental.llm_qpu.evaluation_report.v2",
            "^unknown EvaluationReport",
        ),
        (("object_kind",), "feature_batch", "^unknown EvaluationReport schema$"),
        (("outcomes",), {"first": 1}, "^report sequences must be lists on wire$"),
        (("outcomes", 0, "unexpected"), True, "^SampleOutcome fields mismatch$"),
        (("outcomes", 0, "value"), 0, "^sample outcome must be an f64 tag on wire$"),
        (("arm_metrics", 0), "classical", "^named values must be \\[name, f64\\] pairs on wire$"),
        (("mean_delta",), 0, "^report numbers must be f64 tags on wire$"),
        (("interval",), [{"$f64": "0x0.0p+0"}], "^interval must be a lower and an upper bound on"),
        (
            ("costs", 0),
            "provider_resource_units",
            "^costs must be \\[name, QuotedCount\\] pairs on",
        ),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_evaluation_report(canonical_bytes(_forge(report.to_wire(), path, value)))
    for path, forgery, refusal in _WORKER_FORGERIES:
        response = _worker(_request(design, _forge(report.to_wire(), path, forgery)), tmp_path)
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    _, other_split, other_protocol = _design("mse")
    leaked = ("dev-a", "dev-a", "dev-b", "test-a", "dev-c", "dev-c")
    regrouped = (*_outcomes()[:6], replace(_outcomes()[6], source_group="dev-b"), *_outcomes()[7:])
    iqm, ibm = _receipt(tag="1"), _receipt("ibm", tag="2")
    fixture = _receipt(origin="synthetic_classical", tag="2")
    lower, upper = report.interval
    consistent: tuple[tuple[EvaluationReport, tuple[ProviderBatchReceipt, ...], str], ...] = (
        (_report(other_protocol, split), (), "the report names a different frozen protocol"),
        (_report(protocol, other_split), (), "report, protocol and split name different splits"),
        (_report(protocol, split, metric="mse"), (), "metric or arms differ from the protocol"),
        (_report(protocol, split, outcomes=_outcomes(groups=leaked)), (), "cannot mix splits"),
        (
            _report(protocol, split, outcomes=regrouped),
            (),
            "one sample belongs to one source group",
        ),
        (_report(protocol, split, outcomes=_outcomes()[:-1]), (), "needs a paired outcome"),
        (
            _report(protocol, split, receipts=(iqm,)),
            (),
            "report receipts differ from the receipts",
        ),
        (
            _report(protocol, split, receipts=(iqm, ibm)),
            (iqm, ibm),
            "an evaluation cannot mix providers",
        ),
        (
            _report(protocol, split, receipts=(fixture,)),
            (fixture,),
            "needs provider receipts, not",
        ),
        (
            _report(protocol, split, origin="provider_result"),
            (),
            "only a hardware evaluation derives",
        ),
        (
            _report(protocol, split, arm_metrics=(("classical", 0.5), report.arm_metrics[1])),
            (),
            "reported metrics or paired deltas differ",
        ),
        (
            _report(protocol, split, mean_delta=report.mean_delta + 2**-40),
            (),
            "paired deltas differ",
        ),
        (_report(protocol, split, interval=(lower - 2**-40, upper)), (), "seeded group bootstrap"),
        (_report(protocol, split, relevant=not report.relevant), (), "lower bound above the"),
    )
    for edited, receipts, refusal in consistent:
        response = _worker(_request(design, edited.to_wire(), receipts), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    forged_receipt = _forge(iqm.to_wire(), ("provider",), "aws")
    request = _request(design, _report(protocol, split, receipts=(iqm,)).to_wire())
    request["receipts"] = [forged_receipt]
    response = _worker(request, tmp_path)
    assert response.returncode == 2
    assert "receipt provider must be ibm or iqm" in json.loads(response.stdout)["reason"]
    edge = {"classical": (0.02,) * 6, "quantum": (0.0,) * 6}
    boundary = _report(protocol, split, outcomes=_outcomes(edge))
    response = _worker(_request(design, boundary.to_wire()), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    flipped = _report(protocol, split, outcomes=_outcomes(edge), relevant=True)
    response = _worker(_request(design, flipped.to_wire()), tmp_path)
    assert response.returncode == 2
    assert "lower bound above the protocol delta" in json.loads(response.stdout)["reason"]
    accuracy = _design("accuracy")
    hits = {"classical": (1.0, 0.0, 1.0, 0.0, 1.0, 0.0), "quantum": (1.0, 1.0, 1.0, 0.0, 1.0, 1.0)}
    scored = _report(accuracy[2], accuracy[1], outcomes=_outcomes(hits)).to_wire()
    half = _forge(scored, ("outcomes", 0, "value"), {"$f64": "0x1.0000000000000p-1"})
    response = _worker(_request(accuracy, half), tmp_path)
    assert response.returncode == 2
    assert "0 or 1 for accuracy" in json.loads(response.stdout)["reason"]
    missing = _request(design, report.to_wire())
    del missing["receipts"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "evaluation request fields mismatch"
