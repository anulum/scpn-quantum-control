# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Exercise a frozen comparison across public and isolated worker boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArtifactHeader,
    ExperimentProtocol,
    SplitManifest,
    TaskSpec,
    canonical_bytes,
    decode_contract,
    f64,
    validate_experiment_protocol,
)

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(kind: str, content: dict[str, object], parents: tuple[str, ...]) -> ArtifactHeader:
    """Return a design-only synthetic header bound to the content and parents."""
    return ArtifactHeader(
        kind,
        _sha(content),
        tuple(sorted(parents)),
        _BASE,
        _REVISION,
        "offline_design",
        "synthetic_classical",
        "design_only",
    )


def _fixture() -> tuple[TaskSpec, SplitManifest, ExperimentProtocol]:
    """Return a frozen task, its split and a log-loss protocol bound to both."""
    task_fields: dict[str, Any] = {
        "task_id": "fixture-label",
        "objective": "Predict held-out generated labels",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": "mean_log_loss",
        "group_definition": "One generated scene per group",
    }
    task = TaskSpec(
        **task_fields,
        header=_header(
            "task_spec",
            {
                "schema": "scpn.experimental.llm_qpu.task_spec.v2",
                "object_kind": "task_spec",
                **task_fields,
            },
            ("a" * 64,),
        ),
    )
    split_fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "dataset_digest": "b" * 64,
        "train_groups": ("train",),
        "dev_groups": ("dev",),
        "test_groups": ("test",),
        "seed": 7,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split = SplitManifest(
        **split_fields,
        header=_header(
            "split_manifest",
            {
                "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
                "object_kind": "split_manifest",
                **{
                    key: list(value) if key.endswith("_groups") else value
                    for key, value in split_fields.items()
                },
            },
            (split_fields["task_digest"], split_fields["dataset_digest"]),
        ),
    )
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
        "primary_metric": "mean_log_loss",
        "contrast_direction": "classical_minus_quantum",
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
            key: list(value)
            if key == "arms"
            else f64(value)
            if key == "minimum_relevant_delta"
            else value
            for key, value in fields.items()
        },
    }
    parents = tuple(
        fields[key]
        for key in (
            "task_digest",
            "split_digest",
            "model_digest",
            "compressor_digest",
            "kernel_plan_digest",
            "measurement_plan_digest",
            "analysis_plan_digest",
        )
    )
    protocol = ExperimentProtocol(
        **fields, header=_header("experiment_protocol", content, parents)
    )
    return task, split, protocol


def _worker(protocol: dict[str, object]) -> tuple[int, dict[str, Any]]:
    """Round-trip one protocol wire through the standalone worker."""
    result = subprocess.run(
        [sys.executable, str(_WORKER)],
        input=json.dumps({"op": "roundtrip_experiment_protocol", "protocol": protocol}).encode(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result.returncode, json.loads(result.stdout)


def test_protocol_roundtrip_and_task_binding() -> None:
    """Round-trip a protocol and check its binding to the task and split."""
    task, split, protocol = _fixture()
    validate_experiment_protocol(protocol, task, split)
    raw = canonical_bytes(protocol.to_wire())
    assert decode_contract(raw) == protocol
    code, response = _worker(protocol.to_wire())
    assert code == 0
    assert response["protocol_sha256"] == hashlib.sha256(raw).hexdigest()
    assert response["protocol"] == protocol.to_wire()
    assert response["hardware_submission_enabled"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("arms", ["quantum", "classical"]),
        ("max_dev_fits_per_arm", True),
        ("stopping_rule", "stop_when_significant"),
        ("failure_policy", "drop_missing"),
        ("inference_unit", "shot"),
        ("primary_metric", "accuracy"),
        ("contrast_direction", "quantum_minus_classical"),
        ("minimum_relevant_delta", f64(0.0)),
    ],
)
def test_tampered_protocol_refused_on_both_surfaces(field: str, value: object) -> None:
    """Refuse a tampered protocol in the decoder and in the worker."""
    _, _, protocol = _fixture()
    wire: dict[str, Any] = protocol.to_wire()
    wire[field] = value
    wire["header"]["content_digest"] = _sha(
        {key: item for key, item in wire.items() if key != "header"}
    )
    with pytest.raises(ValueError):
        decode_contract(canonical_bytes(wire))
    code, response = _worker(wire)
    assert code == 2
    assert response["status"] == "refused"


def test_protocol_cannot_rebind_task_or_split() -> None:
    """Refuse a protocol rebound to a changed task or split."""
    task, split, protocol = _fixture()
    changed_task = task._scientific_wire()
    changed_task["primary_metric"] = "accuracy"
    wrong_task = replace(
        task,
        primary_metric="accuracy",
        header=_header("task_spec", changed_task, (task.label_schema_digest,)),
    )
    with pytest.raises(ValueError):
        validate_experiment_protocol(protocol, wrong_task, split)
    changed_split = split._scientific_wire()
    changed_split["dataset_digest"] = "1" * 64
    wrong_split = replace(
        split,
        dataset_digest="1" * 64,
        header=_header("split_manifest", changed_split, (split.task_digest, "1" * 64)),
    )
    with pytest.raises(ValueError):
        validate_experiment_protocol(protocol, task, wrong_split)


@pytest.mark.parametrize("origin", ["owner_checkpoint", "hardware"])
def test_protocol_refuses_non_task_data_origin(origin: str) -> None:
    """Refuse a protocol whose data origin is not a task source."""
    _, _, protocol = _fixture()
    wire: dict[str, Any] = protocol.to_wire()
    wire["header"]["data_origin"] = origin
    with pytest.raises(ValueError):
        decode_contract(canonical_bytes(wire))
    code, response = _worker(wire)
    assert code == 2
    assert response["status"] == "refused"


def test_protocol_refuses_impossible_classification_effect() -> None:
    """Refuse a classification effect larger than one."""
    _, _, protocol = _fixture()
    wire: dict[str, Any] = protocol.to_wire()
    wire["primary_metric"] = "accuracy"
    wire["contrast_direction"] = "quantum_minus_classical"
    wire["minimum_relevant_delta"] = f64(1.1)
    wire["header"]["content_digest"] = _sha(
        {key: item for key, item in wire.items() if key != "header"}
    )
    with pytest.raises(ValueError, match="cannot exceed one"):
        decode_contract(canonical_bytes(wire))
    code, response = _worker(wire)
    assert code == 2
    assert "cannot exceed one" in response["reason"]


def test_v1_wire_does_not_silently_upgrade() -> None:
    """Refuse a v1 wire instead of upgrading it silently."""
    _, _, protocol = _fixture()
    wire: dict[str, Any] = protocol.to_wire()
    wire["schema"] = "scpn.experimental.llm_qpu.experiment_protocol.v1"
    wire.pop("contrast_direction")
    wire.pop("minimum_relevant_delta")
    wire["header"]["content_digest"] = _sha(
        {key: item for key, item in wire.items() if key != "header"}
    )
    with pytest.raises(ValueError):
        decode_contract(canonical_bytes(wire))
    code, response = _worker(wire)
    assert code == 2
    assert response["status"] == "refused"


_LINEAGE = (
    "task_digest",
    "split_digest",
    "model_digest",
    "compressor_digest",
    "kernel_plan_digest",
    "measurement_plan_digest",
    "analysis_plan_digest",
)


def _rewire(
    protocol: ExperimentProtocol, origin: str | None = None, **changes: Any
) -> dict[str, Any]:
    """Return the protocol wire with changed fields and a header bound to them."""
    wire: dict[str, Any] = protocol.to_wire()
    wire.update(changes)
    header = wire["header"]
    header["content_digest"] = _sha({key: item for key, item in wire.items() if key != "header"})
    header["parents"] = sorted(wire[name] for name in _LINEAGE)
    if origin is not None:
        header["data_origin"] = origin
    return wire


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"compressor_digest": "c" * 64}, "protocol parent digests must identify distinct"),
        ({"mode": "streaming"}, "unknown memory experiment mode"),
        ({"prediction_mode": "beam_search"}, "unknown prediction mode"),
        ({"arms": ("classical",)}, "protocol needs bounded comparison arms"),
        ({"arms": ["classical", "quantum"]}, "protocol needs bounded comparison arms"),
        ({"arms": tuple(f"arm-{index:02d}" for index in range(33))}, "protocol needs bounded"),
        ({"classical_arm": "unregistered"}, "primary contrast must name distinct registered arms"),
        ({"quantum_arm": "classical"}, "primary contrast must name distinct registered arms"),
        ({"primary_metric": "auroc"}, "unknown primary metric"),
        ({"analysis_scope": "post_hoc"}, "analysis scope must be declared"),
    ],
)
def test_protocol_refuses_unregistered_designs(changes: dict[str, Any], message: str) -> None:
    """A protocol refuses shared lineage, unknown modes, unbounded arms, contrasts and scopes."""
    _, _, protocol = _fixture()
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(protocol, **changes)


def test_protocol_refuses_a_header_of_another_kind() -> None:
    """A protocol accepts only its own experiment-protocol header."""
    task, _, protocol = _fixture()
    header_wire: Any = protocol.header.to_wire()
    for header in (task.header, header_wire):
        with pytest.raises(
            ValueError, match="^ExperimentProtocol requires its exact artifact header$"
        ):
            replace(protocol, header=header)


def test_protocol_wire_refuses_other_fields_schemas_arms_and_thresholds() -> None:
    """The wire decoder refuses an edited field set, schema, arm list or delta tag."""
    _, _, protocol = _fixture()
    encoded = protocol.to_wire()
    cases: list[tuple[object, str]] = [
        ("protocol", "ExperimentProtocol fields mismatch"),
        ({**encoded, "extra": 1}, "ExperimentProtocol fields mismatch"),
        ({**encoded, "schema": "v0"}, "unknown ExperimentProtocol schema"),
        ({**encoded, "object_kind": "task_spec"}, "unknown ExperimentProtocol schema"),
        ({**encoded, "arms": "classical"}, "protocol arms must be a list on wire"),
        ({**encoded, "minimum_relevant_delta": 0.02}, "minimum relevant delta must be tagged f64"),
        (
            {**encoded, "minimum_relevant_delta": {"$f64": 5}},
            "minimum relevant delta must be tagged",
        ),
        (
            {**encoded, "minimum_relevant_delta": {"$f64": "zz"}},
            "minimum relevant delta must be tagged",
        ),
        (
            {**encoded, "minimum_relevant_delta": {"$f64": "0x1p-1"}},
            "minimum relevant delta must be canonical",
        ),
        (
            {**encoded, "minimum_relevant_delta": {"$f64": "inf"}},
            "minimum relevant delta must be canonical",
        ),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}"):
            ExperimentProtocol.from_wire(value)


def test_validation_refuses_a_protocol_bound_to_another_design() -> None:
    """Validation needs the frozen protocol bound to the task, its metric and its origin."""
    task, split, protocol = _fixture()
    loose: Any = protocol.to_wire()
    with pytest.raises(ValueError, match="^protocol must be a validated contract$"):
        validate_experiment_protocol(loose, task, split)
    cases: list[tuple[dict[str, Any], str]] = [
        (_rewire(protocol, task_digest="1" * 64), "protocol task digest mismatch"),
        (
            _rewire(
                protocol,
                primary_metric="accuracy",
                contrast_direction="quantum_minus_classical",
            ),
            "protocol primary metric differs from frozen task",
        ),
        (
            _rewire(protocol, origin="owner_dataset"),
            "protocol data origin differs from frozen task",
        ),
    ]
    for wire, message in cases:
        rebound = decode_contract(canonical_bytes(wire))
        assert isinstance(rebound, ExperimentProtocol)
        with pytest.raises(ValueError, match=f"^{message}$"):
            validate_experiment_protocol(rebound, task, split)
