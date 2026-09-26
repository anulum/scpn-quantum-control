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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(kind: str, content: dict[str, object], parents: tuple[str, ...]) -> ArtifactHeader:
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
    task_fields = {
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
    split_fields = {
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
    fields = {
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


def _worker(protocol: dict[str, object]) -> tuple[int, dict[str, object]]:
    result = subprocess.run(
        [sys.executable, str(_WORKER)],
        input=json.dumps({"op": "roundtrip_experiment_protocol", "protocol": protocol}).encode(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result.returncode, json.loads(result.stdout)


def test_protocol_roundtrip_and_task_binding() -> None:
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
    _, _, protocol = _fixture()
    wire = protocol.to_wire()
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
    _, _, protocol = _fixture()
    wire = protocol.to_wire()
    wire["header"]["data_origin"] = origin
    with pytest.raises(ValueError):
        decode_contract(canonical_bytes(wire))
    code, response = _worker(wire)
    assert code == 2
    assert response["status"] == "refused"


def test_protocol_refuses_impossible_classification_effect() -> None:
    _, _, protocol = _fixture()
    wire = protocol.to_wire()
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
    _, _, protocol = _fixture()
    wire = protocol.to_wire()
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
