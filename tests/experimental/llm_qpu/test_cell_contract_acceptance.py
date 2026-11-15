# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contract acceptance
"""Exercise strict offline cell identities and the standalone wire boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArrayDescriptor,
    ArtifactHeader,
    CellKey,
    PlannedCell,
    SplitManifest,
    TaskSpec,
    attempt_id,
    canonical_bytes,
    content_id,
    decode_contract,
    f64,
    request_id,
    run_id,
    validate_completion,
    validate_task_split,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_DIGEST = "a" * 64
_KERNEL = "xy_static_digital_v1"
_BASE_COMMIT = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"


def _cell(*, shots: int = 8, metadata: dict[str, object] | None = None) -> PlannedCell:
    """Return a two-bit data cell with the given shots and metadata."""
    return PlannedCell.create(
        key=CellKey("experiment", "sample", "sequence", "arm", "basis", "r1", "e1", "b1"),
        circuit_plan_digest=_DIGEST,
        shots=shots,
        measurement_width=2,
        role="data",
        metadata={} if metadata is None else metadata,
    )


def _result(cell: PlannedCell) -> dict[str, object]:
    """Return the exact raw result a provider would report for the cell."""
    return {
        "key": cell.key.to_wire(),
        "provider_job_id": "job-1",
        "pub_index": 0,
        "binding_index": 0,
        "counts": {"00": cell.shots},
        "actual_shots": cell.shots,
        "joint_shape": [cell.shots, cell.measurement_width],
    }


@pytest.mark.parametrize("shots", [True, 1.5, "1024", 0, -1])
def test_w02_01_refuses_noninteger_shots_before_sdk(shots: Any) -> None:
    """Refuse a non-integer shot count before any SDK sees it."""
    with pytest.raises(ValueError, match="shots"):
        _cell(shots=shots)


def test_w02_01_refuses_nonfinite_or_untagged_metadata() -> None:
    """Refuse non-finite floats and floats without a float tag in cell metadata."""
    for value in (float("nan"), float("inf"), 0.5):
        with pytest.raises(ValueError, match="floats"):
            _cell(metadata={"angle": value})
    with pytest.raises(ValueError, match="finite"):
        f64(float("nan"))
    assert f64(-0.0) == {"$f64": "0x0.0p+0"}


def test_w02_02_nested_input_mutation_cannot_change_approved_bytes() -> None:
    """Check that mutating the caller's nested input cannot change the approved cell bytes."""
    source: dict[str, object] = {"config": {"values": [1, 2], "angle": f64(0.25)}}
    cell = _cell(metadata=source)
    approved = canonical_bytes(cell.to_wire())
    config = source["config"]
    assert isinstance(config, dict)
    values = config["values"]
    assert isinstance(values, list)
    values.append(99)
    config["angle"] = f64(0.5)
    source["new_field"] = True
    assert canonical_bytes(cell.to_wire()) == approved
    assert cell.to_wire()["metadata"] != source


def test_w02_03_json_and_binary_roundtrip_in_standalone_worker(tmp_path: Path) -> None:
    """Round-trip JSON and binary payloads through the standalone worker unchanged."""
    cell = _cell(metadata={"angle": f64(0.25), "labels": ["e\u0301", "x"]})
    payload = struct.pack("<dd", 0.25, -0.0)
    descriptor = ArrayDescriptor("<f8", (2,), len(payload), hashlib.sha256(payload).hexdigest())
    descriptor.validate_payload(payload)
    env = {
        "PYTHONPATH": str(tmp_path),
        "IQM_TOKEN": "poison",
        "IBM_QUANTUM_TOKEN": "poison",
    }
    request = {
        "op": "roundtrip_contract",
        "cell": cell.to_wire(),
        "array": descriptor.to_wire(),
        "array_base64": base64.b64encode(payload).decode("ascii"),
    }
    result = subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode("utf-8"),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    response = json.loads(result.stdout)
    assert response["status"] == "validated_roundtrip_no_compute"
    assert response["hardware_submission_enabled"] is False
    assert response["cell_sha256"] == hashlib.sha256(canonical_bytes(cell.to_wire())).hexdigest()
    assert (
        response["array_sha256"]
        == hashlib.sha256(canonical_bytes(descriptor.to_wire())).hexdigest()
    )
    assert decode_contract(canonical_bytes(response["cell"])) == cell
    assert decode_contract(canonical_bytes(response["array"])) == descriptor
    with pytest.raises(ValueError, match="digest"):
        descriptor.validate_payload(payload[:-1] + b"x")


def test_w02_04_content_run_and_attempt_identity_split() -> None:
    """Check that content, run and attempt identities are separate digests."""
    kwargs: dict[str, Any] = {
        "kernel_id": _KERNEL,
        "map_version": "angle_tanh_v1",
        "protocol_digest": _DIGEST,
        "scientific_fields": {"angle": f64(0.25)},
    }
    first = content_id(**kwargs, output_dir="/tmp/first")
    assert content_id(**kwargs, output_dir="/tmp/second") == first
    assert content_id(**{**kwargs, "kernel_id": "xy_sequence_digital_v1"}) != first
    request = request_id(first, _cell(), kernel_id=_KERNEL)
    assert request_id(first, _cell(shots=16), kernel_id=_KERNEL) != request
    assert request_id(first, _cell(), kernel_id="xy_sequence_digital_v1") != request
    run = run_id(request, repetition=1, calibration_epoch_id="cal-1")
    assert run_id(request, repetition=1, calibration_epoch_id="cal-2") != run
    nonce = str(uuid.uuid4())
    assert attempt_id(run, nonce=nonce) == attempt_id(run, nonce=nonce)
    assert attempt_id(run, nonce=str(uuid.uuid4())) != attempt_id(run, nonce=nonce)


def test_w02_05_refuses_unknown_schema_kernel_and_metadata_shadow() -> None:
    """Refuse an unknown schema, an unknown kernel and metadata that shadows a field."""
    cell = _cell()
    wire: dict[str, Any] = cell.to_wire()
    wire["schema"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        PlannedCell.from_wire(wire)
    wire = cell.to_wire()
    wire["unreviewed"] = True
    with pytest.raises(ValueError, match="fields"):
        PlannedCell.from_wire(wire)
    with pytest.raises(ValueError, match="unknown kernel"):
        content_id(
            kernel_id="unreviewed",
            map_version="angle_tanh_v1",
            protocol_digest=_DIGEST,
            scientific_fields={"angle": f64(0.25)},
        )
    with pytest.raises(ValueError, match="shadows"):
        _cell(metadata={"provider": "test"})
    with pytest.raises(ValueError, match="unsupported"):
        decode_contract(canonical_bytes({"schema": "scpn.experimental.llm_qpu.owner_approval.v1"}))
    with pytest.raises(ValueError, match="noncanonical"):
        decode_contract(b'{"schema":"scpn.experimental.llm_qpu.owner_approval.v1", "x":1}')


def test_w02_06_raw_completion_refuses_coercion_missing_and_shape() -> None:
    """Refuse raw completions with coerced, missing or mis-shaped results."""
    cell = _cell()
    validate_completion([cell], [_result(cell)])
    with pytest.raises(ValueError, match="incomplete"):
        validate_completion([cell], [])
    for count in (1.0, True):
        result = _result(cell)
        result["counts"] = {"00": count}
        with pytest.raises(ValueError, match="integers"):
            validate_completion([cell], [result])
    result = _result(cell)
    result["joint_shape"] = [cell.shots, 1]
    with pytest.raises(ValueError, match="shape"):
        validate_completion([cell], [result])
    result = _result(cell)
    result["counts"] = {}
    with pytest.raises(ValueError, match="nonempty"):
        validate_completion([cell], [result])
    result = _result(cell)
    result["actual_shots"] = True
    with pytest.raises(ValueError, match="actual shots"):
        validate_completion([cell], [result])
    with pytest.raises(ValueError, match="duplicate"):
        validate_completion([cell, replace(cell, role="sentinel")], [_result(cell), _result(cell)])


def _task() -> TaskSpec:
    """Return a frozen owner-dataset classification task."""
    fields: dict[str, Any] = {
        "task_id": "causal-classification-v1",
        "objective": "Predict a predeclared label from prefix tokens",
        "source_kind": "owner_dataset",
        "target_origin": "independent_ground_truth",
        "label_schema_digest": _DIGEST,
        "causal_cutoff": 128,
        "primary_metric": "balanced_accuracy",
        "group_definition": "One original document per source group",
    }
    header = _header(
        "task_spec",
        {"schema": "scpn.experimental.llm_qpu.task_spec.v2", "object_kind": "task_spec", **fields},
        (_DIGEST,),
    )
    return TaskSpec(**fields, header=header)


def _header(kind: str, content: dict[str, object], parents: tuple[str, ...]) -> ArtifactHeader:
    """Return a design-only owner-dataset header bound to the content and parents."""
    return ArtifactHeader(
        object_kind=kind,
        content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest(),
        parents=tuple(sorted(parents)),
        base_repo_commit=_BASE_COMMIT,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin="owner_dataset",
        claim_scope="design_only",
    )


def _split(task: TaskSpec) -> SplitManifest:
    """Return the frozen group split of the task."""
    fields: dict[str, Any] = {
        "task_digest": hashlib.sha256(canonical_bytes(task.to_wire())).hexdigest(),
        "dataset_digest": "b" * 64,
        "train_groups": ("source-01", "source-02"),
        "dev_groups": ("source-03",),
        "test_groups": ("source-04",),
        "seed": 17,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{
            name: list(value) if name.endswith("_groups") else value
            for name, value in fields.items()
        },
    }
    header = _header("split_manifest", content, (fields["task_digest"], fields["dataset_digest"]))
    return SplitManifest(**fields, header=header)


def test_w02_task_split_strict_worker_roundtrip(tmp_path: Path) -> None:
    """Round-trip a task and its split through the strict worker."""
    task = _task()
    split = _split(task)
    validate_task_split(task, split)
    request: dict[str, Any] = {
        "op": "roundtrip_task_split",
        "task": task.to_wire(),
        "split": split.to_wire(),
    }
    result = subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout.decode()
    response = json.loads(result.stdout)
    assert response["hardware_submission_enabled"] is False
    assert response["task_sha256"] == split.task_digest
    assert response["split_sha256"] == hashlib.sha256(canonical_bytes(split.to_wire())).hexdigest()
    assert decode_contract(canonical_bytes(response["task"])) == task
    assert decode_contract(canonical_bytes(response["split"])) == split


def test_w02_task_split_refuses_leakage_and_bad_origin(tmp_path: Path) -> None:
    """Refuse split leakage and a target origin that is not allowed."""
    task = _task()
    with pytest.raises(ValueError, match="QPU-generated"):
        replace(task, target_origin="qpu_generated")
    with pytest.raises(ValueError, match="causal cutoff"):
        replace(task, causal_cutoff=True)
    split = _split(task)
    wrong_task_fields = {**split.to_wire(), "task_digest": "c" * 64}
    wrong_task_fields.pop("header")
    wrong_task_header = _header(
        "split_manifest", wrong_task_fields, ("c" * 64, split.dataset_digest)
    )
    with pytest.raises(ValueError, match="does not bind"):
        validate_task_split(task, replace(split, task_digest="c" * 64, header=wrong_task_header))
    with pytest.raises(ValueError, match="overlap"):
        replace(split, test_groups=("source-01",))
    with pytest.raises(ValueError, match="overlap"):
        replace(split, train_groups=("e\u0301",), test_groups=("é",))
    normalized_fields = {**split.to_wire(), "train_groups": ["é"]}
    normalized_fields.pop("header")
    normalized_header = _header(
        "split_manifest", normalized_fields, (split.task_digest, split.dataset_digest)
    )
    normalized = replace(split, train_groups=("e\u0301",), header=normalized_header)
    assert normalized.train_groups == ("é",)
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(task, objective="Changed after header freeze")
    with pytest.raises(ValueError, match="confirmation"):
        replace(task.header, claim_scope="confirmation")
    with pytest.raises(ValueError, match="train groups only"):
        replace(split, transform_fit_split="all")
    with pytest.raises(ValueError, match="locked evaluator"):
        replace(split, test_target_custodian="training_worker")
    wire: dict[str, Any] = task.to_wire()
    wire["unreviewed"] = True
    with pytest.raises(ValueError, match="fields"):
        decode_contract(canonical_bytes(wire))
    old_wire: dict[str, Any] = task.to_wire()
    old_wire.pop("header")
    old_wire["schema"] = "scpn.experimental.llm_qpu.task_spec.v1"
    with pytest.raises(ValueError, match="unsupported"):
        decode_contract(canonical_bytes(old_wire))
    request: dict[str, Any] = {
        "op": "roundtrip_task_split",
        "task": task.to_wire(),
        "split": split.to_wire(),
    }
    request["split"]["task_digest"] = "c" * 64
    result = subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "refused"
    forged: dict[str, Any] = {
        "op": "roundtrip_task_split",
        "task": task.to_wire(),
        "split": split.to_wire(),
    }
    forged["task"]["header"]["content_digest"] = "f" * 64
    refused = subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(forged).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert refused.returncode == 2
    assert "content digest mismatch" in json.loads(refused.stdout)["reason"]


def _key(sample: str) -> CellKey:
    """Return a cell identity that differs from the default cell only by source sample."""
    return CellKey("experiment", sample, "sequence", "arm", "basis", "r1", "e1", "b1")


def test_cell_key_and_planned_cell_refuse_unbound_identity_and_metadata() -> None:
    """Identity fields, role and metadata bytes are refused unless exact and bounded."""
    key = _cell().key.to_wire()
    refused: tuple[object, ...] = (
        [],
        {**key, "extra": "x"},
        {name: key[name] for name in list(key)[:-1]},
    )
    for value in refused:
        with pytest.raises(ValueError, match="^CellKey fields mismatch$"):
            CellKey.from_wire(value)
    cell = _cell()
    loose_key: Any = cell.key.to_wire()
    mutable: Any = bytearray(b"{}")
    for changes, message in (
        ({"key": loose_key}, "planned cell requires CellKey"),
        ({"role": "pilot"}, "unknown cell role"),
        ({"metadata_blob": mutable}, "metadata bytes exceed limit"),
        ({"metadata_blob": b'{"n":"' + b"x" * 8192 + b'"}'}, "metadata bytes exceed limit"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            replace(cell, **changes)
    listed: Any = [("n", 1)]
    with pytest.raises(ValueError, match="^metadata must be a plain mapping$"):
        PlannedCell.create(
            key=cell.key,
            circuit_plan_digest=_DIGEST,
            shots=8,
            measurement_width=2,
            role="sentinel",
            metadata=listed,
        )


def test_identity_digests_refuse_untyped_inputs_and_noncanonical_nonces() -> None:
    """Content, request and attempt identities refuse inputs they cannot bind exactly."""
    listed: Any = [("value", 1)]
    with pytest.raises(ValueError, match="^scientific fields must be a plain mapping$"):
        content_id(
            kernel_id=_KERNEL, map_version="v1", protocol_digest=_DIGEST, scientific_fields=listed
        )
    path: Any = Path("results")
    with pytest.raises(ValueError, match="^output directory must be text$"):
        content_id(
            kernel_id=_KERNEL,
            map_version="v1",
            protocol_digest=_DIGEST,
            scientific_fields={},
            output_dir=path,
        )
    loose_cell: Any = _cell().to_wire()
    for kernel, cell in (("xy_unknown_v1", _cell()), (_KERNEL, loose_cell)):
        with pytest.raises(ValueError, match="^unknown kernel or planned cell$"):
            request_id(_DIGEST, cell, kernel_id=kernel)
    run = run_id(_DIGEST, repetition=1, calibration_epoch_id="epoch-1")
    number: Any = 4
    for nonce, message in (
        (number, "attempt nonce must be UUID text"),
        ("not-a-uuid", "attempt nonce must be UUID text"),
        (str(uuid.uuid1()), "attempt nonce must be canonical UUIDv4"),
        (str(uuid.uuid4()).upper(), "attempt nonce must be canonical UUIDv4"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            attempt_id(run, nonce=nonce)
    nonce = str(uuid.uuid4())
    assert attempt_id(run, nonce=nonce) == attempt_id(run, nonce=nonce)


def test_raw_completion_refuses_unmapped_or_inconsistent_results() -> None:
    """Results must name planned cells once, map providers one to one and sum exactly."""
    first = _cell()
    second = replace(first, key=_key("sample-2"))
    good = [_result(first), {**_result(second), "binding_index": 1}]
    validate_completion([first, second], good)
    cases: list[tuple[list[dict[str, object]], str]] = [
        ([{**good[0], "extra": 1}, good[1]], "raw result fields mismatch"),
        ([{**good[0], "key": _key("unplanned").to_wire()}, good[1]], "unknown or duplicate"),
        ([good[0], {**good[1], "key": first.key.to_wire()}], "unknown or duplicate result cell"),
        ([{**good[0], "pub_index": -1}, good[1]], "provider indices must be nonnegative"),
        ([{**good[0], "binding_index": 1_000_001}, good[1]], "provider indices must be"),
        ([{**good[0], "pub_index": True}, good[1]], "provider indices must be nonnegative"),
        ([good[0], {**good[1], "binding_index": 0}], "provider mapping is not bijective"),
        ([{**good[0], "counts": {"000": 8}}, good[1]], "raw bitstring width mismatch"),
        ([{**good[0], "counts": {"0x": 8}}, good[1]], "raw bitstring width mismatch"),
        ([{**good[0], "counts": {"00": 3, "11": 4}}, good[1]], "raw counts shot sum mismatch"),
    ]
    for received, message in cases:
        with pytest.raises(ValueError, match=f"^{message}"):
            validate_completion([first, second], received)
