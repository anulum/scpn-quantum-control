# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental latent batch contract
"""Check causal, finite hidden-state wire in the standalone worker."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArrayDescriptor,
    ArtifactHeader,
    LatentBatch,
    ModelDescriptor,
    SplitManifest,
    TaskSpec,
    canonical_bytes,
    decode_contract,
    validate_latent_batch,
)

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"


def _digest(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header bound to the given content and parents."""
    return ArtifactHeader(
        object_kind=kind,
        content_digest=_digest(content),
        parents=tuple(sorted(parents)),
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin=origin,
        claim_scope="design_only",
    )


def _records() -> tuple[TaskSpec, SplitManifest, ModelDescriptor]:
    """Return a frozen synthetic task, its split and a four-wide model descriptor."""
    task_fields: dict[str, Any] = {
        "task_id": "synthetic-prefix-task",
        "objective": "Predict a predeclared label from prefix tokens",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 2,
        "primary_metric": "balanced_accuracy",
        "group_definition": "One original source per group",
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
            (task_fields["label_schema_digest"],),
            "synthetic_classical",
        ),
    )
    split_fields: dict[str, Any] = {
        "task_digest": _digest(task.to_wire()),
        "dataset_digest": "b" * 64,
        "train_groups": ("group-a", "group-b"),
        "dev_groups": ("group-c",),
        "test_groups": ("group-d",),
        "seed": 19,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split_content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{
            name: list(value) if name.endswith("_groups") else value
            for name, value in split_fields.items()
        },
    }
    split = SplitManifest(
        **split_fields,
        header=_header(
            "split_manifest",
            split_content,
            (split_fields["task_digest"], split_fields["dataset_digest"]),
            "synthetic_classical",
        ),
    )
    model_fields: dict[str, Any] = {
        "model_id": "synthetic-contract-source",
        "checkpoint_digest": "c" * 64,
        "tokenizer_digest": "d" * 64,
        "chat_template_digest": "e" * 64,
        "runtime_build_digest": "f" * 64,
        "loader_id": "fixture-loader",
        "quantization": "none",
        "tensor_dtype": "float32",
        "block_count": 4,
        "hidden_width": 4,
        "tap_block_index": 2,
        "tap_stream": "residual_hidden_state",
        "tap_boundary": "before_norm",
        "probe_evidence_digest": "1" * 64,
    }
    model = ModelDescriptor(
        **model_fields,
        header=_header(
            "model_descriptor",
            {
                "schema": "scpn.experimental.llm_qpu.model_descriptor.v1",
                "object_kind": "model_descriptor",
                **model_fields,
            },
            tuple(
                model_fields[name]
                for name in (
                    "checkpoint_digest",
                    "tokenizer_digest",
                    "chat_template_digest",
                    "runtime_build_digest",
                    "probe_evidence_digest",
                )
            ),
            "owner_checkpoint",
        ),
    )
    return task, split, model


def _batch(
    task: TaskSpec, split: SplitManifest, model: ModelDescriptor, *, payload: bytes | None = None
) -> tuple[LatentBatch, bytes]:
    """Return a two-row, three-step sequence batch and its zero-padded payload."""
    if payload is None:
        payload = struct.pack(
            "<24f",
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            9.0,
            10.0,
            11.0,
            12.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    tensor = ArrayDescriptor("<f4", (2, 3, 4), len(payload), hashlib.sha256(payload).hexdigest())
    fields: dict[str, Any] = {
        "task_digest": _digest(task.to_wire()),
        "split_digest": _digest(split.to_wire()),
        "model_digest": _digest(model.to_wire()),
        "split_name": "train",
        "layout": "chunk_isolated",
        "sample_ids": ("sample-1", "sample-2"),
        "source_ids": ("source-a", "source-b"),
        "group_ids": ("group-a", "group-b"),
        "lengths": (2, 1),
        "mask": ((True, True, False), (True, False, False)),
        "token_positions": ((0, 1, None), (0, None, None)),
        "answer_start_positions": ((2, 2, None), (1, None, None)),
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    content: dict[str, Any] = {
        "schema": "scpn.experimental.llm_qpu.latent_batch.v1",
        "object_kind": "latent_batch",
        **{
            name: value.to_wire()
            if name == "tensor"
            else [list(row) for row in value]
            if name in ("mask", "token_positions", "answer_start_positions")
            else list(value)
            if name in ("sample_ids", "source_ids", "group_ids", "lengths")
            else value
            for name, value in fields.items()
        },
    }
    header = _header(
        "latent_batch",
        content,
        (fields["task_digest"], fields["split_digest"], fields["model_digest"], tensor.sha256),
        "synthetic_classical",
    )
    return LatentBatch(**fields, header=header), payload


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )


def _request(
    task: TaskSpec,
    split: SplitManifest,
    model: ModelDescriptor,
    batch: LatentBatch,
    payload: bytes,
) -> dict[str, object]:
    """Return the worker request that round-trips one latent batch."""
    return {
        "op": "roundtrip_latent_batch",
        "task": task.to_wire(),
        "split": split.to_wire(),
        "model": model.to_wire(),
        "latent": batch.to_wire(),
        "tensor_base64": base64.b64encode(payload).decode("ascii"),
    }


def test_latent_batch_standalone_roundtrip(tmp_path: Path) -> None:
    """Round-trip a latent batch through the standalone worker."""
    task, split, model = _records()
    batch, payload = _batch(task, split, model)
    validate_latent_batch(task, split, model, batch, payload)
    response = _worker(_request(task, split, model, batch, payload), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    result = json.loads(response.stdout)
    assert result["status"] == "validated_roundtrip_no_compute"
    assert result["hardware_submission_enabled"] is False
    assert result["latent_sha256"] == _digest(batch.to_wire())
    assert decode_contract(canonical_bytes(result["latent"])) == batch


def test_contextual_latent_rows_do_not_flatten_sequences(tmp_path: Path) -> None:
    """Check that contextual latent rows keep their sequences instead of flattening them."""
    task, split, model = _records()
    sequence, _ = _batch(task, split, model)
    payload = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0)
    tensor = ArrayDescriptor("<f4", (2, 4), len(payload), hashlib.sha256(payload).hexdigest())
    content: dict[str, Any] = {
        **sequence._scientific_wire(),
        "layout": "contextual",
        "lengths": [1, 1],
        "mask": [[True], [True]],
        "token_positions": [[0], [0]],
        "answer_start_positions": [[2], [1]],
        "tensor": tensor.to_wire(),
    }
    header = _header(
        "latent_batch",
        content,
        (sequence.task_digest, sequence.split_digest, sequence.model_digest, tensor.sha256),
        "synthetic_classical",
    )
    batch = LatentBatch.from_wire({**content, "header": header.to_wire()})
    validate_latent_batch(task, split, model, batch, payload)
    response = _worker(_request(task, split, model, batch, payload), tmp_path)
    assert response.returncode == 0, response.stdout.decode()
    assert json.loads(response.stdout)["latent_sha256"] == _digest(batch.to_wire())


def test_latent_batch_refuses_leakage_and_bad_payload(tmp_path: Path) -> None:
    """Refuse latent batches with split leakage or a malformed payload."""
    task, split, model = _records()
    batch, payload = _batch(task, split, model)
    with pytest.raises(ValueError, match="mask and length"):
        replace(batch, mask=((True, False, False), (True, False, False)))
    with pytest.raises(ValueError, match="reaches answer"):
        replace(batch, token_positions=((2, 1, None), (0, None, None)))
    changed_model_digest = "9" * 64
    with pytest.raises(ValueError, match="latent model digest mismatch"):
        validate_latent_batch(
            task,
            split,
            model,
            replace(
                batch,
                model_digest=changed_model_digest,
                header=_header(
                    "latent_batch",
                    {**batch._scientific_wire(), "model_digest": changed_model_digest},
                    (
                        batch.task_digest,
                        batch.split_digest,
                        changed_model_digest,
                        batch.tensor.sha256,
                    ),
                    "synthetic_classical",
                ),
            ),
            payload,
        )
    with pytest.raises(ValueError, match="group outside"):
        validate_latent_batch(
            task,
            split,
            model,
            replace(
                batch,
                group_ids=("group-a", "group-c"),
                header=_header(
                    "latent_batch",
                    {**batch._scientific_wire(), "group_ids": ["group-a", "group-c"]},
                    (
                        batch.task_digest,
                        batch.split_digest,
                        batch.model_digest,
                        batch.tensor.sha256,
                    ),
                    "synthetic_classical",
                ),
            ),
            payload,
        )
    bad = bytearray(payload)
    bad[8 * 4 : 9 * 4] = struct.pack("<f", 3.0)
    changed, changed_payload = _batch(task, split, model, payload=bytes(bad))
    with pytest.raises(ValueError, match="padding"):
        validate_latent_batch(task, split, model, changed, changed_payload)
    refused = _worker(_request(task, split, model, changed, changed_payload), tmp_path)
    assert refused.returncode == 2
    assert "padding" in json.loads(refused.stdout)["reason"]
    nonfinite = bytearray(payload)
    nonfinite[0:4] = struct.pack("<f", float("nan"))
    invalid, invalid_payload = _batch(task, split, model, payload=bytes(nonfinite))
    refused = _worker(_request(task, split, model, invalid, invalid_payload), tmp_path)
    assert refused.returncode == 2
    assert "nonfinite" in json.loads(refused.stdout)["reason"]


def _wire_value(value: object) -> object:
    """Return the wire form of one batch field value."""
    if isinstance(value, ArrayDescriptor):
        return value.to_wire()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    return value


def _rebind(
    batch: LatentBatch, *, origin: str = "synthetic_classical", **changes: Any
) -> LatentBatch:
    """Return the batch with changed fields and a header bound to the changed content."""
    fields = {item.name: getattr(batch, item.name) for item in dataclass_fields(batch)}
    fields.update(changes)
    content = {name: value for name, value in batch.to_wire().items() if name != "header"} | {
        name: _wire_value(value) for name, value in changes.items()
    }
    tensor: ArrayDescriptor = fields["tensor"]
    parents = (
        fields["task_digest"],
        fields["split_digest"],
        fields["model_digest"],
        tensor.sha256,
    )
    fields["header"] = _header("latent_batch", content, parents, origin)
    return LatentBatch(**fields)


def _model_variant(model: ModelDescriptor, **changes: Any) -> ModelDescriptor:
    """Return the model descriptor with changed fields and a header bound to them."""
    content = {name: value for name, value in model.to_wire().items() if name != "header"}
    content.update(changes)
    fields = {item.name: getattr(model, item.name) for item in dataclass_fields(model)}
    fields.update(changes)
    fields["header"] = _header(
        "model_descriptor", content, model.header.parents, "owner_checkpoint"
    )
    return ModelDescriptor(**fields)


def _wide_tensor() -> ArrayDescriptor:
    """Return an [N,d] float32 descriptor with the batch's row count."""
    return ArrayDescriptor("<f4", (2, 4), 32, "0" * 64)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split_name": "validation"}, "unknown latent split"),
        ({"layout": "flat"}, "unknown latent layout"),
        ({"tensor": ArrayDescriptor("<f8", (2, 3, 4), 192, "0" * 64)}, "latent tensor must be"),
        ({"layout": "contextual"}, "contextual latent tensor must have"),
        ({"tensor": "descriptor"}, "latent tensor must be little-endian float32"),
        ({"tensor": _wide_tensor()}, "sequence latent tensor must have"),
        ({"sample_ids": ("sample-1",)}, "sample_ids must match latent row count"),
        ({"group_ids": ["group-a", "group-b"]}, "group_ids must match latent row count"),
        ({"sample_ids": ("same", "same")}, "duplicate latent sample ID"),
        ({"lengths": (2,)}, "latent lengths must match row count"),
        ({"mask": ((True, True), (True, False))}, "latent mask must match tensor steps"),
        ({"token_positions": [(0, 1, None), (0, None, None)]}, "latent token_positions must"),
        ({"lengths": (4, 1)}, "latent length exceeds tensor steps"),
        ({"lengths": (0, 1)}, "latent length exceeds tensor steps"),
        ({"lengths": (True, 1)}, "latent length exceeds tensor steps"),
        ({"mask": ((True, True, False), (True, False, 0))}, "latent mask and length disagree"),
        ({"token_positions": ((0, 1, None), (0, 1, None))}, "latent padding carries token"),
        ({"answer_start_positions": ((2, 2, None), (1, 2, None))}, "latent padding carries"),
        ({"tap_block_index": 1024}, "invalid latent tap block index"),
        ({"tap_block_index": -1}, "invalid latent tap block index"),
        ({"tap_boundary": "inside_norm"}, "latent tap boundary missing"),
    ],
)
def test_latent_batch_refuses_malformed_rows_and_taps(
    changes: dict[str, Any], message: str
) -> None:
    """A batch refuses bad splits, layouts, tensors, row identities, masks, positions and taps."""
    task, split, model = _records()
    batch, _ = _batch(task, split, model)
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(batch, **changes)


def test_contextual_latent_lengths_are_bounded_by_one_step_in_both_validators(
    tmp_path: Path,
) -> None:
    """A contextual row of length two is refused by the step bound in Python and in the worker."""
    task, split, model = _records()
    batch, _ = _batch(task, split, model)
    payload = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0)
    tensor = ArrayDescriptor("<f4", (2, 4), len(payload), hashlib.sha256(payload).hexdigest())
    contextual = _rebind(
        batch,
        layout="contextual",
        lengths=(1, 1),
        mask=((True,), (True,)),
        token_positions=((0,), (0,)),
        answer_start_positions=((2,), (1,)),
        tensor=tensor,
    )
    with pytest.raises(ValueError, match="^latent length exceeds tensor steps$"):
        replace(contextual, lengths=(2, 1))
    content = {name: value for name, value in contextual.to_wire().items() if name != "header"}
    content["lengths"] = [2, 1]
    parents = (
        contextual.task_digest,
        contextual.split_digest,
        contextual.model_digest,
        tensor.sha256,
    )
    header = _header("latent_batch", content, parents, "synthetic_classical")
    request = _request(task, split, model, contextual, payload)
    refused = _worker({**request, "latent": {**content, "header": header.to_wire()}}, tmp_path)
    assert refused.returncode == 2
    assert json.loads(refused.stdout)["reason"] == "latent length exceeds tensor steps"


def test_latent_batch_wire_refusals_through_the_decoder() -> None:
    """The decoder refuses an edited latent wire field by field."""
    task, split, model = _records()
    batch, _ = _batch(task, split, model)
    encoded = batch.to_wire()
    cases: list[tuple[dict[str, object], str]] = [
        ({**encoded, "extra": 1}, "LatentBatch fields mismatch"),
        ({**encoded, "object_kind": "compressed_latent_batch"}, "unknown LatentBatch schema"),
        ({**encoded, "lengths": 2}, "latent rows must be lists on wire"),
        ({**encoded, "sample_ids": "sample-1"}, "latent rows must be lists on wire"),
        (
            {**encoded, "mask": [{"step": True}, [True, False, False]]},
            "latent steps must be lists",
        ),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}"):
            decode_contract(canonical_bytes(value))


def test_latent_validation_refuses_records_that_do_not_bind() -> None:
    """Validation needs frozen records, one data origin, the model's width and tap, and the cutoff."""
    task, split, model = _records()
    batch, payload = _batch(task, split, model)
    loose_task: Any = task.to_wire()
    with pytest.raises(ValueError, match="^latent validation requires frozen contracts$"):
        validate_latent_batch(loose_task, split, model, batch, payload)
    owner = _rebind(batch, origin="owner_dataset")
    with pytest.raises(ValueError, match="^latent data origin mismatch$"):
        validate_latent_batch(task, split, model, owner, payload)
    for changes, message in (
        ({"hidden_width": 8}, "^latent dtype or hidden width mismatch$"),
        ({"tap_block_index": 1}, "^latent tap differs from model descriptor$"),
        ({"tap_boundary": "after_norm"}, "^latent tap differs from model descriptor$"),
    ):
        variant = _model_variant(model, **changes)
        bound = _rebind(batch, model_digest=_digest(variant.to_wire()))
        with pytest.raises(ValueError, match=message):
            validate_latent_batch(task, split, variant, bound, payload)
    beyond = _rebind(
        batch,
        lengths=(3, 1),
        mask=((True, True, True), (True, False, False)),
        token_positions=((0, 1, 3), (0, None, None)),
        answer_start_positions=((2, 2, 4), (1, None, None)),
    )
    with pytest.raises(ValueError, match="^latent token position exceeds causal cutoff$"):
        validate_latent_batch(task, split, model, beyond, payload)


def test_latent_batch_refuses_a_header_of_another_kind() -> None:
    """A batch accepts only its own latent-batch header, not another record's."""
    task, split, model = _records()
    batch, _ = _batch(task, split, model)
    header_wire: Any = batch.header.to_wire()
    for header in (task.header, header_wire):
        with pytest.raises(ValueError, match="^LatentBatch requires its exact artifact header$"):
            replace(batch, header=header)
