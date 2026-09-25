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
from dataclasses import replace
from pathlib import Path

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
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
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
    task_fields = {
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
    split_fields = {
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
    model_fields = {
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
    fields = {
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
    content = {
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
    return {
        "op": "roundtrip_latent_batch",
        "task": task.to_wire(),
        "split": split.to_wire(),
        "model": model.to_wire(),
        "latent": batch.to_wire(),
        "tensor_base64": base64.b64encode(payload).decode("ascii"),
    }


def test_latent_batch_standalone_roundtrip(tmp_path: Path) -> None:
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
    task, split, model = _records()
    sequence, _ = _batch(task, split, model)
    payload = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0)
    tensor = ArrayDescriptor("<f4", (2, 4), len(payload), hashlib.sha256(payload).hexdigest())
    content = {
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
