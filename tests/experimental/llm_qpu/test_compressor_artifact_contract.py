# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental compressor contracts
"""Exercise train-only fitted bytes and radian map through the isolated worker."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArrayDescriptor,
    ArtifactHeader,
    CompressedLatentBatch,
    CompressorArtifact,
    LatentBatch,
    ModelDescriptor,
    SplitManifest,
    TaskSpec,
    canonical_bytes,
    decode_contract,
    validate_compressed_latent_batch,
    validate_compressor_artifact,
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


def _array(values: tuple[float, ...], shape: tuple[int, ...]) -> tuple[ArrayDescriptor, bytes]:
    payload = struct.pack(f"<{len(values)}f", *values)
    return (
        ArrayDescriptor("<f4", shape, len(payload), hashlib.sha256(payload).hexdigest()),
        payload,
    )


def _records() -> tuple[TaskSpec, SplitManifest, ModelDescriptor, LatentBatch, bytes]:
    task_fields = {
        "task_id": "synthetic-compression-task",
        "objective": "Predict a label from a prefix",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": "balanced_accuracy",
        "group_definition": "One generated scene per source group",
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
        "model_id": "synthetic-hidden-source",
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
    tensor, payload = _array((0.1, 0.2, 0.3, 0.4, -0.2, 0.1, 0.0, 0.2), (2, 4))
    latent_fields = {
        "task_digest": _digest(task.to_wire()),
        "split_digest": _digest(split.to_wire()),
        "model_digest": _digest(model.to_wire()),
        "split_name": "train",
        "layout": "contextual",
        "sample_ids": ("sample-a", "sample-b"),
        "source_ids": ("source-a", "source-b"),
        "group_ids": ("group-a", "group-b"),
        "lengths": (1, 1),
        "mask": ((True,), (True,)),
        "token_positions": ((0,), (0,)),
        "answer_start_positions": ((1,), (1,)),
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    latent_content = {
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
            for name, value in latent_fields.items()
        },
    }
    latent = LatentBatch(
        **latent_fields,
        header=_header(
            "latent_batch",
            latent_content,
            (
                latent_fields["task_digest"],
                latent_fields["split_digest"],
                latent_fields["model_digest"],
                tensor.sha256,
            ),
            "synthetic_classical",
        ),
    )
    return task, split, model, latent, payload


def _compressor(
    split: SplitManifest, model: ModelDescriptor, *, scales: tuple[float, ...] = (10.0,) * 4
) -> tuple[CompressorArtifact, bytes, bytes, bytes]:
    center, center_payload = _array((0.0,) * 4, (4,))
    projection, projection_payload = _array(
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (4, 4),
    )
    scale, scales_payload = _array(scales, (4,))
    fields = {
        "model_digest": _digest(model.to_wire()),
        "split_digest": _digest(split.to_wire()),
        "train_group_digest": _digest(list(split.train_groups)),
        "software_digest": "2" * 64,
        "fit_seed": 23,
        "fit_method": "pca_train_svd_v1",
        "map_id": "rms_normalize_center_project_tanh_v2",
        "scale_policy": "block_zero_variance",
        "input_width": 4,
        "output_width": 4,
        "epsilon_hex": (1e-6).hex(),
        "centering": center,
        "projection": projection,
        "scales": scale,
    }
    content = {
        "schema": "scpn.experimental.llm_qpu.compressor_artifact.v1",
        "object_kind": "compressor_artifact",
        **{
            name: value.to_wire() if name in ("centering", "projection", "scales") else value
            for name, value in fields.items()
        },
    }
    compressor = CompressorArtifact(
        **fields,
        header=_header(
            "compressor_artifact",
            content,
            (
                fields["model_digest"],
                fields["split_digest"],
                fields["train_group_digest"],
                fields["software_digest"],
                center.sha256,
                projection.sha256,
                scale.sha256,
            ),
            "synthetic_classical",
        ),
    )
    return compressor, center_payload, projection_payload, scales_payload


def _compressed(
    latent: LatentBatch,
    compressor: CompressorArtifact,
    source_payload: bytes,
    *,
    extra_pi: bool = False,
) -> tuple[CompressedLatentBatch, bytes]:
    values = struct.unpack(f"<{len(source_payload) // 4}f", source_payload)
    angles: list[float] = []
    steps = 1 if latent.layout == "contextual" else latent.tensor.shape[1]
    for row_index, length in enumerate(latent.lengths):
        for step in range(steps):
            if step >= length:
                angles.extend((0.0,) * 4)
                continue
            offset = (row_index * steps + step) * 4
            source = values[offset : offset + 4]
            rms = math.sqrt(
                sum(value * value for value in source) / 4 + float.fromhex(compressor.epsilon_hex)
            )
            angles.extend(
                (math.pi / 2) * math.tanh((value / rms) / 10) * (math.pi if extra_pi else 1)
                for value in source
            )
    tensor, payload = _array(tuple(angles), (*latent.tensor.shape[:-1], 4))
    fields = {
        "latent_digest": _digest(latent.to_wire()),
        "compressor_digest": _digest(compressor.to_wire()),
        "split_name": latent.split_name,
        "layout": latent.layout,
        "sample_ids": latent.sample_ids,
        "source_ids": latent.source_ids,
        "group_ids": latent.group_ids,
        "lengths": latent.lengths,
        "mask": latent.mask,
        "token_positions": latent.token_positions,
        "answer_start_positions": latent.answer_start_positions,
        "angle_unit": "radian",
        "comparison_tolerance_hex": (1e-6).hex(),
        "tensor": tensor,
    }
    content = {
        "schema": "scpn.experimental.llm_qpu.compressed_latent_batch.v1",
        "object_kind": "compressed_latent_batch",
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
    compressed = CompressedLatentBatch(
        **fields,
        header=_header(
            "compressed_latent_batch",
            content,
            (fields["latent_digest"], fields["compressor_digest"], tensor.sha256),
            "synthetic_classical",
        ),
    )
    return compressed, payload


def _request(
    task: TaskSpec,
    split: SplitManifest,
    model: ModelDescriptor,
    latent: LatentBatch,
    compressor: CompressorArtifact,
    compressed: CompressedLatentBatch,
    payloads: tuple[bytes, bytes, bytes, bytes, bytes],
) -> dict[str, object]:
    latent_payload, center_payload, projection_payload, scales_payload, angles_payload = payloads
    return {
        "op": "roundtrip_compression",
        "task": task.to_wire(),
        "split": split.to_wire(),
        "model": model.to_wire(),
        "latent": latent.to_wire(),
        "latent_base64": base64.b64encode(latent_payload).decode(),
        "compressor": compressor.to_wire(),
        "centering_base64": base64.b64encode(center_payload).decode(),
        "projection_base64": base64.b64encode(projection_payload).decode(),
        "scales_base64": base64.b64encode(scales_payload).decode(),
        "compressed": compressed.to_wire(),
        "angles_base64": base64.b64encode(angles_payload).decode(),
    }


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


def test_compressor_train_lineage_and_worker_roundtrip(tmp_path: Path) -> None:
    task, split, model, latent, latent_payload = _records()
    compressor, center, projection, scales = _compressor(split, model)
    compressed, angles = _compressed(latent, compressor, latent_payload)
    validate_compressed_latent_batch(
        task,
        split,
        model,
        latent,
        compressor,
        compressed,
        latent_payload,
        center,
        projection,
        scales,
        angles,
    )
    assert decode_contract(canonical_bytes(compressor.to_wire())) == compressor
    assert decode_contract(canonical_bytes(compressed.to_wire())) == compressed
    response = _worker(
        _request(
            task,
            split,
            model,
            latent,
            compressor,
            compressed,
            (latent_payload, center, projection, scales, angles),
        ),
        tmp_path,
    )
    assert response.returncode == 0, response.stdout.decode()
    result = json.loads(response.stdout)
    assert result["compressed_sha256"] == _digest(compressed.to_wire())
    assert result["hardware_submission_enabled"] is False


def test_compressor_refuses_dev_fit_and_zero_scale() -> None:
    _, split, model, _, _ = _records()
    compressor, center, projection, scales = _compressor(split, model)
    forged_group_digest = _digest(list(split.dev_groups))
    forged_fields = {
        **compressor._scientific_wire(),
        "train_group_digest": forged_group_digest,
    }
    forged_header = _header(
        "compressor_artifact",
        forged_fields,
        (
            compressor.model_digest,
            compressor.split_digest,
            forged_group_digest,
            compressor.software_digest,
            compressor.centering.sha256,
            compressor.projection.sha256,
            compressor.scales.sha256,
        ),
        "synthetic_classical",
    )
    forged = replace(compressor, train_group_digest=forged_group_digest, header=forged_header)
    with pytest.raises(ValueError, match="fit groups"):
        validate_compressor_artifact(split, model, forged, center, projection, scales)
    zero_scale, _, _, zero_scales = _compressor(split, model, scales=(10.0, 0.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="positive"):
        validate_compressor_artifact(split, model, zero_scale, center, projection, zero_scales)


def test_compressed_batch_refuses_extra_pi_and_row_drift(tmp_path: Path) -> None:
    task, split, model, latent, latent_payload = _records()
    compressor, center, projection, scales = _compressor(split, model)
    wrong, wrong_payload = _compressed(latent, compressor, latent_payload, extra_pi=True)
    with pytest.raises(ValueError, match="frozen map"):
        validate_compressed_latent_batch(
            task,
            split,
            model,
            latent,
            compressor,
            wrong,
            latent_payload,
            center,
            projection,
            scales,
            wrong_payload,
        )
    response = _worker(
        _request(
            task,
            split,
            model,
            latent,
            compressor,
            wrong,
            (latent_payload, center, projection, scales, wrong_payload),
        ),
        tmp_path,
    )
    assert response.returncode == 2
    assert "frozen map" in json.loads(response.stdout)["reason"]
    drift_fields = {**wrong._scientific_wire(), "sample_ids": ["sample-b", "sample-a"]}
    drift = replace(
        wrong,
        sample_ids=("sample-b", "sample-a"),
        header=_header(
            "compressed_latent_batch",
            drift_fields,
            (wrong.latent_digest, wrong.compressor_digest, wrong.tensor.sha256),
            "synthetic_classical",
        ),
    )
    with pytest.raises(ValueError, match="sample_ids differs"):
        validate_compressed_latent_batch(
            task,
            split,
            model,
            latent,
            compressor,
            drift,
            latent_payload,
            center,
            projection,
            scales,
            wrong_payload,
        )


def test_compressed_sequence_retains_zero_padding(tmp_path: Path) -> None:
    task, split, model, original, _ = _records()
    tensor, latent_payload = _array(
        (
            0.1,
            0.2,
            0.3,
            0.4,
            0.2,
            0.3,
            0.4,
            0.5,
            -0.2,
            0.1,
            0.0,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (2, 2, 4),
    )
    fields = {
        **original._scientific_wire(),
        "layout": "chunk_isolated",
        "lengths": [2, 1],
        "mask": [[True, True], [True, False]],
        "token_positions": [[0, 0], [0, None]],
        "answer_start_positions": [[1, 1], [1, None]],
        "tensor": tensor.to_wire(),
    }
    latent = replace(
        original,
        layout="chunk_isolated",
        lengths=(2, 1),
        mask=((True, True), (True, False)),
        token_positions=((0, 0), (0, None)),
        answer_start_positions=((1, 1), (1, None)),
        tensor=tensor,
        header=_header(
            "latent_batch",
            fields,
            (original.task_digest, original.split_digest, original.model_digest, tensor.sha256),
            "synthetic_classical",
        ),
    )
    compressor, center, projection, scales = _compressor(split, model)
    compressed, angles = _compressed(latent, compressor, latent_payload)
    validate_compressed_latent_batch(
        task,
        split,
        model,
        latent,
        compressor,
        compressed,
        latent_payload,
        center,
        projection,
        scales,
        angles,
    )
    response = _worker(
        _request(
            task,
            split,
            model,
            latent,
            compressor,
            compressed,
            (latent_payload, center, projection, scales, angles),
        ),
        tmp_path,
    )
    assert response.returncode == 0, response.stdout.decode()
    corrupted = bytearray(angles)
    struct.pack_into("<f", corrupted, 12 * 4, 0.01)
    corrupt_tensor = ArrayDescriptor(
        "<f4", (2, 2, 4), len(corrupted), hashlib.sha256(corrupted).hexdigest()
    )
    corrupt_fields = {**compressed._scientific_wire(), "tensor": corrupt_tensor.to_wire()}
    forged = replace(
        compressed,
        tensor=corrupt_tensor,
        header=_header(
            "compressed_latent_batch",
            corrupt_fields,
            (compressed.latent_digest, compressed.compressor_digest, corrupt_tensor.sha256),
            "synthetic_classical",
        ),
    )
    with pytest.raises(ValueError, match="padding must be zero"):
        validate_compressed_latent_batch(
            task,
            split,
            model,
            latent,
            compressor,
            forged,
            latent_payload,
            center,
            projection,
            scales,
            bytes(corrupted),
        )
