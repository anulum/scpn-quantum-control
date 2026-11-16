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
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

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


def _array(values: tuple[float, ...], shape: tuple[int, ...]) -> tuple[ArrayDescriptor, bytes]:
    """Return a float32 descriptor of the given shape and its exact payload."""
    payload = struct.pack(f"<{len(values)}f", *values)
    return (
        ArrayDescriptor("<f4", shape, len(payload), hashlib.sha256(payload).hexdigest()),
        payload,
    )


def _records() -> tuple[TaskSpec, SplitManifest, ModelDescriptor, LatentBatch, bytes]:
    """Return frozen task, split and model records and a two-row contextual latent batch."""
    task_fields: dict[str, Any] = {
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
    latent_fields: dict[str, Any] = {
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
    """Return an identity-projection compressor fitted on the train groups and its payloads."""
    centre, centre_payload = _array((0.0,) * 4, (4,))
    projection, projection_payload = _array(
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (4, 4),
    )
    scale, scales_payload = _array(scales, (4,))
    fields: dict[str, Any] = {
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
        "centering": centre,
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
                centre.sha256,
                projection.sha256,
                scale.sha256,
            ),
            "synthetic_classical",
        ),
    )
    return compressor, centre_payload, projection_payload, scales_payload


def _compressed(
    latent: LatentBatch,
    compressor: CompressorArtifact,
    source_payload: bytes,
    *,
    extra_pi: bool = False,
) -> tuple[CompressedLatentBatch, bytes]:
    """Return the batch compressed by the frozen map, optionally with an extra factor of pi."""
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
    fields: dict[str, Any] = {
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
    """Return the worker request that round-trips one compression."""
    latent_payload, centre_payload, projection_payload, scales_payload, angles_payload = payloads
    return {
        "op": "roundtrip_compression",
        "task": task.to_wire(),
        "split": split.to_wire(),
        "model": model.to_wire(),
        "latent": latent.to_wire(),
        "latent_base64": base64.b64encode(latent_payload).decode(),
        "compressor": compressor.to_wire(),
        "centering_base64": base64.b64encode(centre_payload).decode(),
        "projection_base64": base64.b64encode(projection_payload).decode(),
        "scales_base64": base64.b64encode(scales_payload).decode(),
        "compressed": compressed.to_wire(),
        "angles_base64": base64.b64encode(angles_payload).decode(),
    }


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


def test_compressor_train_lineage_and_worker_roundtrip(tmp_path: Path) -> None:
    """Check the train-only lineage of a compressor and its worker round trip."""
    task, split, model, latent, latent_payload = _records()
    compressor, centre, projection, scales = _compressor(split, model)
    compressed, angles = _compressed(latent, compressor, latent_payload)
    validate_compressed_latent_batch(
        task,
        split,
        model,
        latent,
        compressor,
        compressed,
        latent_payload,
        centre,
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
            (latent_payload, centre, projection, scales, angles),
        ),
        tmp_path,
    )
    assert response.returncode == 0, response.stdout.decode()
    result = json.loads(response.stdout)
    assert result["compressed_sha256"] == _digest(compressed.to_wire())
    assert result["hardware_submission_enabled"] is False


def test_compressor_refuses_dev_fit_and_zero_scale() -> None:
    """Refuse a compressor fitted on dev groups and a zero scale."""
    _, split, model, _, _ = _records()
    compressor, centre, projection, scales = _compressor(split, model)
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
        validate_compressor_artifact(split, model, forged, centre, projection, scales)
    zero_scale, _, _, zero_scales = _compressor(split, model, scales=(10.0, 0.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="positive"):
        validate_compressor_artifact(split, model, zero_scale, centre, projection, zero_scales)


def test_compressed_batch_refuses_extra_pi_and_row_drift(tmp_path: Path) -> None:
    """Refuse a compressed batch with an extra pi or rows that drift from the latent batch."""
    task, split, model, latent, latent_payload = _records()
    compressor, centre, projection, scales = _compressor(split, model)
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
            centre,
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
            (latent_payload, centre, projection, scales, wrong_payload),
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
            centre,
            projection,
            scales,
            wrong_payload,
        )


def test_compressed_sequence_retains_zero_padding(tmp_path: Path) -> None:
    """Check that a compressed sequence keeps its zero padding."""
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
    fields: dict[str, Any] = {
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
    compressor, centre, projection, scales = _compressor(split, model)
    compressed, angles = _compressed(latent, compressor, latent_payload)
    validate_compressed_latent_batch(
        task,
        split,
        model,
        latent,
        compressor,
        compressed,
        latent_payload,
        centre,
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
            (latent_payload, centre, projection, scales, angles),
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
            centre,
            projection,
            scales,
            bytes(corrupted),
        )


def _wire_value(value: object) -> object:
    """Return the wire form of one batch field value."""
    if isinstance(value, ArrayDescriptor):
        return value.to_wire()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    return value


def _rebind_compressed(
    compressed: CompressedLatentBatch, *, origin: str = "synthetic_classical", **changes: Any
) -> CompressedLatentBatch:
    """Return the batch with changed fields and a header bound to the changed content."""
    fields = {item.name: getattr(compressed, item.name) for item in dataclass_fields(compressed)}
    fields.update(changes)
    content = {name: value for name, value in compressed.to_wire().items() if name != "header"} | {
        name: _wire_value(value) for name, value in changes.items()
    }
    tensor: ArrayDescriptor = fields["tensor"]
    parents = (fields["latent_digest"], fields["compressor_digest"], tensor.sha256)
    fields["header"] = _header("compressed_latent_batch", content, parents, origin)
    return CompressedLatentBatch(**fields)


def _compression() -> tuple[
    tuple[TaskSpec, SplitManifest, ModelDescriptor, LatentBatch, CompressorArtifact],
    CompressedLatentBatch,
    tuple[bytes, bytes, bytes, bytes, bytes],
]:
    """Return the frozen records, a valid compressed batch, and every payload in validation order."""
    task, split, model, latent, latent_payload = _records()
    compressor, centre, projection, scales = _compressor(split, model)
    compressed, angles = _compressed(latent, compressor, latent_payload)
    return (
        (task, split, model, latent, compressor),
        compressed,
        (latent_payload, centre, projection, scales, angles),
    )


_SEQUENCE = ArrayDescriptor("<f4", (2, 2, 4), 64, "0" * 64)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split_name": "validation"}, "unknown compressed split"),
        ({"layout": "flat"}, "unknown compressed layout"),
        ({"angle_unit": "degree"}, "compressed angles must be radians"),
        ({"comparison_tolerance_hex": (1e-4).hex()}, "angle tolerance"),
        ({"tensor": ArrayDescriptor("<f8", (2, 4), 64, "0" * 64)}, "compressed tensor must be"),
        ({"tensor": "descriptor"}, "compressed tensor must be little-endian float32"),
        ({"layout": "chunk_isolated"}, "compressed tensor rank differs from layout"),
        ({"sample_ids": ("sample-a",)}, "compressed row identity length mismatch"),
        ({"lengths": [1, 1]}, "compressed row identity length mismatch"),
        ({"sample_ids": ("same", "same")}, "duplicate compressed sample ID"),
        ({"mask": ((True, True), (True,))}, "compressed step identity length mismatch"),
        ({"answer_start_positions": [(1,), (1,)]}, "compressed step identity length mismatch"),
        ({"lengths": (2, 1)}, "compressed length exceeds steps"),
        ({"lengths": (0, 1)}, "compressed length exceeds steps"),
        ({"lengths": (True, 1)}, "compressed length exceeds steps"),
        ({"mask": ((True,), (1,))}, "compressed mask and length disagree"),
        ({"token_positions": ((1,), (0,))}, "compressed token position reaches answer"),
        ({"token_positions": ((None,), (0,))}, "compressed token position reaches answer"),
        (
            {
                "layout": "chunk_isolated",
                "tensor": _SEQUENCE,
                "lengths": (1, 1),
                "mask": ((True, False), (True, False)),
                "token_positions": ((0, 0), (0, None)),
                "answer_start_positions": ((1, None), (1, None)),
            },
            "compressed padding carries token positions",
        ),
    ],
)
def test_compressed_batch_refuses_malformed_rows(changes: dict[str, Any], message: str) -> None:
    """A batch refuses bad splits, layouts, units, tolerances, tensors, row identities and steps."""
    _, compressed, _ = _compression()
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(compressed, **changes)


def test_compressed_batch_refuses_a_header_of_another_kind() -> None:
    """A batch accepts only its own compressed-batch header."""
    (_, _, _, latent, _), compressed, _ = _compression()
    header_wire: Any = compressed.header.to_wire()
    for header in (latent.header, header_wire):
        with pytest.raises(ValueError, match="^CompressedLatentBatch requires exact header$"):
            replace(compressed, header=header)


def test_compressed_batch_wire_refusals_through_the_decoder() -> None:
    """The decoder refuses an edited compressed wire field by field."""
    _, compressed, _ = _compression()
    encoded = compressed.to_wire()
    cases: list[tuple[dict[str, object], str]] = [
        ({**encoded, "extra": 1}, "CompressedLatentBatch fields mismatch"),
        ({**encoded, "object_kind": "latent_batch"}, "unknown CompressedLatentBatch schema"),
        ({**encoded, "lengths": 1}, "compressed rows and steps must be lists on wire"),
        ({**encoded, "mask": [{"step": True}, [True]]}, "compressed rows and steps must be lists"),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=f"^{message}"):
            decode_contract(canonical_bytes(value))


def test_compressed_validation_refuses_batches_that_do_not_bind() -> None:
    """Validation needs the frozen batch bound to its latent, compressor, shape, origin and angles."""
    records, compressed, payloads = _compression()
    loose: Any = compressed.to_wire()
    with pytest.raises(ValueError, match="^compressed validation requires frozen contract$"):
        validate_compressed_latent_batch(*records, loose, *payloads)
    angles = struct.unpack("<8f", payloads[4])
    outside, outside_payload = _array((1.6, *angles[1:]), (2, 4))
    cases: list[tuple[CompressedLatentBatch, bytes, str]] = [
        (
            _rebind_compressed(compressed, latent_digest="9" * 64),
            payloads[4],
            "source latent digest",
        ),
        (
            _rebind_compressed(compressed, compressor_digest="9" * 64),
            payloads[4],
            "compressor digest",
        ),
        (
            _rebind_compressed(compressed, tensor=ArrayDescriptor("<f4", (2, 3), 24, "0" * 64)),
            payloads[4],
            "output shape mismatch",
        ),
        (
            _rebind_compressed(compressed, origin="owner_dataset"),
            payloads[4],
            "data origin mismatch",
        ),
        (_rebind_compressed(compressed, tensor=outside), outside_payload, "open radian interval"),
    ]
    for candidate, angle_payload, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_compressed_latent_batch(*records, candidate, *payloads[:4], angle_payload)


def _rebind_compressor(
    compressor: CompressorArtifact, *, origin: str = "synthetic_classical", **changes: Any
) -> CompressorArtifact:
    """Return the compressor with changed fields and a header bound to the changed content."""
    fields = {item.name: getattr(compressor, item.name) for item in dataclass_fields(compressor)}
    fields.update(changes)
    content = {name: value for name, value in compressor.to_wire().items() if name != "header"} | {
        name: _wire_value(value) for name, value in changes.items()
    }
    parents = (
        fields["model_digest"],
        fields["split_digest"],
        fields["train_group_digest"],
        fields["software_digest"],
        fields["centering"].sha256,
        fields["projection"].sha256,
        fields["scales"].sha256,
    )
    fields["header"] = _header("compressor_artifact", content, parents, origin)
    return CompressorArtifact(**fields)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fit_seed": -1}, "invalid compressor fit seed"),
        ({"fit_seed": 2**63}, "invalid compressor fit seed"),
        ({"fit_seed": True}, "invalid compressor fit seed"),
        ({"fit_method": "pca_full_svd_v1"}, "unknown compressor fit method"),
        ({"map_id": "rms_project_tanh_v1"}, "unknown compressor map"),
        ({"scale_policy": "unit_variance"}, "unknown compressor scale policy"),
        ({"output_width": 6}, "compressor output width must be four or eight"),
        ({"output_width": 8}, "compressor output width must be four or eight"),
        ({"epsilon_hex": 1e-6}, "compressor epsilon must be canonical finite hex"),
        ({"epsilon_hex": "zz"}, "compressor epsilon must be canonical finite hex"),
        ({"epsilon_hex": "0x1p99999"}, "compressor epsilon must be canonical finite hex"),
        ({"epsilon_hex": (2.0).hex()}, "compressor epsilon must be canonical bounded positive"),
        ({"epsilon_hex": "0x1p-20"}, "compressor epsilon must be canonical bounded positive"),
        ({"centering": ArrayDescriptor("<f4", (3,), 12, "0" * 64)}, "compressor centering must"),
        ({"projection": ArrayDescriptor("<f8", (4, 4), 128, "0" * 64)}, "compressor projection"),
        ({"scales": "descriptor"}, "compressor scales must be exact float32 shape"),
    ],
)
def test_compressor_refuses_unfrozen_fit_metadata_and_shapes(
    changes: dict[str, Any], message: str
) -> None:
    """A compressor refuses bad seeds, methods, maps, policies, widths, epsilons and arrays."""
    _, split, model, _, _ = _records()
    compressor, _, _, _ = _compressor(split, model)
    with pytest.raises(ValueError, match=f"^{message}"):
        replace(compressor, **changes)


def test_compressor_refuses_another_header_and_an_edited_wire() -> None:
    """A compressor takes only its own header, and the decoder refuses an edited field set."""
    _, split, model, _, _ = _records()
    compressor, _, _, _ = _compressor(split, model)
    header_wire: Any = compressor.header.to_wire()
    for header in (split.header, header_wire):
        with pytest.raises(ValueError, match="^CompressorArtifact requires exact header$"):
            replace(compressor, header=header)
    encoded = compressor.to_wire()
    for value, message in (
        ({**encoded, "extra": 1}, "CompressorArtifact fields mismatch"),
        ({**encoded, "object_kind": "latent_batch"}, "unknown CompressorArtifact schema"),
    ):
        with pytest.raises(ValueError, match=f"^{message}$"):
            decode_contract(canonical_bytes(value))


def test_compressor_validation_refuses_a_compressor_that_does_not_bind() -> None:
    """Validation needs the frozen compressor bound to the split, model, width and origin."""
    _, split, model, _, _ = _records()
    compressor, centre, projection, scales = _compressor(split, model)
    loose: Any = compressor.to_wire()
    with pytest.raises(ValueError, match="^compressor validation requires frozen contracts$"):
        validate_compressor_artifact(split, model, loose, centre, projection, scales)
    wider_center = ArrayDescriptor("<f4", (8,), 32, "5" * 64)
    wider_projection = ArrayDescriptor("<f4", (8, 4), 128, "6" * 64)
    cases: list[tuple[CompressorArtifact, str]] = [
        (
            _rebind_compressor(compressor, split_digest="9" * 64),
            "compressor split digest mismatch",
        ),
        (
            _rebind_compressor(compressor, model_digest="9" * 64),
            "compressor model digest mismatch",
        ),
        (
            _rebind_compressor(
                compressor, input_width=8, centering=wider_center, projection=wider_projection
            ),
            "compressor input differs from model hidden state",
        ),
        (
            _rebind_compressor(compressor, origin="owner_dataset"),
            "compressor data origin mismatch",
        ),
    ]
    for candidate, message in cases:
        with pytest.raises(ValueError, match=f"^{message}$"):
            validate_compressor_artifact(split, model, candidate, centre, projection, scales)
