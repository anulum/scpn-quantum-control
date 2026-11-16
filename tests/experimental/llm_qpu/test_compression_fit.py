# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU compressor fit
"""Fit the train-only PCA compressor and compress batches through the frozen contracts."""

from __future__ import annotations

import hashlib
import math
import platform
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    ArrayDescriptor,
    ArtifactHeader,
    LatentBatch,
    ModelDescriptor,
    SplitManifest,
    TaskSpec,
    canonical_bytes,
)
from scpn_quantum_control.experimental.llm_qpu.data.compression_fit import (
    compress_latents,
    fit_train_compressor,
)

_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_REVISIONS = (_BASE, _REVISION)
_WIDTH = 8
_TRAIN = tuple(f"train-{index:02d}" for index in range(10))
_DEV = ("dev-00", "dev-01", "dev-02")


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


def _design(train: tuple[str, ...] = _TRAIN) -> tuple[TaskSpec, SplitManifest, ModelDescriptor]:
    """Return a frozen task, a train split of the given groups and an eight-wide model."""
    task_fields: dict[str, Any] = {
        "task_id": "synthetic-compression-task",
        "objective": "Predict a label from a prefix",
        "source_kind": "synthetic_classical",
        "target_origin": "classical_generator",
        "label_schema_digest": "a" * 64,
        "causal_cutoff": 0,
        "primary_metric": "accuracy",
        "group_definition": "One generated scene per source group",
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
        "train_groups": train,
        "dev_groups": _DEV,
        "test_groups": ("test-00",),
        "seed": 19,
        "dedup_rule": "exact_source_digest",
        "test_target_custodian": "separate_locked_evaluator",
        "transform_fit_split": "train_only",
    }
    split_content = {
        "schema": "scpn.experimental.llm_qpu.split_manifest.v2",
        "object_kind": "split_manifest",
        **{k: list(v) if k.endswith("_groups") else v for k, v in split_fields.items()},
    }
    split = SplitManifest(
        **split_fields,
        header=_header(
            "split_manifest",
            split_content,
            (split_fields["task_digest"], "b" * 64),
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
        "hidden_width": _WIDTH,
        "tap_block_index": 2,
        "tap_stream": "residual_hidden_state",
        "tap_boundary": "before_norm",
        "probe_evidence_digest": "1" * 64,
    }
    model_content = {
        "schema": "scpn.experimental.llm_qpu.model_descriptor.v1",
        "object_kind": "model_descriptor",
        **model_fields,
    }
    digests = (
        "checkpoint_digest",
        "tokenizer_digest",
        "chat_template_digest",
        "runtime_build_digest",
        "probe_evidence_digest",
    )
    model = ModelDescriptor(
        **model_fields,
        header=_header(
            "model_descriptor",
            model_content,
            tuple(model_fields[n] for n in digests),
            "owner_checkpoint",
        ),
    )
    return task, split, model


def _latent(
    design: tuple[TaskSpec, SplitManifest, ModelDescriptor],
    values: np.ndarray[Any, Any],
    split_name: str,
    groups: tuple[str, ...],
    *,
    lengths: tuple[int, ...] | None = None,
) -> tuple[LatentBatch, bytes]:
    """Return a contextual batch, or a chunk-isolated one when ``lengths`` is given."""
    task, split, model = design
    payload = np.ascontiguousarray(values, dtype="<f4").tobytes()
    tensor = ArrayDescriptor(
        "<f4", tuple(values.shape), len(payload), hashlib.sha256(payload).hexdigest()
    )
    rows = values.shape[0]
    steps = 1 if lengths is None else values.shape[1]
    lengths = lengths or (1,) * rows
    fields: dict[str, Any] = {
        "task_digest": _sha(task.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "model_digest": _sha(model.to_wire()),
        "split_name": split_name,
        "layout": "contextual" if steps == 1 and values.ndim == 2 else "chunk_isolated",
        "sample_ids": tuple(f"sample-{i:02d}" for i in range(rows)),
        "source_ids": tuple(f"source-{i:02d}" for i in range(rows)),
        "group_ids": groups,
        "lengths": lengths,
        "mask": tuple(tuple(step < n for step in range(steps)) for n in lengths),
        "token_positions": tuple(
            tuple(0 if step < n else None for step in range(steps)) for n in lengths
        ),
        "answer_start_positions": tuple(
            tuple(1 if step < n else None for step in range(steps)) for n in lengths
        ),
        "tap_block_index": 2,
        "tap_boundary": "before_norm",
        "tensor": tensor,
    }
    nested = ("mask", "token_positions", "answer_start_positions")
    content = {
        "schema": "scpn.experimental.llm_qpu.latent_batch.v1",
        "object_kind": "latent_batch",
        **{
            name: value.to_wire()
            if name == "tensor"
            else [list(row) for row in value]
            if name in nested
            else list(value)
            if type(value) is tuple
            else value
            for name, value in fields.items()
        },
    }
    parents = (
        fields["task_digest"],
        fields["split_digest"],
        fields["model_digest"],
        tensor.sha256,
    )
    return LatentBatch(
        **fields, header=_header("latent_batch", content, parents, "synthetic_classical")
    ), payload


def _train_values(rows: int = 10) -> np.ndarray[Any, Any]:
    """Return deterministic float32 train latents."""
    return np.random.default_rng(7).normal(size=(rows, _WIDTH)).astype(np.float32)


def _oracle(
    values: np.ndarray[Any, Any], width: int, epsilon: float = 1e-6
) -> tuple[bytes, bytes, bytes]:
    """Recompute the probe's fit independently and freeze it as float32 bytes."""
    data = values.astype(np.float64)
    normalized = data / np.sqrt(np.mean(data * data, axis=1, keepdims=True) + epsilon)
    center = normalized.mean(axis=0)
    _, _, vh = np.linalg.svd(normalized - center, full_matrices=False)
    projection = vh[:width].T.copy()
    for column in range(width):
        if projection[np.argmax(np.abs(projection[:, column])), column] < 0:
            projection[:, column] = -projection[:, column]
    scales = ((normalized - center) @ projection).std(axis=0, ddof=1)
    frozen = [np.ascontiguousarray(a, dtype="<f4").tobytes() for a in (center, projection, scales)]
    return frozen[0], frozen[1], frozen[2]


@pytest.mark.parametrize("output_width", [4, 8])
def test_fit_matches_the_probe_oracle_and_passes_the_frozen_contracts(output_width: int) -> None:
    """The packaged fit reproduces the probe's bytes and its compressed batches validate."""
    design = _design()
    task, split, model = design
    train, train_payload = _latent(design, _train_values(), "train", _TRAIN)
    fitted = fit_train_compressor(
        task,
        split,
        model,
        train,
        train_payload,
        output_width=output_width,
        fit_seed=23,
        revisions=_REVISIONS,
    )
    expected = _oracle(_train_values(), output_width)
    assert (fitted.centering_payload, fitted.projection_payload, fitted.scales_payload) == expected
    artifact = fitted.artifact
    assert (artifact.fit_method, artifact.map_id, artifact.scale_policy) == (
        "pca_train_svd_v1",
        "rms_normalize_center_project_tanh_v2",
        "block_zero_variance",
    )
    assert (artifact.input_width, artifact.output_width, artifact.fit_seed) == (
        _WIDTH,
        output_width,
        23,
    )
    assert artifact.train_group_digest == _sha(list(_TRAIN))
    assert artifact.header.data_origin == "synthetic_classical"
    assert (
        artifact.header.base_repo_commit,
        artifact.header.implementation_revision,
    ) == _REVISIONS
    software = {
        "fit": "pca_train_svd_v1",
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    assert artifact.software_digest == _sha(software)
    dev_values = np.random.default_rng(11).normal(size=(3, _WIDTH)).astype(np.float32)
    dev, dev_payload = _latent(design, dev_values, "dev", _DEV)
    for batch, payload in ((train, train_payload), (dev, dev_payload)):
        compressed, angles = compress_latents(
            task, split, model, batch, payload, fitted, revisions=_REVISIONS
        )
        assert (
            compressed.split_name == batch.split_name and compressed.sample_ids == batch.sample_ids
        )
        values = np.frombuffer(angles, dtype="<f4")
        assert values.shape == (batch.tensor.shape[0] * output_width,)
        assert bool(np.all(np.abs(values) < math.pi / 2))


def test_compress_keeps_sequence_padding_zero() -> None:
    """A chunk-isolated batch keeps zero angles past each row's length."""
    design = _design()
    task, split, model = design
    train, train_payload = _latent(design, _train_values(), "train", _TRAIN)
    fitted = fit_train_compressor(
        task, split, model, train, train_payload, output_width=4, fit_seed=0, revisions=_REVISIONS
    )
    values = np.random.default_rng(5).normal(size=(2, 3, _WIDTH)).astype(np.float32)
    values[0, 2:] = 0.0
    values[1, 1:] = 0.0
    sequence, payload = _latent(design, values, "dev", ("dev-00", "dev-01"), lengths=(2, 1))
    compressed, angles = compress_latents(
        task, split, model, sequence, payload, fitted, revisions=_REVISIONS
    )
    grid = np.frombuffer(angles, dtype="<f4").reshape(2, 3, 4)
    assert compressed.layout == "chunk_isolated"
    assert bool(np.all(grid[0, 2] == 0.0)) and bool(np.all(grid[1, 1:] == 0.0))
    assert bool(np.all(grid[0, :2] != 0.0)) and bool(np.all(grid[1, 0] != 0.0))


def test_fit_refuses_non_train_sequence_partial_small_and_degenerate_batches() -> None:
    """The fit sees exactly the train groups, enough rows and nonzero variance."""
    design = _design()
    task, split, model = design
    cases: list[tuple[LatentBatch, bytes, int, str]] = []
    dev, dev_payload = _latent(
        design, np.random.default_rng(1).normal(size=(3, _WIDTH)).astype(np.float32), "dev", _DEV
    )
    cases.append((dev, dev_payload, 4, "^a compressor fits on the train split only$"))
    sequence_values = np.random.default_rng(2).normal(size=(10, 2, _WIDTH)).astype(np.float32)
    sequence, sequence_payload = _latent(
        design, sequence_values, "train", _TRAIN, lengths=(2,) * 10
    )
    cases.append(
        (sequence, sequence_payload, 4, "^v1 fits contextual batches; sequence fitting needs")
    )
    partial, partial_payload = _latent(design, _train_values(9), "train", _TRAIN[:9])
    cases.append(
        (partial, partial_payload, 4, "^the fit batch must cover every train group and nothing")
    )
    train, train_payload = _latent(design, _train_values(), "train", _TRAIN)
    cases.append((train, train_payload, 5, "^compressor output width must be four or eight$"))
    flat = np.tile(_train_values(1), (10, 1))
    flat_batch, flat_payload = _latent(design, flat, "train", _TRAIN)
    cases.append(
        (flat_batch, flat_payload, 4, "^a component has zero train variance; block_zero_variance")
    )
    leaked, leaked_payload = _latent(design, _train_values(), "train", (*_TRAIN[:9], "dev-00"))
    cases.append((leaked, leaked_payload, 4, "^latent group outside frozen split$"))
    for batch, payload, width, reason in cases:
        with pytest.raises(ValueError, match=reason):
            fit_train_compressor(
                task,
                split,
                model,
                batch,
                payload,
                output_width=width,
                fit_seed=0,
                revisions=_REVISIONS,
            )
    small_design = _design(_TRAIN[:4])
    small, small_payload = _latent(small_design, _train_values(4), "train", _TRAIN[:4])
    with pytest.raises(ValueError, match="^the fit needs more rows than output components$"):
        fit_train_compressor(
            *small_design, small, small_payload, output_width=4, fit_seed=0, revisions=_REVISIONS
        )
