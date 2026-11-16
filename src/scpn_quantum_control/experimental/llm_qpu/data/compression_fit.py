# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU compressor fit
"""Train-only PCA compressor fit and frozen-map compression of latent batches."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..contracts.compressed import CompressedLatentBatch, validate_compressed_latent_batch
from ..contracts.compressor import CompressorArtifact, validate_compressor_artifact
from ..contracts.latent import LatentBatch, validate_latent_batch
from ..contracts.wire import (
    COMPRESSED_SCHEMA,
    COMPRESSOR_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    canonical_bytes,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..contracts.model import ModelDescriptor
    from ..contracts.task import SplitManifest, TaskSpec

_FLOOR = 1e-8


@dataclass(frozen=True, slots=True)
class FittedCompressor:
    """A frozen compressor and the float32 payload bytes its descriptors name."""

    artifact: CompressorArtifact
    centering_payload: bytes
    projection_payload: bytes
    scales_payload: bytes


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _frozen(array: NDArray[np.float64]) -> tuple[ArrayDescriptor, bytes]:
    """Freeze one array as little-endian float32 bytes and its descriptor."""
    import numpy as np

    packed = np.ascontiguousarray(array, dtype="<f4")
    payload = packed.tobytes(order="C")
    descriptor = ArrayDescriptor(
        "<f4", tuple(packed.shape), len(payload), hashlib.sha256(payload).hexdigest()
    )
    return descriptor, payload


def _header(
    kind: str,
    content: dict[str, object],
    parents: tuple[str, ...],
    origin: str,
    revisions: tuple[str, str],
) -> ArtifactHeader:
    """Return the design-only header that hashes exactly this content."""
    return ArtifactHeader(
        object_kind=kind,
        content_digest=_sha(content),
        parents=tuple(sorted(parents)),
        base_repo_commit=revisions[0],
        implementation_revision=revisions[1],
        execution_origin="offline_design",
        data_origin=origin,
        claim_scope="design_only",
    )


def _check_fit_inputs(split: SplitManifest, latent: LatentBatch, output_width: int) -> None:
    """Refuse a non-train, sequence-shaped or too-small batch before any arithmetic."""
    if split.transform_fit_split != "train_only" or latent.split_name != "train":
        raise ValueError("a compressor fits on the train split only")
    if latent.layout != "contextual":
        raise ValueError("v1 fits contextual batches; sequence fitting needs its own rule")
    if set(latent.group_ids) != set(split.train_groups):
        raise ValueError("the fit batch must cover every train group and nothing else")
    if latent.tensor.shape[0] <= output_width:
        raise ValueError("the fit needs more rows than output components")


def _fit_arrays(
    values: NDArray[np.float64], output_width: int, epsilon: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the centre, sign-fixed principal projection and sample-std scales."""
    import numpy as np

    rms = np.sqrt(np.mean(values * values, axis=1, keepdims=True) + epsilon)
    normalized = values / rms
    center = np.mean(normalized, axis=0)
    centered = normalized - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = vh[:output_width, :].T.copy()
    for column in range(output_width):
        if projection[np.argmax(np.abs(projection[:, column])), column] < 0:
            projection[:, column] *= -1.0
    scales = np.std(centered @ projection, axis=0, ddof=1)
    if np.any(scales <= _FLOOR):
        raise ValueError("a component has zero train variance; block_zero_variance refuses it")
    return center, projection, scales


def fit_train_compressor(
    task: TaskSpec,
    split: SplitManifest,
    model: ModelDescriptor,
    latent: LatentBatch,
    latent_payload: bytes,
    *,
    output_width: int,
    fit_seed: int,
    revisions: tuple[str, str],
    epsilon: float = 1e-6,
) -> FittedCompressor:
    """Fit the frozen PCA map on train groups only and validate the result.

    ``revisions`` holds the base repository commit and the implementation revision
    recorded in the artifact header.
    """
    import numpy as np

    validate_latent_batch(task, split, model, latent, latent_payload)
    _check_fit_inputs(split, latent, output_width)
    values = (
        np.frombuffer(latent_payload, dtype="<f4").reshape(latent.tensor.shape).astype(np.float64)
    )
    arrays = [_frozen(array) for array in _fit_arrays(values, output_width, epsilon)]
    software = {
        "fit": "pca_train_svd_v1",
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    fields: dict[str, Any] = {
        "model_digest": _sha(model.to_wire()),
        "split_digest": _sha(split.to_wire()),
        "train_group_digest": _sha(list(split.train_groups)),
        "software_digest": _sha(software),
        "fit_seed": fit_seed,
        "fit_method": "pca_train_svd_v1",
        "map_id": "rms_normalize_center_project_tanh_v2",
        "scale_policy": "block_zero_variance",
        "input_width": model.hidden_width,
        "output_width": output_width,
        "epsilon_hex": epsilon.hex(),
    }
    artifact = _compressor_record(fields, [d for d, _ in arrays], split, revisions)
    payloads = [payload for _, payload in arrays]
    validate_compressor_artifact(split, model, artifact, *payloads)
    return FittedCompressor(artifact, *payloads)


def _compressor_record(
    fields: dict[str, Any],
    arrays: list[ArrayDescriptor],
    split: SplitManifest,
    revisions: tuple[str, str],
) -> CompressorArtifact:
    """Build the compressor record whose header hashes exactly its content."""
    descriptors: dict[str, Any] = dict(
        zip(("centering", "projection", "scales"), arrays, strict=True)
    )
    content = {
        "schema": COMPRESSOR_SCHEMA,
        "object_kind": "compressor_artifact",
        **fields,
        **{name: descriptor.to_wire() for name, descriptor in descriptors.items()},
    }
    lineage = ("model_digest", "split_digest", "train_group_digest", "software_digest")
    parents = (*(fields[name] for name in lineage), *(array.sha256 for array in arrays))
    header = _header("compressor_artifact", content, parents, split.header.data_origin, revisions)
    return CompressorArtifact(**fields, **descriptors, header=header)


def _angles(
    latent: LatentBatch, latent_payload: bytes, fitted: FittedCompressor
) -> NDArray[np.float64]:
    """Map every valid row through the frozen float32 arrays; padding stays zero."""
    import numpy as np

    artifact = fitted.artifact
    epsilon = float.fromhex(artifact.epsilon_hex)
    width, output = artifact.input_width, artifact.output_width
    center = np.frombuffer(fitted.centering_payload, dtype="<f4").astype(np.float64)
    projection = (
        np.frombuffer(fitted.projection_payload, dtype="<f4")
        .astype(np.float64)
        .reshape(width, output)
    )
    scales = np.frombuffer(fitted.scales_payload, dtype="<f4").astype(np.float64)
    steps = 1 if latent.layout == "contextual" else latent.tensor.shape[1]
    values = np.frombuffer(latent_payload, dtype="<f4").astype(np.float64).reshape(-1, width)
    rms = np.sqrt(np.mean(values * values, axis=1, keepdims=True) + epsilon)
    angles = (np.pi / 2) * np.tanh(((values / rms - center) @ projection) / scales)
    for row, length in enumerate(latent.lengths):
        angles[row * steps + length : (row + 1) * steps] = 0.0
    return angles.reshape(*latent.tensor.shape[:-1], output)


def compress_latents(
    task: TaskSpec,
    split: SplitManifest,
    model: ModelDescriptor,
    latent: LatentBatch,
    latent_payload: bytes,
    fitted: FittedCompressor,
    *,
    revisions: tuple[str, str],
    tolerance: float = 1e-6,
) -> tuple[CompressedLatentBatch, bytes]:
    """Compress any split's batch with the frozen map and validate every angle."""
    angles, angles_bytes = _frozen(_angles(latent, latent_payload, fitted))
    compressed = _compressed_record(latent, fitted.artifact, angles, tolerance, revisions)
    validate_compressed_latent_batch(
        task,
        split,
        model,
        latent,
        fitted.artifact,
        compressed,
        latent_payload,
        fitted.centering_payload,
        fitted.projection_payload,
        fitted.scales_payload,
        angles_bytes,
    )
    return compressed, angles_bytes


def _compressed_record(
    latent: LatentBatch,
    artifact: CompressorArtifact,
    angles: ArrayDescriptor,
    tolerance: float,
    revisions: tuple[str, str],
) -> CompressedLatentBatch:
    """Build the compressed record that keeps the latent row identities."""
    fields: dict[str, Any] = {
        "latent_digest": _sha(latent.to_wire()),
        "compressor_digest": _sha(artifact.to_wire()),
        **{
            name: getattr(latent, name)
            for name in (
                "split_name",
                "layout",
                "sample_ids",
                "source_ids",
                "group_ids",
                "lengths",
                "mask",
                "token_positions",
                "answer_start_positions",
            )
        },
        "angle_unit": "radian",
        "comparison_tolerance_hex": tolerance.hex(),
    }
    nested = ("mask", "token_positions", "answer_start_positions")
    content = {
        "schema": COMPRESSED_SCHEMA,
        "object_kind": "compressed_latent_batch",
        **{
            name: [list(row) for row in value]
            if name in nested
            else list(value)
            if type(value) is tuple
            else value
            for name, value in fields.items()
        },
        "tensor": angles.to_wire(),
    }
    parents = (fields["latent_digest"], fields["compressor_digest"], angles.sha256)
    header = _header(
        "compressed_latent_batch", content, parents, latent.header.data_origin, revisions
    )
    return CompressedLatentBatch(**fields, tensor=angles, header=header)
