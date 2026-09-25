# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Stable public imports for provider-free LLM-QPU contracts."""

from __future__ import annotations

from .cells import (
    CellKey,
    PlannedCell,
    attempt_id,
    content_id,
    request_id,
    run_id,
    validate_completion,
)
from .compressed import CompressedLatentBatch, validate_compressed_latent_batch
from .compressor import CompressorArtifact, validate_compressor_artifact
from .latent import LatentBatch, validate_latent_batch
from .model import ModelDescriptor
from .static import (
    ReservoirSpec,
    StaticCircuitPlan,
    build_static_circuit_plan,
    validate_static_circuit_plan,
)
from .task import SplitManifest, TaskSpec, validate_task_split
from .wire import (
    ARRAY_SCHEMA,
    COMPRESSED_SCHEMA,
    COMPRESSOR_SCHEMA,
    HEADER_SCHEMA,
    LATENT_SCHEMA,
    MODEL_SCHEMA,
    RESERVOIR_SCHEMA,
    SCHEMA,
    SPLIT_SCHEMA,
    STATIC_PLAN_SCHEMA,
    TASK_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    _strict_json,
    canonical_bytes,
    f64,
)

__all__ = (
    "SCHEMA",
    "ARRAY_SCHEMA",
    "TASK_SCHEMA",
    "SPLIT_SCHEMA",
    "HEADER_SCHEMA",
    "MODEL_SCHEMA",
    "LATENT_SCHEMA",
    "COMPRESSOR_SCHEMA",
    "COMPRESSED_SCHEMA",
    "RESERVOIR_SCHEMA",
    "STATIC_PLAN_SCHEMA",
    "CellKey",
    "PlannedCell",
    "ArrayDescriptor",
    "ArtifactHeader",
    "TaskSpec",
    "SplitManifest",
    "ModelDescriptor",
    "LatentBatch",
    "CompressorArtifact",
    "CompressedLatentBatch",
    "ReservoirSpec",
    "StaticCircuitPlan",
    "f64",
    "canonical_bytes",
    "content_id",
    "request_id",
    "run_id",
    "attempt_id",
    "validate_completion",
    "validate_task_split",
    "validate_latent_batch",
    "validate_compressor_artifact",
    "validate_compressed_latent_batch",
    "build_static_circuit_plan",
    "validate_static_circuit_plan",
    "decode_contract",
)


def decode_contract(
    raw: bytes,
) -> (
    PlannedCell
    | ArrayDescriptor
    | TaskSpec
    | SplitManifest
    | ModelDescriptor
    | LatentBatch
    | CompressorArtifact
    | CompressedLatentBatch
    | ReservoirSpec
    | StaticCircuitPlan
):
    """Decode only implemented records; all other chapter objects refuse."""
    value = _strict_json(raw)
    if type(value) is not dict:
        raise ValueError("contract wire must be an object")
    if value.get("schema") == SCHEMA:
        return PlannedCell.from_wire(value)
    if value.get("schema") == ARRAY_SCHEMA:
        return ArrayDescriptor.from_wire(value)
    if value.get("schema") == TASK_SCHEMA:
        return TaskSpec.from_wire(value)
    if value.get("schema") == SPLIT_SCHEMA:
        return SplitManifest.from_wire(value)
    if value.get("schema") == MODEL_SCHEMA:
        return ModelDescriptor.from_wire(value)
    if value.get("schema") == LATENT_SCHEMA:
        return LatentBatch.from_wire(value)
    if value.get("schema") == COMPRESSOR_SCHEMA:
        return CompressorArtifact.from_wire(value)
    if value.get("schema") == COMPRESSED_SCHEMA:
        return CompressedLatentBatch.from_wire(value)
    if value.get("schema") == RESERVOIR_SCHEMA:
        return ReservoirSpec.from_wire(value)
    if value.get("schema") == STATIC_PLAN_SCHEMA:
        return StaticCircuitPlan.from_wire(value)
    raise ValueError("unsupported contract schema")
