# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Stable public imports for provider-free LLM-QPU contracts."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from .measurement import (
        MeasurementPlan,
        SampledBasisEstimate,
        build_measurement_plan,
        estimate_sampled_basis,
    )
    from .model import ModelDescriptor
    from .protocol import ExperimentProtocol, validate_experiment_protocol
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
        EXPERIMENT_PROTOCOL_SCHEMA,
        HEADER_SCHEMA,
        LATENT_SCHEMA,
        MEASUREMENT_PLAN_SCHEMA,
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
    | MeasurementPlan
    | ExperimentProtocol
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
    if value.get("schema") == MEASUREMENT_PLAN_SCHEMA:
        return MeasurementPlan.from_wire(value)
    if value.get("schema") == EXPERIMENT_PROTOCOL_SCHEMA:
        return ExperimentProtocol.from_wire(value)
    raise ValueError("unsupported contract schema")


_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "CellKey": ("scpn_quantum_control.experimental.llm_qpu.contracts.cells", "CellKey"),
    "PlannedCell": ("scpn_quantum_control.experimental.llm_qpu.contracts.cells", "PlannedCell"),
    "attempt_id": ("scpn_quantum_control.experimental.llm_qpu.contracts.cells", "attempt_id"),
    "content_id": ("scpn_quantum_control.experimental.llm_qpu.contracts.cells", "content_id"),
    "request_id": ("scpn_quantum_control.experimental.llm_qpu.contracts.cells", "request_id"),
    "run_id": ("scpn_quantum_control.experimental.llm_qpu.contracts.cells", "run_id"),
    "validate_completion": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.cells",
        "validate_completion",
    ),
    "CompressedLatentBatch": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.compressed",
        "CompressedLatentBatch",
    ),
    "validate_compressed_latent_batch": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.compressed",
        "validate_compressed_latent_batch",
    ),
    "CompressorArtifact": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.compressor",
        "CompressorArtifact",
    ),
    "validate_compressor_artifact": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.compressor",
        "validate_compressor_artifact",
    ),
    "LatentBatch": ("scpn_quantum_control.experimental.llm_qpu.contracts.latent", "LatentBatch"),
    "validate_latent_batch": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.latent",
        "validate_latent_batch",
    ),
    "MeasurementPlan": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.measurement",
        "MeasurementPlan",
    ),
    "SampledBasisEstimate": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.measurement",
        "SampledBasisEstimate",
    ),
    "build_measurement_plan": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.measurement",
        "build_measurement_plan",
    ),
    "estimate_sampled_basis": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.measurement",
        "estimate_sampled_basis",
    ),
    "ModelDescriptor": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.model",
        "ModelDescriptor",
    ),
    "ExperimentProtocol": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.protocol",
        "ExperimentProtocol",
    ),
    "validate_experiment_protocol": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.protocol",
        "validate_experiment_protocol",
    ),
    "ReservoirSpec": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.static",
        "ReservoirSpec",
    ),
    "StaticCircuitPlan": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.static",
        "StaticCircuitPlan",
    ),
    "build_static_circuit_plan": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.static",
        "build_static_circuit_plan",
    ),
    "validate_static_circuit_plan": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.static",
        "validate_static_circuit_plan",
    ),
    "SplitManifest": ("scpn_quantum_control.experimental.llm_qpu.contracts.task", "SplitManifest"),
    "TaskSpec": ("scpn_quantum_control.experimental.llm_qpu.contracts.task", "TaskSpec"),
    "validate_task_split": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.task",
        "validate_task_split",
    ),
    "ARRAY_SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "ARRAY_SCHEMA"),
    "COMPRESSED_SCHEMA": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "COMPRESSED_SCHEMA",
    ),
    "COMPRESSOR_SCHEMA": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "COMPRESSOR_SCHEMA",
    ),
    "EXPERIMENT_PROTOCOL_SCHEMA": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "EXPERIMENT_PROTOCOL_SCHEMA",
    ),
    "HEADER_SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "HEADER_SCHEMA"),
    "LATENT_SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "LATENT_SCHEMA"),
    "MEASUREMENT_PLAN_SCHEMA": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "MEASUREMENT_PLAN_SCHEMA",
    ),
    "MODEL_SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "MODEL_SCHEMA"),
    "RESERVOIR_SCHEMA": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "RESERVOIR_SCHEMA",
    ),
    "SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "SCHEMA"),
    "SPLIT_SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "SPLIT_SCHEMA"),
    "STATIC_PLAN_SCHEMA": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "STATIC_PLAN_SCHEMA",
    ),
    "TASK_SCHEMA": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "TASK_SCHEMA"),
    "ArrayDescriptor": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "ArrayDescriptor",
    ),
    "ArtifactHeader": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "ArtifactHeader",
    ),
    "_strict_json": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "_strict_json"),
    "canonical_bytes": (
        "scpn_quantum_control.experimental.llm_qpu.contracts.wire",
        "canonical_bytes",
    ),
    "f64": ("scpn_quantum_control.experimental.llm_qpu.contracts.wire", "f64"),
}

_INLINE_EXPORTS = {"decode_contract": decode_contract}
del globals()["decode_contract"]
_INLINE_DEPENDENCIES = (
    "CellKey",
    "PlannedCell",
    "attempt_id",
    "content_id",
    "request_id",
    "run_id",
    "validate_completion",
    "CompressedLatentBatch",
    "validate_compressed_latent_batch",
    "CompressorArtifact",
    "validate_compressor_artifact",
    "LatentBatch",
    "validate_latent_batch",
    "MeasurementPlan",
    "SampledBasisEstimate",
    "build_measurement_plan",
    "estimate_sampled_basis",
    "ModelDescriptor",
    "ExperimentProtocol",
    "validate_experiment_protocol",
    "ReservoirSpec",
    "StaticCircuitPlan",
    "build_static_circuit_plan",
    "validate_static_circuit_plan",
    "SplitManifest",
    "TaskSpec",
    "validate_task_split",
    "ARRAY_SCHEMA",
    "COMPRESSED_SCHEMA",
    "COMPRESSOR_SCHEMA",
    "EXPERIMENT_PROTOCOL_SCHEMA",
    "HEADER_SCHEMA",
    "LATENT_SCHEMA",
    "MEASUREMENT_PLAN_SCHEMA",
    "MODEL_SCHEMA",
    "RESERVOIR_SCHEMA",
    "SCHEMA",
    "SPLIT_SCHEMA",
    "STATIC_PLAN_SCHEMA",
    "TASK_SCHEMA",
    "ArrayDescriptor",
    "ArtifactHeader",
    "_strict_json",
    "canonical_bytes",
    "f64",
)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    if name in _INLINE_EXPORTS:
        for dependency in _INLINE_DEPENDENCIES:
            __getattr__(dependency)
        globals().update(_INLINE_EXPORTS)
        return _INLINE_EXPORTS[name]
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS) | set(_INLINE_EXPORTS))


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
    "EXPERIMENT_PROTOCOL_SCHEMA",
    "RESERVOIR_SCHEMA",
    "STATIC_PLAN_SCHEMA",
    "MEASUREMENT_PLAN_SCHEMA",
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
    "ExperimentProtocol",
    "ReservoirSpec",
    "StaticCircuitPlan",
    "MeasurementPlan",
    "SampledBasisEstimate",
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
    "validate_experiment_protocol",
    "build_static_circuit_plan",
    "validate_static_circuit_plan",
    "build_measurement_plan",
    "estimate_sampled_basis",
    "decode_contract",
)
