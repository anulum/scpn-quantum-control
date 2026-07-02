# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- PyTorch Checkpoint Matrix Utilities
"""Long-lived checkpoint matrix utilities for bounded PyTorch phase-QNN modules.

The matrix in this module wraps ``run_torch_module_checkpoint_audit(...)`` and
adds schema, runtime, repeated local CPU load, and tensor-manifest evidence for
the bounded ``torch_bounded_qnn_module`` route. It intentionally keeps
cross-runtime, CUDA, external long-lived artefact, provider, hardware,
isolated-benchmark, and performance promotion blocked until those artefacts
exist.
"""

from __future__ import annotations

import io
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from numpy.typing import ArrayLike

from .torch_bridge import _load_torch
from .torch_checkpoint import (
    TORCH_CHECKPOINT_SCHEMA,
    run_torch_module_checkpoint_audit,
)

TORCH_CHECKPOINT_MATRIX_SCHEMA = (
    "scpn_quantum_control.phase.torch_bounded_qnn_checkpoint_matrix.v1"
)
TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY = (
    "bounded PyTorch long-lived checkpoint matrix for the local phase-QNN "
    "nn.Module route only; schema, tensor metadata, runtime fingerprint, and "
    "repeated weights_only CPU checkpoint loads are recorded with no "
    "cross-runtime, CUDA, provider, hardware, external long-lived artefact, "
    "isolated benchmark, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchCheckpointMatrixRoute:
    """One route in the bounded PyTorch checkpoint matrix."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready checkpoint-matrix route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchCheckpointMatrixTensorMetadata:
    """Tensor manifest entry loaded from a bounded PyTorch checkpoint."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    numel: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready tensor metadata."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "numel": self.numel,
        }


@dataclass(frozen=True)
class PhaseTorchCheckpointRuntimeFingerprint:
    """Runtime fingerprint for bounded PyTorch checkpoint replay."""

    python_version: str
    platform: str
    machine: str
    torch_version: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-ready runtime fingerprint metadata."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "torch_version": self.torch_version,
        }


@dataclass(frozen=True)
class PhaseTorchCheckpointMatrixResult:
    """Long-lived checkpoint-matrix evidence for bounded PyTorch QNN modules."""

    matrix_schema: str
    checkpoint_schema: str
    checkpoint_path: str | None
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    replay_count: int
    tensor_metadata: tuple[PhaseTorchCheckpointMatrixTensorMetadata, ...]
    runtime_fingerprint: PhaseTorchCheckpointRuntimeFingerprint
    base_checkpoint_passed: bool
    base_checkpoint_open_gaps: tuple[str, ...]
    routes: tuple[PhaseTorchCheckpointMatrixRoute, ...]
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_long_lived_checkpoint_matrix"
    claim_boundary: str = TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether local matrix evidence passed without failed routes."""
        return (
            self.base_checkpoint_passed
            and self.route_status("versioned_checkpoint_schema") == "passed"
            and self.route_status("weights_only_local_cpu_replay") == "passed"
            and self.route_status("repeated_local_cpu_replay") == "passed"
            and self.route_status("tensor_metadata_manifest") == "passed"
            and self.route_status("runtime_fingerprint_recorded") == "passed"
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return matrix routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named checkpoint-matrix route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch checkpoint matrix route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready checkpoint-matrix evidence."""
        return {
            "matrix_schema": self.matrix_schema,
            "checkpoint_schema": self.checkpoint_schema,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "replay_count": self.replay_count,
            "tensor_metadata": [metadata.to_dict() for metadata in self.tensor_metadata],
            "runtime_fingerprint": self.runtime_fingerprint.to_dict(),
            "base_checkpoint_passed": self.base_checkpoint_passed,
            "base_checkpoint_open_gaps": list(self.base_checkpoint_open_gaps),
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "passed": self.passed,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def run_torch_long_lived_checkpoint_matrix(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    checkpoint_path: str | Path | None = None,
    replay_count: int = 2,
    learning_rate: float = 0.05,
    tolerance: float = 1e-6,
) -> PhaseTorchCheckpointMatrixResult:
    """Audit bounded PyTorch checkpoint persistence as a replay matrix.

    Parameters
    ----------
    features:
        Real-valued feature matrix with shape ``(n_samples, n_parameters)``.
    labels:
        Binary label vector with shape ``(n_samples,)``.
    initial_params:
        Initial bounded phase-QNN parameter vector with width matching
        ``features``.
    checkpoint_path:
        Optional checkpoint destination. When omitted, a temporary local
        checkpoint is used for matrix inspection and the public result keeps
        ``checkpoint_path`` as ``None``.
    replay_count:
        Positive number of repeated weights-only CPU loads to perform against
        the same checkpoint bytes.
    learning_rate:
        Positive Adam learning rate used by the underlying checkpoint audit.
    tolerance:
        Non-negative absolute tolerance for the underlying replay checks.

    Returns
    -------
    PhaseTorchCheckpointMatrixResult
        Schema, runtime, tensor-manifest, repeated local replay, and blocked
        portability rows for the bounded checkpoint route.
    """
    if replay_count <= 0:
        raise ValueError("replay_count must be positive")

    if checkpoint_path is None:
        with TemporaryDirectory(prefix="scpn_torch_checkpoint_matrix_") as directory:
            internal_path = Path(directory) / "bounded_phase_qnn.pt"
            return _run_checkpoint_matrix_with_path(
                features=features,
                labels=labels,
                initial_params=initial_params,
                internal_checkpoint_path=internal_path,
                public_checkpoint_path=None,
                replay_count=replay_count,
                learning_rate=learning_rate,
                tolerance=tolerance,
            )

    resolved_path = Path(checkpoint_path)
    return _run_checkpoint_matrix_with_path(
        features=features,
        labels=labels,
        initial_params=initial_params,
        internal_checkpoint_path=resolved_path,
        public_checkpoint_path=str(resolved_path),
        replay_count=replay_count,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )


def _run_checkpoint_matrix_with_path(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    internal_checkpoint_path: Path,
    public_checkpoint_path: str | None,
    replay_count: int,
    learning_rate: float,
    tolerance: float,
) -> PhaseTorchCheckpointMatrixResult:
    """Build matrix evidence around a concrete checkpoint path."""
    torch_module = _load_torch()
    base_audit = run_torch_module_checkpoint_audit(
        features=features,
        labels=labels,
        initial_params=initial_params,
        checkpoint_path=internal_checkpoint_path,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )
    checkpoint_bytes = internal_checkpoint_path.read_bytes()
    loaded_payloads = tuple(
        _load_checkpoint_payload_from_bytes(
            torch_module=torch_module,
            checkpoint_bytes=checkpoint_bytes,
        )
        for _ in range(replay_count)
    )
    loaded_schema = _required_string(loaded_payloads[-1], "checkpoint_schema")
    tensor_metadata = _tensor_metadata_from_payload(loaded_payloads[-1])
    runtime_fingerprint = PhaseTorchCheckpointRuntimeFingerprint(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
    )
    module_tensor_names = {
        metadata.name.removeprefix("module_state_dict.")
        for metadata in tensor_metadata
        if metadata.name.startswith("module_state_dict.")
    }
    tensor_manifest_passed = bool(tensor_metadata) and set(
        base_audit.state_dict_keys,
    ).issubset(module_tensor_names)
    repeated_replay_passed = len(loaded_payloads) == replay_count and all(
        _required_string(payload, "checkpoint_schema") == TORCH_CHECKPOINT_SCHEMA
        for payload in loaded_payloads
    )
    runtime_recorded = all(runtime_fingerprint.to_dict().values())
    routes = (
        PhaseTorchCheckpointMatrixRoute(
            name="versioned_checkpoint_schema",
            status="passed"
            if loaded_schema == TORCH_CHECKPOINT_SCHEMA == base_audit.checkpoint_schema
            else "failed",
            reason="bounded checkpoint schema is present and versioned",
            requires=()
            if loaded_schema == TORCH_CHECKPOINT_SCHEMA == base_audit.checkpoint_schema
            else ("matching_bounded_checkpoint_schema",),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="weights_only_local_cpu_replay",
            status="passed"
            if base_audit.route_status("checkpoint_weights_only_cpu_load") == "passed"
            else "failed",
            reason="checkpoint bytes loaded with weights_only=True and map_location='cpu'",
            requires=()
            if base_audit.route_status("checkpoint_weights_only_cpu_load") == "passed"
            else ("weights_only_cpu_load",),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="repeated_local_cpu_replay",
            status="passed" if base_audit.passed and repeated_replay_passed else "failed",
            reason="same checkpoint bytes passed repeated local CPU schema loads",
            requires=()
            if base_audit.passed and repeated_replay_passed
            else ("repeated_local_checkpoint_loads",),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="tensor_metadata_manifest",
            status="passed" if tensor_manifest_passed else "failed",
            reason="module and optimizer tensor metadata were collected from the payload",
            requires=() if tensor_manifest_passed else ("checkpoint_tensor_manifest",),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="runtime_fingerprint_recorded",
            status="passed" if runtime_recorded else "failed",
            reason="Python, platform, machine, and PyTorch versions are recorded",
            requires=() if runtime_recorded else ("runtime_fingerprint",),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="cross_runtime_checkpoint_replay",
            status="blocked",
            reason="no cross-runtime checkpoint replay artefact is present",
            requires=("second_runtime_checkpoint_replay", "compatibility_matrix_runner"),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="cuda_checkpoint_replay",
            status="blocked",
            reason="CUDA checkpoint replay requires a real CUDA smoke artefact",
            requires=("cuda_checkpoint_smoke",),
        ),
        PhaseTorchCheckpointMatrixRoute(
            name="long_lived_external_checkpoint_artifact",
            status="blocked",
            reason="no externally retained long-lived checkpoint corpus is present",
            requires=("published_checkpoint_artifact", "retention_manifest"),
        ),
    )
    return PhaseTorchCheckpointMatrixResult(
        matrix_schema=TORCH_CHECKPOINT_MATRIX_SCHEMA,
        checkpoint_schema=loaded_schema,
        checkpoint_path=public_checkpoint_path,
        checkpoint_size_bytes=base_audit.checkpoint_size_bytes,
        checkpoint_sha256=base_audit.checkpoint_sha256,
        replay_count=replay_count,
        tensor_metadata=tensor_metadata,
        runtime_fingerprint=runtime_fingerprint,
        base_checkpoint_passed=base_audit.passed,
        base_checkpoint_open_gaps=base_audit.open_gaps,
        routes=routes,
    )


def _load_checkpoint_payload_from_bytes(
    *,
    torch_module: Any,
    checkpoint_bytes: bytes,
) -> Mapping[str, object]:
    """Load checkpoint bytes with the bounded weights-only CPU policy."""
    load = getattr(torch_module, "load", None)
    if not callable(load):
        raise RuntimeError("PyTorch runtime does not expose torch.load")
    payload = load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError("PyTorch checkpoint payload must be a mapping")
    return cast(Mapping[str, object], payload)


def _tensor_metadata_from_payload(
    payload: Mapping[str, object],
) -> tuple[PhaseTorchCheckpointMatrixTensorMetadata, ...]:
    """Collect tensor metadata from module and optimizer checkpoint entries."""
    metadata: list[PhaseTorchCheckpointMatrixTensorMetadata] = []
    _collect_tensor_metadata(
        "module_state_dict",
        _required_mapping(payload, "module_state_dict"),
        metadata,
    )
    _collect_tensor_metadata(
        "optimizer_state_dict",
        _required_mapping(payload, "optimizer_state_dict"),
        metadata,
    )
    return tuple(sorted(metadata, key=lambda entry: entry.name))


def _collect_tensor_metadata(
    prefix: str,
    value: object,
    metadata: list[PhaseTorchCheckpointMatrixTensorMetadata],
) -> None:
    """Recursively collect tensor metadata from checkpoint payload values."""
    if _is_tensor_like(value):
        metadata.append(_tensor_metadata(prefix, value))
        return
    if isinstance(value, Mapping):
        nested_mapping = cast(Mapping[object, object], value)
        for key in sorted(nested_mapping, key=lambda item: str(item)):
            _collect_tensor_metadata(f"{prefix}.{key}", nested_mapping[key], metadata)
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _collect_tensor_metadata(f"{prefix}[{index}]", item, metadata)


def _tensor_metadata(
    name: str,
    value: object,
) -> PhaseTorchCheckpointMatrixTensorMetadata:
    """Convert one tensor-like checkpoint value into metadata."""
    tensor = cast(Any, value)
    shape = tuple(int(axis) for axis in tensor.shape)
    numel = int(tensor.numel())
    return PhaseTorchCheckpointMatrixTensorMetadata(
        name=name,
        shape=shape,
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        requires_grad=bool(getattr(tensor, "requires_grad", False)),
        numel=numel,
    )


def _is_tensor_like(value: object) -> bool:
    """Return whether a checkpoint value behaves like a PyTorch tensor."""
    candidate = cast(Any, value)
    return (
        hasattr(candidate, "shape")
        and hasattr(candidate, "dtype")
        and hasattr(candidate, "device")
        and callable(getattr(candidate, "numel", None))
    )


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return a required mapping entry from a checkpoint payload."""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"PyTorch checkpoint entry {key!r} must be a mapping")
    return cast(Mapping[str, object], value)


def _required_string(payload: Mapping[str, object], key: str) -> str:
    """Return a required string entry from a checkpoint payload."""
    value = payload.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"PyTorch checkpoint entry {key!r} must be a string")
    return value


__all__ = [
    "PhaseTorchCheckpointMatrixResult",
    "PhaseTorchCheckpointMatrixRoute",
    "PhaseTorchCheckpointMatrixTensorMetadata",
    "PhaseTorchCheckpointRuntimeFingerprint",
    "TORCH_CHECKPOINT_MATRIX_CLAIM_BOUNDARY",
    "TORCH_CHECKPOINT_MATRIX_SCHEMA",
    "run_torch_long_lived_checkpoint_matrix",
]
