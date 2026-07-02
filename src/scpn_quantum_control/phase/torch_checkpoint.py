# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- PyTorch Checkpoint Utilities
"""Checkpoint replay utilities for bounded PyTorch phase-QNN modules.

The public audit in this module targets the bounded
``torch_bounded_qnn_module`` route only. It writes a real PyTorch checkpoint,
reloads it with CPU ``map_location`` and ``weights_only=True``, then verifies
strict module and Adam optimiser replay without promoting provider, hardware,
cross-runtime, isolated-benchmark, or performance claims.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .torch_bridge import (
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _as_positive_learning_rate,
    _load_torch,
    _torch_values_to_numpy,
    torch_bounded_qnn_module,
)
from .torch_module_state import (
    _adam_optimizer,
    _clone_state_mapping,
    _load_optimizer_state,
    _module_loss_and_gradient,
    _module_state_dict,
    _optimizer_state_counts,
    _optimizer_state_dict,
    _optimizer_step,
    _strict_load_module_state,
    validate_torch_bounded_qnn_state_dict,
)

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_CHECKPOINT_SCHEMA = "scpn_quantum_control.phase.torch_bounded_qnn_checkpoint.v1"
TORCH_CHECKPOINT_CLAIM_BOUNDARY = (
    "bounded PyTorch module checkpoint audit for the local phase-QNN nn.Module "
    "route; torch.save checkpoints are reloaded with map_location='cpu' and "
    "weights_only=True, then replayed through load_state_dict(strict=True) and "
    "Adam optimizer state loading, with no provider, hardware, CUDA, isolated "
    "benchmark, cross-runtime portability, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchCheckpointRoute:
    """One bounded PyTorch checkpoint audit route."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready checkpoint route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchCheckpointAuditResult:
    """Durable checkpoint replay evidence for bounded PyTorch QNN modules."""

    checkpoint_schema: str
    checkpoint_path: str | None
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    state_dict_keys: tuple[str, ...]
    strict_load_missing_keys: tuple[str, ...]
    strict_load_unexpected_keys: tuple[str, ...]
    module_loss_error: float
    module_gradient_error: float
    optimizer_state_entry_count: int
    optimizer_param_group_count: int
    optimizer_replay_parameter_error: float
    optimizer_replay_loss_error: float
    tolerance: float
    torch_version: str
    routes: tuple[PhaseTorchCheckpointRoute, ...]
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_qnn_module_checkpoint_audit"
    claim_boundary: str = TORCH_CHECKPOINT_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether bounded local checkpoint replay passed."""
        return (
            self.route_status("checkpoint_file_round_trip") == "passed"
            and self.route_status("checkpoint_weights_only_cpu_load") == "passed"
            and self.route_status("module_state_checkpoint_replay") == "passed"
            and self.route_status("optimizer_state_checkpoint_replay") == "passed"
            and self.module_loss_error <= self.tolerance
            and self.module_gradient_error <= self.tolerance
            and self.optimizer_replay_parameter_error <= self.tolerance
            and self.optimizer_replay_loss_error <= self.tolerance
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return checkpoint routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named PyTorch checkpoint route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch checkpoint route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready checkpoint audit evidence."""
        return {
            "checkpoint_schema": self.checkpoint_schema,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "state_dict_keys": list(self.state_dict_keys),
            "strict_load_missing_keys": list(self.strict_load_missing_keys),
            "strict_load_unexpected_keys": list(self.strict_load_unexpected_keys),
            "module_loss_error": self.module_loss_error,
            "module_gradient_error": self.module_gradient_error,
            "optimizer_state_entry_count": self.optimizer_state_entry_count,
            "optimizer_param_group_count": self.optimizer_param_group_count,
            "optimizer_replay_parameter_error": self.optimizer_replay_parameter_error,
            "optimizer_replay_loss_error": self.optimizer_replay_loss_error,
            "tolerance": self.tolerance,
            "torch_version": self.torch_version,
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "passed": self.passed,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def run_torch_module_checkpoint_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    checkpoint_path: str | Path | None = None,
    learning_rate: float = 0.05,
    tolerance: float = 1e-6,
) -> PhaseTorchCheckpointAuditResult:
    """Audit bounded PyTorch module checkpoint replay.

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
        Optional destination for a real ``torch.save`` checkpoint. When omitted,
        the audit writes and reloads an in-memory ``BytesIO`` checkpoint.
    learning_rate:
        Positive Adam learning rate used to populate and replay optimiser state.
    tolerance:
        Non-negative absolute tolerance for replayed loss, gradient, and
        optimiser continuation checks.

    Returns
    -------
    PhaseTorchCheckpointAuditResult
        Checkpoint size/hash metadata plus module and optimiser replay evidence.
    """
    torch_module = _load_torch()
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "initial_params",
        _torch_values_to_numpy(initial_params),
        width=feature_matrix.shape[1],
    )
    learning_rate_value = _as_positive_learning_rate(learning_rate)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    source_module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
    )
    source_optimizer = _adam_optimizer(torch_module, source_module, learning_rate_value)
    _optimizer_step(source_module, source_optimizer)
    module_state = _clone_state_mapping(_module_state_dict(source_module))
    optimizer_state = _clone_state_mapping(_optimizer_state_dict(source_optimizer))
    checkpoint_payload = _checkpoint_payload(
        module_state=module_state,
        optimizer_state=optimizer_state,
        feature_matrix=feature_matrix,
        label_vector=label_vector,
        parameter_values=parameter_values,
        learning_rate=learning_rate_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
    )
    checkpoint_bytes, resolved_path = _write_checkpoint_payload(
        torch_module=torch_module,
        payload=checkpoint_payload,
        checkpoint_path=checkpoint_path,
    )
    loaded_payload = _load_checkpoint_payload(
        torch_module=torch_module,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_path=resolved_path,
    )
    loaded_schema = _required_string(loaded_payload, "checkpoint_schema")
    if loaded_schema != TORCH_CHECKPOINT_SCHEMA:
        raise RuntimeError("loaded PyTorch checkpoint schema does not match bounded QNN schema")
    loaded_module_state = _required_mapping(loaded_payload, "module_state_dict")
    loaded_optimizer_state = _required_mapping(loaded_payload, "optimizer_state_dict")
    replay_module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=np.zeros_like(parameter_values),
    )
    validation = validate_torch_bounded_qnn_state_dict(replay_module, loaded_module_state)
    load_result = _strict_load_module_state(replay_module, loaded_module_state)
    replay_optimizer = _adam_optimizer(torch_module, replay_module, learning_rate_value)
    _load_optimizer_state(replay_optimizer, loaded_optimizer_state)
    state_count, group_count = _optimizer_state_counts(loaded_optimizer_state)
    module_loss, module_gradient = _module_loss_and_gradient(
        source_module,
        parameter_values.size,
    )
    replay_loss, replay_gradient = _module_loss_and_gradient(
        replay_module,
        parameter_values.size,
    )
    module_loss_error = abs(module_loss - replay_loss)
    gradient_delta = module_gradient - replay_gradient
    module_gradient_error = float(np.max(np.abs(gradient_delta))) if gradient_delta.size else 0.0
    original_loss, original_params = _optimizer_step(source_module, source_optimizer)
    replay_step_loss, replay_params = _optimizer_step(replay_module, replay_optimizer)
    parameter_delta = original_params - replay_params
    optimizer_parameter_error = (
        float(np.max(np.abs(parameter_delta))) if parameter_delta.size else 0.0
    )
    optimizer_loss_error = abs(original_loss - replay_step_loss)
    checkpoint_written = len(checkpoint_bytes) > 0 and (
        resolved_path is None or resolved_path.is_file()
    )
    weights_only_loaded = bool(loaded_schema == TORCH_CHECKPOINT_SCHEMA)
    module_replayed = bool(
        validation.passed
        and not load_result[0]
        and not load_result[1]
        and module_loss_error <= tolerance_value
        and module_gradient_error <= tolerance_value
    )
    optimizer_replayed = bool(
        state_count > 0
        and group_count == 1
        and optimizer_parameter_error <= tolerance_value
        and optimizer_loss_error <= tolerance_value
    )
    routes = (
        PhaseTorchCheckpointRoute(
            name="checkpoint_file_round_trip",
            status="passed" if checkpoint_written else "failed",
            reason="torch.save checkpoint bytes were written and reloaded",
            requires=() if checkpoint_written else ("torch_save_checkpoint_artifact",),
        ),
        PhaseTorchCheckpointRoute(
            name="checkpoint_weights_only_cpu_load",
            status="passed" if weights_only_loaded else "failed",
            reason="torch.load used map_location='cpu' with weights_only=True",
            requires=() if weights_only_loaded else ("weights_only_cpu_checkpoint_load",),
        ),
        PhaseTorchCheckpointRoute(
            name="module_state_checkpoint_replay",
            status="passed" if module_replayed else "failed",
            reason=("checkpoint module state replay passed with load_state_dict(strict=True)")
            if module_replayed
            else "checkpoint module state replay did not match",
            requires=() if module_replayed else ("matching_checkpoint_module_state_dict",),
        ),
        PhaseTorchCheckpointRoute(
            name="optimizer_state_checkpoint_replay",
            status="passed" if optimizer_replayed else "failed",
            reason="checkpoint Adam optimizer state replay passed",
            requires=() if optimizer_replayed else ("matching_checkpoint_optimizer_state_dict",),
        ),
        PhaseTorchCheckpointRoute(
            name="cross_runtime_checkpoint_portability",
            status="blocked",
            reason=(
                "cross-runtime, cross-platform, CUDA, and long-lived checkpoint "
                "portability require external artefacts beyond this local CPU replay"
            ),
            requires=(
                "cross_runtime_checkpoint_matrix",
                "compatible_cuda_checkpoint_replay",
                "long_lived_checkpoint_artifact",
            ),
        ),
    )
    return PhaseTorchCheckpointAuditResult(
        checkpoint_schema=loaded_schema,
        checkpoint_path=None if resolved_path is None else str(resolved_path),
        checkpoint_size_bytes=len(checkpoint_bytes),
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        state_dict_keys=validation.state_dict_keys,
        strict_load_missing_keys=load_result[0],
        strict_load_unexpected_keys=load_result[1],
        module_loss_error=module_loss_error,
        module_gradient_error=module_gradient_error,
        optimizer_state_entry_count=state_count,
        optimizer_param_group_count=group_count,
        optimizer_replay_parameter_error=optimizer_parameter_error,
        optimizer_replay_loss_error=optimizer_loss_error,
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        routes=routes,
    )


def _checkpoint_payload(
    *,
    module_state: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    feature_matrix: FloatArray,
    label_vector: FloatArray,
    parameter_values: FloatArray,
    learning_rate: float,
    torch_version: str,
) -> dict[str, object]:
    """Build a weights-only-compatible checkpoint payload."""
    return {
        "checkpoint_schema": TORCH_CHECKPOINT_SCHEMA,
        "module_state_dict": dict(module_state),
        "optimizer_state_dict": dict(optimizer_state),
        "metadata": {
            "feature_shape": [int(axis) for axis in feature_matrix.shape],
            "label_shape": [int(axis) for axis in label_vector.shape],
            "parameter_width": int(parameter_values.size),
            "learning_rate": float(learning_rate),
            "torch_version": torch_version,
        },
    }


def _write_checkpoint_payload(
    *,
    torch_module: Any,
    payload: Mapping[str, object],
    checkpoint_path: str | Path | None,
) -> tuple[bytes, Path | None]:
    """Write a PyTorch checkpoint payload to bytes or an explicit path."""
    save = getattr(torch_module, "save", None)
    if not callable(save):
        raise RuntimeError("PyTorch runtime does not expose torch.save")
    if checkpoint_path is None:
        buffer = io.BytesIO()
        save(dict(payload), buffer)
        return buffer.getvalue(), None
    resolved_path = Path(checkpoint_path)
    if not resolved_path.parent.exists():
        raise ValueError("checkpoint_path parent must exist")
    save(dict(payload), resolved_path)
    return resolved_path.read_bytes(), resolved_path


def _load_checkpoint_payload(
    *,
    torch_module: Any,
    checkpoint_bytes: bytes,
    checkpoint_path: Path | None,
) -> Mapping[str, object]:
    """Load a checkpoint payload with CPU map-location and weights-only mode."""
    load = getattr(torch_module, "load", None)
    if not callable(load):
        raise RuntimeError("PyTorch runtime does not expose torch.load")
    source: object = (
        checkpoint_path if checkpoint_path is not None else io.BytesIO(checkpoint_bytes)
    )
    payload = load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError("PyTorch checkpoint payload must be a mapping")
    return cast(Mapping[str, object], payload)


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
    "PhaseTorchCheckpointAuditResult",
    "PhaseTorchCheckpointRoute",
    "TORCH_CHECKPOINT_CLAIM_BOUNDARY",
    "TORCH_CHECKPOINT_SCHEMA",
    "run_torch_module_checkpoint_audit",
]
