# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- PyTorch Device-State Utilities
"""Device-state replay utilities for bounded PyTorch phase-QNN modules.

The public audit in this module targets the bounded
``torch_bounded_qnn_module`` route only. It verifies CPU module-device state
transfer through strict state replay and classifies CUDA transfer as passed or
blocked from real PyTorch runtime metadata, without promoting provider,
hardware, isolated-benchmark, or performance claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .torch_bridge import (
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _load_torch,
    _torch_cuda_metadata,
    _torch_values_to_numpy,
    torch_bounded_qnn_module,
)
from .torch_module_state import (
    _clone_state_mapping,
    _module_loss_and_gradient,
    _module_state_dict,
    _strict_load_module_state,
)

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_DEVICE_STATE_CLAIM_BOUNDARY = (
    "bounded PyTorch module device-state audit for the local phase-QNN nn.Module "
    "route; CPU state_dict replay is checked through module.to('cpu') and "
    "load_state_dict(strict=True), CUDA state replay is attempted only when a "
    "real CUDA smoke succeeds, with no provider, hardware, isolated benchmark, "
    "durable checkpoint-portability, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchDeviceStateRoute:
    """One bounded PyTorch device-state audit route."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready device-state route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class _DeviceReplayEvidence:
    """State replay deltas and tensor-device evidence for one torch device."""

    state_dict_keys: tuple[str, ...]
    state_devices: dict[str, str]
    loss_error: float
    gradient_error: float


@dataclass(frozen=True)
class PhaseTorchDeviceStateAuditResult:
    """Device-transfer state replay evidence for bounded PyTorch QNN modules."""

    state_dict_keys: tuple[str, ...]
    cpu_state_devices: dict[str, str]
    cpu_loss_error: float
    cpu_gradient_error: float
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    cuda_smoke_passed: bool
    cuda_skip_reason: str
    cuda_state_devices: dict[str, str]
    cuda_loss_error: float | None
    cuda_gradient_error: float | None
    tolerance: float
    torch_version: str
    routes: tuple[PhaseTorchDeviceStateRoute, ...]
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_qnn_module_device_state_audit"
    claim_boundary: str = TORCH_DEVICE_STATE_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether bounded local CPU device-state replay passed."""
        return (
            self.route_status("cpu_module_state_transfer") == "passed"
            and self.cpu_loss_error <= self.tolerance
            and self.cpu_gradient_error <= self.tolerance
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return device-state routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named PyTorch device-state route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch device-state route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready device-state audit evidence."""
        return {
            "state_dict_keys": list(self.state_dict_keys),
            "cpu_state_devices": dict(self.cpu_state_devices),
            "cpu_loss_error": self.cpu_loss_error,
            "cpu_gradient_error": self.cpu_gradient_error,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_names": list(self.cuda_device_names),
            "cuda_smoke_passed": self.cuda_smoke_passed,
            "cuda_skip_reason": self.cuda_skip_reason,
            "cuda_state_devices": dict(self.cuda_state_devices),
            "cuda_loss_error": self.cuda_loss_error,
            "cuda_gradient_error": self.cuda_gradient_error,
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


def run_torch_module_device_state_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    target_devices: Sequence[str] = ("cpu", "cuda"),
    tolerance: float = 1e-6,
) -> PhaseTorchDeviceStateAuditResult:
    """Audit bounded PyTorch module state replay across requested devices.

    Parameters
    ----------
    features:
        Real-valued feature matrix with shape ``(n_samples, n_parameters)``.
    labels:
        Binary label vector with shape ``(n_samples,)``.
    initial_params:
        Initial bounded phase-QNN parameter vector with width matching
        ``features``.
    target_devices:
        Device routes to classify. Supported values are ``"cpu"`` and
        ``"cuda"``. The CPU route is always included as the replay baseline.
    tolerance:
        Non-negative absolute tolerance for replayed loss and gradient checks.

    Returns
    -------
    PhaseTorchDeviceStateAuditResult
        CPU replay evidence plus CUDA pass/blocked metadata for fail-closed
        callers.
    """
    selected_devices = _normalise_target_devices(target_devices)
    torch_module = _load_torch()
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "initial_params",
        _torch_values_to_numpy(initial_params),
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    cuda_available, cuda_count, cuda_names, cuda_smoke_passed, cuda_reason = _torch_cuda_metadata(
        torch_module
    )

    cpu_evidence = _run_device_replay(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        device=_torch_device(torch_module, "cpu"),
    )
    cpu_passed = (
        cpu_evidence.loss_error <= tolerance_value
        and cpu_evidence.gradient_error <= tolerance_value
        and set(cpu_evidence.state_devices.values()) == {"cpu"}
    )
    cpu_reason = (
        "bounded module.to('cpu') state_dict replay passed with load_state_dict(strict=True)"
        if cpu_passed
        else (
            "bounded CPU module-state replay exceeded tolerance or produced non-CPU state tensors"
        )
    )
    routes = [
        PhaseTorchDeviceStateRoute(
            name="cpu_module_state_transfer",
            status="passed" if cpu_passed else "failed",
            reason=cpu_reason,
            requires=() if cpu_passed else ("cpu_tensor_state_replay",),
        )
    ]

    cuda_evidence: _DeviceReplayEvidence | None = None
    if "cuda" in selected_devices and cuda_smoke_passed:
        cuda_evidence = _run_device_replay(
            features=feature_matrix,
            labels=label_vector,
            initial_params=parameter_values,
            device=_torch_device(torch_module, "cuda"),
        )
        cuda_passed = (
            cuda_evidence.loss_error <= tolerance_value
            and cuda_evidence.gradient_error <= tolerance_value
            and set(cuda_evidence.state_devices.values()) == {"cuda"}
        )
        routes.append(
            PhaseTorchDeviceStateRoute(
                name="cuda_module_state_transfer",
                status="passed" if cuda_passed else "failed",
                reason=(
                    "bounded module.to('cuda') state_dict replay passed with "
                    "load_state_dict(strict=True)"
                )
                if cuda_passed
                else "bounded CUDA module-state replay did not match",
                requires=() if cuda_passed else ("matching_cuda_state_dict_replay",),
            )
        )
    elif "cuda" in selected_devices:
        routes.append(
            PhaseTorchDeviceStateRoute(
                name="cuda_module_state_transfer",
                status="blocked",
                reason=cuda_reason,
                requires=(
                    "compatible_cuda_device",
                    "successful_cuda_tensor_smoke_artifact",
                    "cuda_module_state_transfer_artifact",
                ),
            )
        )

    return PhaseTorchDeviceStateAuditResult(
        state_dict_keys=cpu_evidence.state_dict_keys,
        cpu_state_devices=cpu_evidence.state_devices,
        cpu_loss_error=cpu_evidence.loss_error,
        cpu_gradient_error=cpu_evidence.gradient_error,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
        cuda_smoke_passed=cuda_smoke_passed,
        cuda_skip_reason="" if cuda_smoke_passed else cuda_reason,
        cuda_state_devices={} if cuda_evidence is None else cuda_evidence.state_devices,
        cuda_loss_error=None if cuda_evidence is None else cuda_evidence.loss_error,
        cuda_gradient_error=None if cuda_evidence is None else cuda_evidence.gradient_error,
        tolerance=tolerance_value,
        torch_version=torch_version,
        routes=tuple(routes),
    )


def _normalise_target_devices(target_devices: Sequence[str]) -> tuple[str, ...]:
    """Return a de-duplicated target-device tuple with CPU baseline first."""
    if not target_devices:
        raise ValueError("target_devices must include at least one route")
    result: list[str] = ["cpu"]
    for target in target_devices:
        if target not in {"cpu", "cuda"}:
            raise ValueError("target_devices entries must be 'cpu' or 'cuda'")
        if target not in result:
            result.append(target)
    return tuple(result)


def _torch_device(torch_module: Any, target: str) -> object:
    """Build a torch device object for a normalized target name."""
    device_factory = getattr(torch_module, "device", None)
    if not callable(device_factory):
        raise RuntimeError("PyTorch runtime does not expose torch.device")
    if target == "cuda":
        return device_factory("cuda", 0)
    return device_factory(target)


def _move_module_to_device(module: object, device: object) -> object:
    """Move a bounded PyTorch module to a target device and return it."""
    to_device = getattr(module, "to", None)
    if not callable(to_device):
        raise RuntimeError("bounded PyTorch module must expose a callable to() method")
    moved = to_device(device)
    return module if moved is None else moved


def _run_device_replay(
    *,
    features: FloatArray,
    labels: FloatArray,
    initial_params: FloatArray,
    device: object,
) -> _DeviceReplayEvidence:
    """Replay a strict state_dict transfer for one concrete torch device."""
    source = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
    )
    source = _move_module_to_device(source, device)
    source_state = _clone_state_mapping(_module_state_dict(source))
    replay = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=np.zeros_like(initial_params),
    )
    replay = _move_module_to_device(replay, device)
    _strict_load_module_state(replay, source_state)
    source_loss, source_gradient = _module_loss_and_gradient(source, initial_params.size)
    replay_loss, replay_gradient = _module_loss_and_gradient(replay, initial_params.size)
    gradient_delta = source_gradient - replay_gradient
    gradient_error = float(np.max(np.abs(gradient_delta))) if gradient_delta.size else 0.0
    return _DeviceReplayEvidence(
        state_dict_keys=tuple(source_state),
        state_devices=_state_tensor_devices(source_state),
        loss_error=abs(source_loss - replay_loss),
        gradient_error=gradient_error,
    )


def _state_tensor_devices(state_dict: dict[str, object]) -> dict[str, str]:
    """Return torch device-type strings for tensors in a state dictionary."""
    devices: dict[str, str] = {}
    for key, value in state_dict.items():
        device = getattr(value, "device", None)
        device_type = getattr(device, "type", None)
        devices[key] = str(device_type if device_type is not None else device)
    return devices


__all__ = [
    "PhaseTorchDeviceStateAuditResult",
    "PhaseTorchDeviceStateRoute",
    "TORCH_DEVICE_STATE_CLAIM_BOUNDARY",
    "run_torch_module_device_state_audit",
]
