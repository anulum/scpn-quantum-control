# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch Module State Utilities
"""State-dictionary utilities for bounded PyTorch phase-QNN modules.

The public functions in this module deliberately target the bounded
``torch_bounded_qnn_module`` route. They verify strict module state replay and
optimizer-state continuity without promoting CUDA, provider, hardware,
checkpoint-portability, or performance claims.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    _torch_scalar_to_float,
    _torch_values_to_numpy,
    torch_bounded_qnn_module,
)

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_MODULE_STATE_CLAIM_BOUNDARY = (
    "bounded PyTorch module-state audit for the local phase-QNN nn.Module route; "
    "strict state_dict/load_state_dict and Adam optimizer-state replay are checked "
    "on local CPU-compatible tensors only, with no provider, hardware, CUDA, "
    "isolated benchmark, durable checkpoint-portability, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchModuleStateTensorMismatch:
    """Shape or dtype mismatch for one PyTorch module state entry."""

    key: str
    expected_shape: tuple[int, ...]
    observed_shape: tuple[int, ...]
    expected_dtype: str
    observed_dtype: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready tensor mismatch metadata."""
        return {
            "key": self.key,
            "expected_shape": list(self.expected_shape),
            "observed_shape": list(self.observed_shape),
            "expected_dtype": self.expected_dtype,
            "observed_dtype": self.observed_dtype,
        }


@dataclass(frozen=True)
class PhaseTorchModuleStateValidationResult:
    """Validation result for a bounded PyTorch module ``state_dict``."""

    state_dict_keys: tuple[str, ...]
    expected_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    mismatched_tensors: tuple[PhaseTorchModuleStateTensorMismatch, ...]
    claim_boundary: str = (
        "strict PyTorch module state_dict validation for bounded phase-QNN modules; "
        "no load, provider, hardware, CUDA, isolated benchmark, or performance claim"
    )

    @property
    def passed(self) -> bool:
        """Return whether keys, shapes, and dtypes match the target module."""
        return not self.missing_keys and not self.unexpected_keys and not self.mismatched_tensors

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready validation metadata."""
        return {
            "state_dict_keys": list(self.state_dict_keys),
            "expected_keys": list(self.expected_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "mismatched_tensors": [mismatch.to_dict() for mismatch in self.mismatched_tensors],
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseTorchModuleStateRoute:
    """One bounded PyTorch module-state audit route."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready module-state route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchModuleStateAuditResult:
    """Strict module and optimizer state replay evidence for bounded PyTorch QNNs."""

    state_dict_keys: tuple[str, ...]
    strict_load_missing_keys: tuple[str, ...]
    strict_load_unexpected_keys: tuple[str, ...]
    module_loss: float
    replay_loss: float
    module_loss_error: float
    module_gradient: FloatArray
    replay_gradient: FloatArray
    module_gradient_error: float
    optimizer_state_entry_count: int
    optimizer_param_group_count: int
    optimizer_replay_parameter_error: float
    optimizer_replay_loss_error: float
    tolerance: float
    torch_version: str
    routes: tuple[PhaseTorchModuleStateRoute, ...]
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_qnn_module_state_audit"
    claim_boundary: str = TORCH_MODULE_STATE_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether the bounded local module-state routes passed."""
        return (
            self.route_status("module_state_dict_round_trip") == "passed"
            and self.route_status("optimizer_state_dict_round_trip") == "passed"
            and self.module_loss_error <= self.tolerance
            and self.module_gradient_error <= self.tolerance
            and self.optimizer_replay_parameter_error <= self.tolerance
            and self.optimizer_replay_loss_error <= self.tolerance
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return module-state routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named module-state route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch module-state route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready module-state audit evidence."""
        return {
            "state_dict_keys": list(self.state_dict_keys),
            "strict_load_missing_keys": list(self.strict_load_missing_keys),
            "strict_load_unexpected_keys": list(self.strict_load_unexpected_keys),
            "module_loss": self.module_loss,
            "replay_loss": self.replay_loss,
            "module_loss_error": self.module_loss_error,
            "module_gradient": self.module_gradient.tolist(),
            "replay_gradient": self.replay_gradient.tolist(),
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


def validate_torch_bounded_qnn_state_dict(
    module: object,
    state_dict: Mapping[str, object],
) -> PhaseTorchModuleStateValidationResult:
    """Validate a bounded PyTorch phase-QNN module ``state_dict`` without loading it.

    Parameters
    ----------
    module:
        Module returned by ``torch_bounded_qnn_module`` or an API-compatible
        bounded phase-QNN module.
    state_dict:
        Candidate state dictionary to compare against the module's current
        parameter and persistent-buffer keys.

    Returns
    -------
    PhaseTorchModuleStateValidationResult
        Key, shape, dtype, and claim-boundary metadata for fail-closed callers.
    """
    expected_state = _module_state_dict(module)
    observed_keys = tuple(str(key) for key in state_dict)
    expected_keys = tuple(expected_state)
    missing_keys = tuple(key for key in expected_keys if key not in state_dict)
    unexpected_keys = tuple(key for key in observed_keys if key not in expected_state)
    mismatches: list[PhaseTorchModuleStateTensorMismatch] = []
    for key in expected_keys:
        if key not in state_dict:
            continue
        expected_shape, expected_dtype = _tensor_descriptor(expected_state[key])
        observed_shape, observed_dtype = _tensor_descriptor(state_dict[key])
        if expected_shape != observed_shape or expected_dtype != observed_dtype:
            mismatches.append(
                PhaseTorchModuleStateTensorMismatch(
                    key=key,
                    expected_shape=expected_shape,
                    observed_shape=observed_shape,
                    expected_dtype=expected_dtype,
                    observed_dtype=observed_dtype,
                )
            )
    return PhaseTorchModuleStateValidationResult(
        state_dict_keys=observed_keys,
        expected_keys=expected_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        mismatched_tensors=tuple(mismatches),
    )


def run_torch_module_state_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    learning_rate: float = 0.05,
    tolerance: float = 1e-6,
) -> PhaseTorchModuleStateAuditResult:
    """Audit strict bounded PyTorch module and optimizer state replay.

    The audit checks the bounded ``torch_bounded_qnn_module`` route only. It
    verifies exact key matching through ``load_state_dict(strict=True)``, checks
    replayed loss and gradient values against the source module, then creates
    Adam optimizer state through one real optimisation step and verifies that a
    fresh optimizer created over the same module parameters continues with
    identical next-step loss and parameters after ``load_state_dict``.
    """
    torch_module = _load_torch()
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "initial_params",
        _torch_values_to_numpy(initial_params),
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    learning_rate_value = _as_positive_learning_rate(learning_rate)
    module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
    )
    module_state = _clone_state_mapping(_module_state_dict(module))
    validation = validate_torch_bounded_qnn_state_dict(module, module_state)
    replay_module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=np.zeros_like(parameter_values),
    )
    load_result = _strict_load_module_state(replay_module, module_state)
    module_loss, module_gradient = _module_loss_and_gradient(module, parameter_values.size)
    replay_loss, replay_gradient = _module_loss_and_gradient(
        replay_module,
        parameter_values.size,
    )
    module_loss_error = abs(module_loss - replay_loss)
    gradient_delta = module_gradient - replay_gradient
    module_gradient_error = float(np.max(np.abs(gradient_delta))) if gradient_delta.size else 0.0
    (
        optimizer_state_entry_count,
        optimizer_param_group_count,
        optimizer_parameter_error,
        optimizer_loss_error,
    ) = _optimizer_replay_error(
        torch_module=torch_module,
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        learning_rate=learning_rate_value,
    )
    module_route_passed = bool(
        validation.passed
        and not load_result[0]
        and not load_result[1]
        and module_loss_error <= tolerance_value
        and module_gradient_error <= tolerance_value
    )
    optimizer_route_passed = bool(
        optimizer_state_entry_count > 0
        and optimizer_param_group_count == 1
        and optimizer_parameter_error <= tolerance_value
        and optimizer_loss_error <= tolerance_value
    )
    routes = (
        PhaseTorchModuleStateRoute(
            name="module_state_dict_round_trip",
            status="passed" if module_route_passed else "failed",
            reason=(
                "bounded phase-QNN module state_dict replay passed with "
                "load_state_dict(strict=True)"
            )
            if module_route_passed
            else "bounded phase-QNN module state_dict replay did not match",
            requires=()
            if module_route_passed
            else ("matching_module_state_dict_keys_shapes_and_dtypes",),
        ),
        PhaseTorchModuleStateRoute(
            name="optimizer_state_dict_round_trip",
            status="passed" if optimizer_route_passed else "failed",
            reason=(
                "Adam optimizer state_dict replay passed after loading into an "
                "optimizer created with the same module parameters"
            )
            if optimizer_route_passed
            else "Adam optimizer state_dict replay did not match",
            requires=()
            if optimizer_route_passed
            else ("same_module_parameters", "populated_adam_optimizer_state"),
        ),
        PhaseTorchModuleStateRoute(
            name="device_state_transfer",
            status="blocked",
            reason=(
                "CUDA/device state transfer is a separate promotion route and "
                "requires compatible accelerator artefacts"
            ),
            requires=(
                "compatible_cuda_device",
                "module_state_dict_device_transfer_artifact",
                "optimizer_state_dict_device_transfer_artifact",
            ),
        ),
    )
    return PhaseTorchModuleStateAuditResult(
        state_dict_keys=validation.state_dict_keys,
        strict_load_missing_keys=load_result[0],
        strict_load_unexpected_keys=load_result[1],
        module_loss=module_loss,
        replay_loss=replay_loss,
        module_loss_error=module_loss_error,
        module_gradient=module_gradient,
        replay_gradient=replay_gradient,
        module_gradient_error=module_gradient_error,
        optimizer_state_entry_count=optimizer_state_entry_count,
        optimizer_param_group_count=optimizer_param_group_count,
        optimizer_replay_parameter_error=optimizer_parameter_error,
        optimizer_replay_loss_error=optimizer_loss_error,
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        routes=routes,
    )


def _tensor_descriptor(value: object) -> tuple[tuple[int, ...], str]:
    """Return shape and dtype metadata for a tensor-like state entry."""
    shape = getattr(value, "shape", None)
    if shape is None:
        array = np.asarray(value)
        return tuple(int(axis) for axis in array.shape), str(array.dtype)
    dtype = getattr(value, "dtype", None)
    return tuple(int(axis) for axis in shape), str(dtype)


def _clone_state_value(value: object) -> object:
    """Clone tensor-like leaves inside a PyTorch state mapping."""
    if isinstance(value, Mapping):
        return {key: _clone_state_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(child) for child in value)
    detach = getattr(value, "detach", None)
    candidate = detach() if callable(detach) else value
    clone = getattr(candidate, "clone", None)
    if callable(clone):
        return clone()
    return candidate


def _clone_state_mapping(state: Mapping[str, object]) -> dict[str, object]:
    """Clone a string-keyed PyTorch state mapping."""
    return {key: _clone_state_value(value) for key, value in state.items()}


def _module_state_dict(module: object) -> dict[str, object]:
    """Return the string-keyed ``state_dict`` for a PyTorch module."""
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("module must expose a callable state_dict")
    raw_state = state_dict()
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("module state_dict must return a mapping")
    result: dict[str, object] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise RuntimeError("module state_dict keys must be strings")
        result[key] = value
    return result


def _strict_load_module_state(
    module: object,
    state_dict: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load a module state mapping with PyTorch strict key checks enabled."""
    load_state_dict = getattr(module, "load_state_dict", None)
    if not callable(load_state_dict):
        raise RuntimeError("module must expose a callable load_state_dict")
    load_result = load_state_dict(dict(state_dict), strict=True)
    return (
        _string_tuple_from_sequence(getattr(load_result, "missing_keys", ())),
        _string_tuple_from_sequence(getattr(load_result, "unexpected_keys", ())),
    )


def _string_tuple_from_sequence(values: object) -> tuple[str, ...]:
    """Convert a PyTorch result sequence into a tuple of strings."""
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return tuple(str(value) for value in values)
    return ()


def _module_loss_and_gradient(module: object, width: int) -> tuple[float, FloatArray]:
    """Return bounded module loss and gradient for the current parameters."""
    zero_grad = getattr(module, "zero_grad", None)
    if callable(zero_grad):
        zero_grad()
    module_call = cast(Callable[[], object], module)
    loss = module_call()
    backward = getattr(loss, "backward", None)
    if not callable(backward):
        raise RuntimeError("module loss must expose a callable backward method")
    backward()
    params = getattr(module, "params", None)
    grad = getattr(params, "grad", None)
    if grad is None:
        raise RuntimeError("module parameters did not receive a gradient")
    return _torch_scalar_to_float(loss), _as_parameter_vector(
        "bounded PyTorch module gradient",
        _torch_values_to_numpy(grad),
        width=width,
    )


def _adam_optimizer(torch_module: Any, module: object, learning_rate: float) -> Any:
    """Return an Adam optimizer bound to the module parameters."""
    optim = getattr(torch_module, "optim", None)
    adam_cls = getattr(optim, "Adam", None)
    if not callable(adam_cls):
        raise RuntimeError("PyTorch Adam optimizer is unavailable")
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise RuntimeError("module must expose callable parameters")
    parameter_tuple = tuple(parameters())
    if not parameter_tuple:
        raise RuntimeError("module must expose at least one trainable parameter")
    return adam_cls(parameter_tuple, lr=learning_rate)


def _optimizer_state_dict(optimizer: object) -> dict[str, object]:
    """Return a string-keyed PyTorch optimizer state dictionary."""
    state_dict = getattr(optimizer, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("optimizer must expose a callable state_dict")
    raw_state = state_dict()
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("optimizer state_dict must return a mapping")
    result: dict[str, object] = {}
    for key, value in raw_state.items():
        result[str(key)] = value
    return result


def _optimizer_state_counts(state_dict: Mapping[str, object]) -> tuple[int, int]:
    """Return optimizer state-entry and parameter-group counts."""
    state = state_dict.get("state", {})
    param_groups = state_dict.get("param_groups", ())
    if not isinstance(state, Mapping):
        raise RuntimeError("optimizer state_dict['state'] must be a mapping")
    if not isinstance(param_groups, Sequence) or isinstance(param_groups, (str, bytes)):
        raise RuntimeError("optimizer state_dict['param_groups'] must be a sequence")
    return len(state), len(param_groups)


def _load_optimizer_state(optimizer: object, state_dict: Mapping[str, object]) -> None:
    """Load an optimizer state dictionary into an existing optimizer."""
    load_state_dict = getattr(optimizer, "load_state_dict", None)
    if not callable(load_state_dict):
        raise RuntimeError("optimizer must expose a callable load_state_dict")
    load_state_dict(dict(state_dict))


def _optimizer_step(module: object, optimizer: object) -> tuple[float, FloatArray]:
    """Run one optimiser step and return pre-step loss plus post-step parameters."""
    zero_grad = getattr(optimizer, "zero_grad", None)
    if callable(zero_grad):
        zero_grad()
    module_call = cast(Callable[[], object], module)
    loss = module_call()
    backward = getattr(loss, "backward", None)
    if not callable(backward):
        raise RuntimeError("module loss must expose a callable backward method")
    backward()
    step = getattr(optimizer, "step", None)
    if not callable(step):
        raise RuntimeError("optimizer must expose a callable step")
    step()
    params = getattr(module, "params", None)
    parameter_values = _torch_values_to_numpy(params)
    return _torch_scalar_to_float(loss), parameter_values


def _optimizer_replay_error(
    *,
    torch_module: Any,
    features: FloatArray,
    labels: FloatArray,
    initial_params: FloatArray,
    learning_rate: float,
) -> tuple[int, int, float, float]:
    """Return optimizer state counts and deterministic replay errors."""
    module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
    )
    optimizer = _adam_optimizer(torch_module, module, learning_rate)
    _optimizer_step(module, optimizer)
    module_state = _clone_state_mapping(_module_state_dict(module))
    optimizer_state = _clone_state_mapping(_optimizer_state_dict(optimizer))
    state_count, group_count = _optimizer_state_counts(optimizer_state)
    replay_module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=np.zeros_like(initial_params),
    )
    _strict_load_module_state(replay_module, module_state)
    replay_optimizer = _adam_optimizer(torch_module, replay_module, learning_rate)
    _load_optimizer_state(replay_optimizer, optimizer_state)
    original_loss, original_params = _optimizer_step(module, optimizer)
    replay_loss, replay_params = _optimizer_step(replay_module, replay_optimizer)
    parameter_delta = original_params - replay_params
    parameter_error = float(np.max(np.abs(parameter_delta))) if parameter_delta.size else 0.0
    return state_count, group_count, parameter_error, abs(original_loss - replay_loss)


__all__ = [
    "PhaseTorchModuleStateAuditResult",
    "PhaseTorchModuleStateRoute",
    "PhaseTorchModuleStateTensorMismatch",
    "PhaseTorchModuleStateValidationResult",
    "TORCH_MODULE_STATE_CLAIM_BOUNDARY",
    "run_torch_module_state_audit",
    "validate_torch_bounded_qnn_state_dict",
]
