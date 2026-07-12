# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — torch aot autograd export module
# scpn-quantum-control -- PyTorch AOTAutograd Export Utilities
"""AOTAutograd FX graph persistence for bounded PyTorch phase-QNN modules.

This module promotes one narrow PyTorch training-compiler route: a bounded
phase-QNN loss function is compiled with ``torch._functorch.aot_autograd``, its
forward and backward FX ``GraphModule`` objects are captured, persisted with
``torch.save(...)``, reloaded with ``torch.load(..., weights_only=False)``, and
used to replay the local CPU loss and parameter gradient. The artifact is a
self-produced PyTorch FX pickle for local audit replay only. It is not a stable
cross-runtime export format and does not promote CUDA, provider, hardware,
dynamic-shape, isolated benchmark, or performance claims.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)
from .torch_bridge import (
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _load_torch,
    _torch_autograd_grad,
    _torch_bounded_qnn_loss_tensor,
    _torch_scalar_to_float,
    _torch_tensor,
    _torch_values_to_numpy,
)
from .torch_export import _graph_node_count

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA = (
    "scpn_quantum_control.phase.torch_bounded_qnn_aot_autograd_export.v1"
)
TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY = (
    "bounded PyTorch AOTAutograd FX graph persistence audit for the local "
    "phase-QNN loss route only; forward and backward FX GraphModule artifacts "
    "are captured with torch._functorch.aot_autograd, saved, loaded, and "
    "replayed locally against SCPN parameter-shift references with no stable "
    "cross-runtime export, CUDA, provider, hardware, dynamic-shape, isolated "
    "benchmark, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchAOTAutogradGraphRecord:
    """Persistent FX graph artifact metadata for one AOTAutograd graph."""

    kind: str
    artifact_path: str
    artifact_size_bytes: int
    artifact_sha256: str
    graph_node_count: int
    graph_module_type: str
    graph_code_sha256: str
    operation_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready AOTAutograd graph artifact metadata."""
        return {
            "kind": self.kind,
            "artifact_path": self.artifact_path,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_sha256": self.artifact_sha256,
            "graph_node_count": self.graph_node_count,
            "graph_module_type": self.graph_module_type,
            "graph_code_sha256": self.graph_code_sha256,
            "operation_names": list(self.operation_names),
        }


@dataclass(frozen=True)
class PhaseTorchAOTAutogradExportRoute:
    """One route in the bounded PyTorch AOTAutograd export audit."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready AOTAutograd route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchAOTAutogradExportResult:
    """Local AOTAutograd FX persistence evidence for bounded PyTorch QNNs."""

    matrix_schema: str
    artifact_dir: str
    forward_graph: PhaseTorchAOTAutogradGraphRecord
    backward_graph: PhaseTorchAOTAutogradGraphRecord
    reference_loss: float
    compiled_loss: float
    loaded_loss: float
    compiled_loss_error: float
    loaded_loss_error: float
    loss_error: float
    reference_gradient: FloatArray
    compiled_gradient: FloatArray
    loaded_gradient: FloatArray
    compiled_gradient_error: float
    loaded_gradient_error: float
    gradient_shape: tuple[int, ...]
    tolerance: float
    torch_version: str
    routes: tuple[PhaseTorchAOTAutogradExportRoute, ...]
    aot_autograd_fx_persistence_claim: bool = True
    cross_runtime_claim: bool = False
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_qnn_aot_autograd_export_audit"
    claim_boundary: str = TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether the local AOTAutograd FX persistence audit passed."""
        return (
            self.route_status("aot_autograd_forward_backward_capture") == "passed"
            and self.route_status("aot_autograd_graph_file_round_trip") == "passed"
            and self.route_status("loaded_backward_gradient_replay") == "passed"
            and self.loss_error <= self.tolerance
            and self.compiled_gradient_error <= self.tolerance
            and self.loaded_gradient_error <= self.tolerance
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return AOTAutograd routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named PyTorch AOTAutograd export route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch AOTAutograd export route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready AOTAutograd export audit evidence."""
        return {
            "matrix_schema": self.matrix_schema,
            "artifact_dir": self.artifact_dir,
            "forward_graph": self.forward_graph.to_dict(),
            "backward_graph": self.backward_graph.to_dict(),
            "reference_loss": self.reference_loss,
            "compiled_loss": self.compiled_loss,
            "loaded_loss": self.loaded_loss,
            "compiled_loss_error": self.compiled_loss_error,
            "loaded_loss_error": self.loaded_loss_error,
            "loss_error": self.loss_error,
            "reference_gradient": self.reference_gradient.tolist(),
            "compiled_gradient": self.compiled_gradient.tolist(),
            "loaded_gradient": self.loaded_gradient.tolist(),
            "compiled_gradient_error": self.compiled_gradient_error,
            "loaded_gradient_error": self.loaded_gradient_error,
            "gradient_shape": list(self.gradient_shape),
            "tolerance": self.tolerance,
            "torch_version": self.torch_version,
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "passed": self.passed,
            "aot_autograd_fx_persistence_claim": self.aot_autograd_fx_persistence_claim,
            "cross_runtime_claim": self.cross_runtime_claim,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def run_torch_aot_autograd_export_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    artifact_dir: str | Path,
    tolerance: float = 1e-6,
) -> PhaseTorchAOTAutogradExportResult:
    """Audit local AOTAutograd forward/backward FX graph persistence.

    Parameters
    ----------
    features:
        Real-valued feature matrix with shape ``(n_samples, n_parameters)``.
    labels:
        Binary label vector with shape ``(n_samples,)``.
    initial_params:
        Initial bounded phase-QNN parameter vector with width matching
        ``features``.
    artifact_dir:
        Directory where self-produced AOTAutograd forward/backward FX graph
        artifacts are retained. Its parent must already exist.
    tolerance:
        Non-negative absolute tolerance for compiled and loaded value/gradient
        replay.

    Returns
    -------
    PhaseTorchAOTAutogradExportResult
        Persistent local FX graph metadata plus compiled and loaded gradient
        replay evidence.
    """
    resolved_dir = Path(artifact_dir)
    if not resolved_dir.parent.exists():
        raise ValueError("artifact_dir parent must exist")
    if resolved_dir.exists() and not resolved_dir.is_dir():
        raise ValueError("artifact_dir must be a directory")
    resolved_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "initial_params",
        _torch_values_to_numpy(initial_params),
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    reference_loss = float(
        parameter_shift_qnn_classifier_loss(
            feature_matrix,
            label_vector,
            parameter_values,
        ),
    )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )

    torch_module = _load_torch()
    aot_function, make_boxed_func = _require_aot_autograd()
    parameter_tensor = _trainable_tensor(torch_module, parameter_values)
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)
    captured_graphs: dict[str, object] = {}

    def _forward_compiler(graph_module: object, _example_inputs: object) -> object:
        captured_graphs["forward"] = graph_module
        return make_boxed_func(_graph_forward(graph_module))

    def _backward_compiler(graph_module: object, _example_inputs: object) -> object:
        captured_graphs["backward"] = graph_module
        return make_boxed_func(_graph_forward(graph_module))

    def _loss_fn(params: object, batch_features: object, batch_labels: object) -> object:
        return _torch_bounded_qnn_loss_tensor(
            torch_module,
            batch_features,
            batch_labels,
            params,
        )

    compiled_loss_fn = aot_function(
        _loss_fn,
        fw_compiler=_forward_compiler,
        bw_compiler=_backward_compiler,
    )
    compiled_call = cast(Callable[[object, object, object], object], compiled_loss_fn)
    compiled_loss_tensor = compiled_call(parameter_tensor, feature_tensor, label_tensor)
    autograd_grad = _torch_autograd_grad(torch_module)
    compiled_gradient_tensor = autograd_grad(compiled_loss_tensor, (parameter_tensor,))[0]
    compiled_loss = _torch_scalar_to_float(compiled_loss_tensor)
    compiled_gradient = _torch_values_to_numpy(compiled_gradient_tensor)

    forward_graph = _require_captured_graph(captured_graphs, "forward")
    backward_graph = _require_captured_graph(captured_graphs, "backward")
    forward_path = resolved_dir / "aot_forward_graph.pt"
    backward_path = resolved_dir / "aot_backward_graph.pt"
    _torch_save(torch_module, forward_graph, forward_path)
    _torch_save(torch_module, backward_graph, backward_path)
    loaded_forward_graph = _torch_load_graph(torch_module, forward_path)
    loaded_backward_graph = _torch_load_graph(torch_module, backward_path)
    loaded_loss, loaded_gradient = _replay_loaded_aot_graphs(
        torch_module=torch_module,
        forward_graph=loaded_forward_graph,
        backward_graph=loaded_backward_graph,
        parameter_values=parameter_values,
        feature_matrix=feature_matrix,
        label_vector=label_vector,
    )
    compiled_loss_error = abs(reference_loss - compiled_loss)
    loaded_loss_error = abs(reference_loss - loaded_loss)
    compiled_gradient_error = _max_abs_error(reference_gradient, compiled_gradient)
    loaded_gradient_error = _max_abs_error(reference_gradient, loaded_gradient)
    forward_record = _graph_record(
        kind="forward",
        graph_module=loaded_forward_graph,
        artifact_path=forward_path,
    )
    backward_record = _graph_record(
        kind="backward",
        graph_module=loaded_backward_graph,
        artifact_path=backward_path,
    )
    routes = _aot_autograd_routes(
        forward_record=forward_record,
        backward_record=backward_record,
        compiled_loss_error=compiled_loss_error,
        loaded_loss_error=loaded_loss_error,
        compiled_gradient_error=compiled_gradient_error,
        loaded_gradient_error=loaded_gradient_error,
        tolerance=tolerance_value,
    )
    return PhaseTorchAOTAutogradExportResult(
        matrix_schema=TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA,
        artifact_dir=str(resolved_dir),
        forward_graph=forward_record,
        backward_graph=backward_record,
        reference_loss=reference_loss,
        compiled_loss=compiled_loss,
        loaded_loss=loaded_loss,
        compiled_loss_error=compiled_loss_error,
        loaded_loss_error=loaded_loss_error,
        loss_error=max(compiled_loss_error, loaded_loss_error),
        reference_gradient=reference_gradient.copy(),
        compiled_gradient=compiled_gradient.copy(),
        loaded_gradient=loaded_gradient.copy(),
        compiled_gradient_error=compiled_gradient_error,
        loaded_gradient_error=loaded_gradient_error,
        gradient_shape=tuple(int(value) for value in reference_gradient.shape),
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        routes=routes,
    )


def _require_aot_autograd() -> tuple[Callable[..., object], Callable[[object], object]]:
    """Return the installed AOTAutograd function and boxed wrapper."""
    try:
        from torch._functorch.aot_autograd import aot_function, make_boxed_func
    except ImportError as exc:
        raise RuntimeError("PyTorch AOTAutograd is unavailable") from exc
    return aot_function, make_boxed_func


def _trainable_tensor(torch_module: Any, values: FloatArray) -> object:
    """Return a trainable PyTorch tensor for parameter-gradient capture."""
    tensor = _torch_tensor(torch_module, values)
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    clone = getattr(tensor, "clone", None)
    if callable(clone):
        tensor = clone()
    requires_grad = getattr(tensor, "requires_grad_", None)
    if not callable(requires_grad):
        raise RuntimeError("PyTorch tensor does not expose requires_grad_")
    return requires_grad(True)


def _graph_forward(graph_module: object) -> object:
    """Return a graph module forward function or fail closed."""
    forward = getattr(graph_module, "forward", None)
    if not callable(forward):
        raise RuntimeError("AOTAutograd graph module must expose forward")
    return forward


def _require_captured_graph(graphs: Mapping[str, object], kind: str) -> object:
    """Return a captured AOTAutograd graph by kind."""
    graph = graphs.get(kind)
    if graph is None:
        raise RuntimeError(f"AOTAutograd did not capture a {kind} graph")
    return graph


def _torch_save(torch_module: Any, graph_module: object, path: Path) -> None:
    """Persist a self-produced PyTorch graph artifact."""
    save = getattr(torch_module, "save", None)
    if not callable(save):
        raise RuntimeError("PyTorch module does not expose torch.save")
    save(graph_module, path)


def _torch_load_graph(torch_module: Any, path: Path) -> object:
    """Load a self-produced PyTorch graph artifact."""
    load = getattr(torch_module, "load", None)
    if not callable(load):
        raise RuntimeError("PyTorch module does not expose torch.load")
    return load(path, weights_only=False)


def _replay_loaded_aot_graphs(
    *,
    torch_module: Any,
    forward_graph: object,
    backward_graph: object,
    parameter_values: FloatArray,
    feature_matrix: FloatArray,
    label_vector: FloatArray,
) -> tuple[float, FloatArray]:
    """Replay loaded AOTAutograd forward/backward graphs locally."""
    forward_call = cast(Callable[..., object], forward_graph)
    backward_call = cast(Callable[..., object], backward_graph)
    parameter_tensor = _torch_tensor(torch_module, parameter_values)
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)
    forward_output = _as_tuple(
        forward_call(parameter_tensor, feature_tensor, label_tensor),
        name="loaded AOTAutograd forward output",
    )
    if not forward_output:
        raise RuntimeError("loaded AOTAutograd forward graph returned no outputs")
    loaded_loss_tensor = forward_output[0]
    saved_tensors = forward_output[1:]
    ones_like = getattr(torch_module, "ones_like", None)
    if not callable(ones_like):
        raise RuntimeError("PyTorch module does not expose torch.ones_like")
    backward_output = _as_tuple(
        backward_call(*saved_tensors, ones_like(loaded_loss_tensor)),
        name="loaded AOTAutograd backward output",
    )
    if not backward_output:
        raise RuntimeError("loaded AOTAutograd backward graph returned no outputs")
    return (
        _torch_scalar_to_float(loaded_loss_tensor),
        _torch_values_to_numpy(backward_output[0]),
    )


def _as_tuple(values: object, *, name: str) -> tuple[object, ...]:
    """Return tuple outputs from loaded FX graphs."""
    if not isinstance(values, tuple):
        raise RuntimeError(f"{name} must be a tuple")
    return values


def _graph_record(
    *,
    kind: str,
    graph_module: object,
    artifact_path: Path,
) -> PhaseTorchAOTAutogradGraphRecord:
    """Build persistent metadata for a saved AOTAutograd graph."""
    artifact_bytes = artifact_path.read_bytes()
    graph_code = str(getattr(graph_module, "code", ""))
    return PhaseTorchAOTAutogradGraphRecord(
        kind=kind,
        artifact_path=str(artifact_path),
        artifact_size_bytes=len(artifact_bytes),
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        graph_node_count=_graph_node_count(graph_module),
        graph_module_type=type(graph_module).__name__,
        graph_code_sha256=hashlib.sha256(graph_code.encode("utf-8")).hexdigest(),
        operation_names=_graph_operation_names(graph_module),
    )


def _graph_operation_names(graph_module: object) -> tuple[str, ...]:
    """Return call-function operation names from an FX graph module."""
    graph = getattr(graph_module, "graph", None)
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        return ()
    operation_names: list[str] = []
    for node in nodes:
        if str(getattr(node, "op", "")) == "call_function":
            operation_names.append(str(getattr(node, "target", "")))
    return tuple(operation_names)


def _max_abs_error(expected: FloatArray, observed: FloatArray) -> float:
    """Return the maximum absolute difference between two vectors."""
    if expected.shape != observed.shape:
        return float("inf")
    if expected.size == 0:
        return 0.0
    return float(np.max(np.abs(expected - observed)))


def _aot_autograd_routes(
    *,
    forward_record: PhaseTorchAOTAutogradGraphRecord,
    backward_record: PhaseTorchAOTAutogradGraphRecord,
    compiled_loss_error: float,
    loaded_loss_error: float,
    compiled_gradient_error: float,
    loaded_gradient_error: float,
    tolerance: float,
) -> tuple[PhaseTorchAOTAutogradExportRoute, ...]:
    """Return fail-closed route statuses for AOTAutograd FX persistence."""
    capture_passed = bool(
        forward_record.graph_node_count > 0
        and backward_record.graph_node_count > 0
        and forward_record.operation_names
        and backward_record.operation_names
    )
    round_trip_passed = bool(
        forward_record.artifact_size_bytes > 0
        and backward_record.artifact_size_bytes > 0
        and len(forward_record.artifact_sha256) == 64
        and len(backward_record.artifact_sha256) == 64
    )
    gradient_replay_passed = bool(
        compiled_loss_error <= tolerance
        and loaded_loss_error <= tolerance
        and compiled_gradient_error <= tolerance
        and loaded_gradient_error <= tolerance
    )
    return (
        PhaseTorchAOTAutogradExportRoute(
            name="aot_autograd_forward_backward_capture",
            status="passed" if capture_passed else "failed",
            reason="AOTAutograd captured forward and backward FX GraphModule objects",
            requires=() if capture_passed else ("forward_backward_fx_graph_capture",),
        ),
        PhaseTorchAOTAutogradExportRoute(
            name="aot_autograd_graph_file_round_trip",
            status="passed" if round_trip_passed else "failed",
            reason="self-produced forward and backward FX graphs were saved and loaded locally",
            requires=() if round_trip_passed else ("saved_forward_backward_fx_graphs",),
        ),
        PhaseTorchAOTAutogradExportRoute(
            name="loaded_backward_gradient_replay",
            status="passed" if gradient_replay_passed else "failed",
            reason="loaded backward FX graph reproduced the SCPN parameter-shift gradient",
            requires=() if gradient_replay_passed else ("matching_loaded_backward_gradient",),
        ),
        PhaseTorchAOTAutogradExportRoute(
            name="cross_runtime_aot_autograd_execution",
            status="blocked",
            reason="PyTorch FX pickle artifacts are not a stable cross-runtime export contract",
            requires=("stable_cross_runtime_gradient_graph_format",),
        ),
        PhaseTorchAOTAutogradExportRoute(
            name="cuda_aot_autograd_execution",
            status="blocked",
            reason="CUDA AOTAutograd replay requires compatible visible hardware and wheel support",
            requires=("compatible_cuda_device", "cuda_aot_autograd_replay_artifact"),
        ),
        PhaseTorchAOTAutogradExportRoute(
            name="dynamic_shape_aot_autograd_export",
            status="blocked",
            reason="dynamic-shape AOTAutograd gradient export needs its own symbolic-shape replay",
            requires=("symbolic_shape_aot_autograd_replay_artifact",),
        ),
        PhaseTorchAOTAutogradExportRoute(
            name="isolated_benchmark_aot_autograd",
            status="blocked",
            reason="performance promotion needs isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )


__all__ = [
    "PhaseTorchAOTAutogradExportResult",
    "PhaseTorchAOTAutogradExportRoute",
    "PhaseTorchAOTAutogradGraphRecord",
    "TORCH_AOT_AUTOGRAD_EXPORT_CLAIM_BOUNDARY",
    "TORCH_AOT_AUTOGRAD_EXPORT_SCHEMA",
    "run_torch_aot_autograd_export_audit",
]
