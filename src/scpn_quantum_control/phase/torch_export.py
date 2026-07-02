# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- PyTorch Export Utilities
"""Export-persistence utilities for bounded PyTorch phase-QNN modules.

The public audit in this module targets the bounded
``torch_bounded_qnn_module`` route only. It exports the no-input bounded module
with ``torch.export.export(...)``, persists it with ``torch.export.save(...)``,
loads it with ``torch.export.load(...)``, and verifies local CPU value replay.
It does not promote AOTAutograd gradient export, dynamic shapes, provider,
hardware, CUDA, isolated benchmark, or performance claims.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
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
    _load_torch,
    _torch_scalar_to_float,
    _torch_values_to_numpy,
    torch_bounded_qnn_module,
)

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_EXPORT_CLAIM_BOUNDARY = (
    "bounded PyTorch module export-persistence audit for the local phase-QNN "
    "nn.Module route; torch.export.export, torch.export.save, and "
    "torch.export.load are checked on a local CPU value route only, with no "
    "provider, hardware, CUDA, dynamic-shape, AOTAutograd gradient-export, "
    "isolated benchmark, cross-runtime deployment, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchExportRoute:
    """One bounded PyTorch export audit route."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready export route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchExportAuditResult:
    """Local export-persistence evidence for bounded PyTorch QNN modules."""

    export_path: str
    export_size_bytes: int
    export_sha256: str
    state_dict_keys: tuple[str, ...]
    graph_node_count: int
    graph_signature: str
    graph_module_type: str
    exported_program_type: str
    original_loss: float
    exported_loss: float
    loaded_loss: float
    original_loss_error: float
    loaded_loss_error: float
    tolerance: float
    torch_version: str
    routes: tuple[PhaseTorchExportRoute, ...]
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_qnn_module_export_audit"
    claim_boundary: str = TORCH_EXPORT_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether bounded local export persistence passed."""
        return (
            self.route_status("module_exported_program") == "passed"
            and self.route_status("exported_program_file_round_trip") == "passed"
            and self.route_status("exported_program_loaded_cpu_replay") == "passed"
            and self.route_status("exported_program_graph_signature") == "passed"
            and self.original_loss_error <= self.tolerance
            and self.loaded_loss_error <= self.tolerance
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return export routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named PyTorch export route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch export route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready export audit evidence."""
        return {
            "export_path": self.export_path,
            "export_size_bytes": self.export_size_bytes,
            "export_sha256": self.export_sha256,
            "state_dict_keys": list(self.state_dict_keys),
            "graph_node_count": self.graph_node_count,
            "graph_signature": self.graph_signature,
            "graph_module_type": self.graph_module_type,
            "exported_program_type": self.exported_program_type,
            "original_loss": self.original_loss,
            "exported_loss": self.exported_loss,
            "loaded_loss": self.loaded_loss,
            "original_loss_error": self.original_loss_error,
            "loaded_loss_error": self.loaded_loss_error,
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


def run_torch_module_export_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    export_path: str | Path,
    tolerance: float = 1e-6,
) -> PhaseTorchExportAuditResult:
    """Audit local ``torch.export`` persistence for a bounded PyTorch module.

    Parameters
    ----------
    features:
        Real-valued feature matrix with shape ``(n_samples, n_parameters)``.
    labels:
        Binary label vector with shape ``(n_samples,)``.
    initial_params:
        Initial bounded phase-QNN parameter vector with width matching
        ``features``.
    export_path:
        Destination for the real ``torch.export.save`` artefact. Its parent
        directory must already exist.
    tolerance:
        Non-negative absolute tolerance for exported and loaded value replay.

    Returns
    -------
    PhaseTorchExportAuditResult
        Export file metadata plus graph-signature and local replay evidence.
    """
    resolved_path = Path(export_path)
    if not resolved_path.parent.exists():
        raise ValueError("export_path parent must exist")
    torch_module = _load_torch()
    export_module = _require_export_module(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "initial_params",
        _torch_values_to_numpy(initial_params),
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
    )
    original_loss = _module_loss(module)
    exported_program = export_module.export(module, ())
    exported_loss = _exported_program_loss(exported_program)
    export_module.save(exported_program, str(resolved_path))
    loaded_program = export_module.load(str(resolved_path))
    loaded_loss = _exported_program_loss(loaded_program)
    export_bytes = resolved_path.read_bytes()
    state_dict_keys = _state_dict_keys(exported_program)
    graph_signature = str(getattr(exported_program, "graph_signature", ""))
    graph_module = getattr(exported_program, "graph_module", None)
    graph_node_count = _graph_node_count(graph_module)
    original_loss_error = abs(original_loss - exported_loss)
    loaded_loss_error = abs(original_loss - loaded_loss)
    module_exported = bool(
        type(exported_program).__name__
        and callable(getattr(exported_program, "module", None))
        and original_loss_error <= tolerance_value
    )
    export_round_trip = bool(resolved_path.is_file() and len(export_bytes) > 0)
    loaded_replay = bool(
        type(loaded_program).__name__
        and callable(getattr(loaded_program, "module", None))
        and loaded_loss_error <= tolerance_value
    )
    graph_signature_present = bool(
        graph_node_count > 0
        and "PARAMETER" in graph_signature
        and "BUFFER" in graph_signature
        and "USER_OUTPUT" in graph_signature
    )
    routes = (
        PhaseTorchExportRoute(
            name="module_exported_program",
            status="passed" if module_exported else "failed",
            reason="torch.export.export produced a replayable ExportedProgram",
            requires=() if module_exported else ("torch_export_exported_program",),
        ),
        PhaseTorchExportRoute(
            name="exported_program_file_round_trip",
            status="passed" if export_round_trip else "failed",
            reason="torch.export.save wrote a persistent artifact and torch.export.load read it",
            requires=() if export_round_trip else ("torch_export_save_artifact",),
        ),
        PhaseTorchExportRoute(
            name="exported_program_loaded_cpu_replay",
            status="passed" if loaded_replay else "failed",
            reason="loaded ExportedProgram.module() reproduced the local CPU loss",
            requires=() if loaded_replay else ("matching_loaded_exported_program_value",),
        ),
        PhaseTorchExportRoute(
            name="exported_program_graph_signature",
            status="passed" if graph_signature_present else "failed",
            reason="ExportedProgram graph signature records parameter, buffers, and output",
            requires=() if graph_signature_present else ("exported_program_graph_signature",),
        ),
        PhaseTorchExportRoute(
            name="aotautograd_gradient_export_persistence",
            status="blocked",
            reason=(
                "AOTAutograd gradient-export persistence is separate from this "
                "local value-route torch.export artifact"
            ),
            requires=(
                "aotautograd_gradient_export_artifact",
                "gradient_export_replay_contract",
            ),
        ),
        PhaseTorchExportRoute(
            name="dynamic_shape_export",
            status="blocked",
            reason="dynamic-shape export needs explicit shape constraints and replay artifacts",
            requires=(
                "dynamic_shape_constraints",
                "multi_shape_export_replay_matrix",
            ),
        ),
    )
    return PhaseTorchExportAuditResult(
        export_path=str(resolved_path),
        export_size_bytes=len(export_bytes),
        export_sha256=hashlib.sha256(export_bytes).hexdigest(),
        state_dict_keys=state_dict_keys,
        graph_node_count=graph_node_count,
        graph_signature=graph_signature,
        graph_module_type=type(graph_module).__name__,
        exported_program_type=type(exported_program).__name__,
        original_loss=original_loss,
        exported_loss=exported_loss,
        loaded_loss=loaded_loss,
        original_loss_error=original_loss_error,
        loaded_loss_error=loaded_loss_error,
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        routes=routes,
    )


def _require_export_module(torch_module: Any) -> Any:
    """Return the ``torch.export`` module after checking required APIs."""
    export_module = getattr(torch_module, "export", None)
    if export_module is None:
        raise RuntimeError("PyTorch torch.export is unavailable")
    for name in ("export", "save", "load"):
        if not callable(getattr(export_module, name, None)):
            raise RuntimeError(f"PyTorch torch.export.{name} is unavailable")
    return export_module


def _module_loss(module: object) -> float:
    """Return the scalar loss from a no-argument bounded PyTorch module."""
    module_call = cast(Callable[[], object], module)
    return _torch_scalar_to_float(module_call())


def _exported_program_loss(exported_program: object) -> float:
    """Return the scalar loss from an exported no-argument PyTorch program."""
    module = getattr(exported_program, "module", None)
    if not callable(module):
        raise RuntimeError("ExportedProgram must expose module()")
    exported_module = module()
    exported_call = cast(Callable[[], object], exported_module)
    return _torch_scalar_to_float(exported_call())


def _state_dict_keys(exported_program: object) -> tuple[str, ...]:
    """Return string state keys recorded by an exported PyTorch program."""
    state_dict = getattr(exported_program, "state_dict", None)
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("ExportedProgram state_dict must be a mapping")
    return tuple(str(key) for key in state_dict)


def _graph_node_count(graph_module: object) -> int:
    """Return the number of nodes in an exported FX graph module."""
    graph = getattr(graph_module, "graph", None)
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        return 0
    return sum(1 for _ in nodes)


__all__ = [
    "PhaseTorchExportAuditResult",
    "PhaseTorchExportRoute",
    "TORCH_EXPORT_CLAIM_BOUNDARY",
    "run_torch_module_export_audit",
]
