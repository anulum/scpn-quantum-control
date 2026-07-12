# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — torch dynamic shape export module
# scpn-quantum-control -- PyTorch Dynamic-Shape Export Utilities
"""Dynamic-batch ``torch.export`` replay for bounded phase-QNN modules.

The audit in this module promotes one narrow dynamic-shape route: a bounded
phase-QNN PyTorch ``nn.Module`` whose features and labels are real inputs, not
captured buffers, is exported with a symbolic batch dimension and replayed after
``torch.export.save``/``torch.export.load`` across multiple concrete batch
sizes. It does not claim dynamic feature width, AOTAutograd gradient-export
persistence, CUDA, provider, hardware, cross-runtime deployment, isolated
benchmark, or performance readiness.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qnn_training import parameter_shift_qnn_classifier_loss
from .torch_bridge import (
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _load_torch,
    _torch_bounded_qnn_loss_tensor,
    _torch_nn_module_and_parameter,
    _torch_scalar_to_float,
    _torch_tensor,
    _torch_values_to_numpy,
)
from .torch_export import _graph_node_count, _require_export_module

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA = (
    "scpn_quantum_control.phase.torch_bounded_qnn_dynamic_shape_export.v1"
)
TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY = (
    "bounded PyTorch dynamic batch torch.export audit for the local phase-QNN "
    "nn.Module route only; one input-driven ExportedProgram is exported with "
    "symbolic batch constraints, saved, loaded, and replayed on a local CPU "
    "value route across multiple batch sizes with no dynamic feature-width, "
    "AOTAutograd gradient-export, CUDA, provider, hardware, isolated "
    "benchmark, cross-runtime deployment, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchDynamicShapeExportReplayCase:
    """One concrete replay case for a dynamic-batch phase-QNN export."""

    name: str
    features: tuple[tuple[float, ...], ...]
    labels: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate and normalize the replay case shape metadata."""
        name = str(self.name).strip()
        if not name:
            raise ValueError("replay case name must be non-empty")
        if any(character.isspace() for character in name):
            raise ValueError("replay case name must not contain whitespace")
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("replay case name must be a plain artifact stem")

        features = tuple(tuple(float(value) for value in row) for row in self.features)
        if not features:
            raise ValueError("features must contain at least one row")
        row_widths = {len(row) for row in features}
        if row_widths == {0}:
            raise ValueError("features must contain at least one column")
        if len(row_widths) != 1:
            raise ValueError("features must be rectangular")

        labels = tuple(float(value) for value in self.labels)
        if len(labels) != len(features):
            raise ValueError("labels must match the feature row count")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Return the concrete feature matrix shape for this replay case."""
        return (len(self.features), len(self.features[0]))

    @property
    def batch_size(self) -> int:
        """Return the replay case batch size."""
        return self.feature_shape[0]

    @property
    def feature_width(self) -> int:
        """Return the replay case feature width."""
        return self.feature_shape[1]

    def feature_matrix(self) -> FloatArray:
        """Return the replay case features as a float64 matrix."""
        return np.array(self.features, dtype=np.float64)

    def label_vector(self) -> FloatArray:
        """Return the replay case labels as a float64 vector."""
        return np.array(self.labels, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready replay-case metadata."""
        return {
            "name": self.name,
            "feature_shape": list(self.feature_shape),
            "batch_size": self.batch_size,
            "feature_width": self.feature_width,
            "label_count": len(self.labels),
        }


@dataclass(frozen=True)
class PhaseTorchDynamicShapeExportRecord:
    """Replay evidence for one batch size against one dynamic export artifact."""

    case_name: str
    feature_shape: tuple[int, int]
    batch_size: int
    reference_loss: float
    exported_loss: float
    loaded_loss: float
    original_loss_error: float
    loaded_loss_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Return whether exported and loaded value replay matched reference."""
        return (
            np.isfinite(self.original_loss_error)
            and np.isfinite(self.loaded_loss_error)
            and self.original_loss_error <= self.tolerance
            and self.loaded_loss_error <= self.tolerance
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dynamic-shape replay evidence."""
        return {
            "case_name": self.case_name,
            "feature_shape": list(self.feature_shape),
            "batch_size": self.batch_size,
            "reference_loss": self.reference_loss,
            "exported_loss": self.exported_loss,
            "loaded_loss": self.loaded_loss,
            "original_loss_error": self.original_loss_error,
            "loaded_loss_error": self.loaded_loss_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PhaseTorchDynamicShapeExportRoute:
    """One route in the bounded PyTorch dynamic-shape export audit."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dynamic-shape route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchDynamicShapeExportResult:
    """Local dynamic-batch export evidence for bounded PyTorch QNN modules."""

    matrix_schema: str
    export_path: str
    export_size_bytes: int
    export_sha256: str
    feature_width: int
    batch_sizes: tuple[int, ...]
    range_constraints: str
    graph_signature: str
    graph_node_count: int
    records: tuple[PhaseTorchDynamicShapeExportRecord, ...]
    routes: tuple[PhaseTorchDynamicShapeExportRoute, ...]
    tolerance: float
    torch_version: str
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    dynamic_shape_claim: bool = True
    dynamic_feature_width_claim: bool = False
    method: str = "torch_bounded_qnn_dynamic_shape_export_audit"
    claim_boundary: str = TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY

    @property
    def replay_count(self) -> int:
        """Return the number of concrete replay cases."""
        return len(self.records)

    @property
    def passed_count(self) -> int:
        """Return the number of replay cases that matched the reference loss."""
        return sum(1 for record in self.records if record.passed)

    @property
    def passed(self) -> bool:
        """Return whether the local dynamic-batch export route passed."""
        return (
            self.replay_count > 0
            and self.passed_count == self.replay_count
            and self.route_status("dynamic_batch_shape_constraints") == "passed"
            and self.route_status("dynamic_exported_program_file_round_trip") == "passed"
            and self.route_status("multi_batch_loaded_cpu_replay") == "passed"
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return dynamic-shape export routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named PyTorch dynamic-shape export route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch dynamic-shape export route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready dynamic-shape export audit evidence."""
        return {
            "matrix_schema": self.matrix_schema,
            "export_path": self.export_path,
            "export_size_bytes": self.export_size_bytes,
            "export_sha256": self.export_sha256,
            "feature_width": self.feature_width,
            "batch_sizes": list(self.batch_sizes),
            "range_constraints": self.range_constraints,
            "graph_signature": self.graph_signature,
            "graph_node_count": self.graph_node_count,
            "records": [record.to_dict() for record in self.records],
            "routes": {route.name: route.to_dict() for route in self.routes},
            "tolerance": self.tolerance,
            "torch_version": self.torch_version,
            "replay_count": self.replay_count,
            "passed_count": self.passed_count,
            "open_gaps": list(self.open_gaps),
            "passed": self.passed,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "dynamic_shape_claim": self.dynamic_shape_claim,
            "dynamic_feature_width_claim": self.dynamic_feature_width_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def default_torch_dynamic_shape_export_replay_cases() -> tuple[
    PhaseTorchDynamicShapeExportReplayCase, ...
]:
    """Return deterministic same-width replay cases with varied batch sizes."""
    return (
        PhaseTorchDynamicShapeExportReplayCase(
            name="batch_2",
            features=((0.0, 1.0), (float(np.pi / 2.0), -0.4)),
            labels=(0.0, 1.0),
        ),
        PhaseTorchDynamicShapeExportReplayCase(
            name="batch_4",
            features=(
                (0.0, 1.0),
                (float(np.pi / 2.0), -0.4),
                (float(np.pi), 0.25),
                (float(3.0 * np.pi / 2.0), 0.75),
            ),
            labels=(0.0, 1.0, 1.0, 0.0),
        ),
        PhaseTorchDynamicShapeExportReplayCase(
            name="batch_5",
            features=(
                (0.0, 1.0),
                (float(np.pi / 2.0), -0.4),
                (float(np.pi), 0.25),
                (float(3.0 * np.pi / 2.0), 0.75),
                (float(2.0 * np.pi), -0.1),
            ),
            labels=(0.0, 1.0, 1.0, 0.0, 0.0),
        ),
    )


def run_torch_dynamic_shape_export_audit(
    *,
    export_path: str | Path,
    replay_cases: Sequence[PhaseTorchDynamicShapeExportReplayCase] | None = None,
    initial_params: ArrayLike | object | None = None,
    tolerance: float = 1e-6,
) -> PhaseTorchDynamicShapeExportResult:
    """Audit one dynamic-batch ``torch.export`` artifact for bounded phase-QNNs.

    Parameters
    ----------
    export_path:
        Destination for the real ``torch.export.save`` artifact. Its parent
        directory must already exist.
    replay_cases:
        Optional same-width replay cases. At least two distinct batch sizes are
        required so the audit proves a symbolic batch dimension, not a static
        export.
    initial_params:
        Initial bounded phase-QNN parameter vector. When omitted, a deterministic
        width-two vector matching the default replay cases is used.
    tolerance:
        Non-negative absolute tolerance for exported and loaded value replay.

    Returns
    -------
    PhaseTorchDynamicShapeExportResult
        Dynamic-batch export artifact metadata plus multi-batch replay evidence.
    """
    resolved_path = Path(export_path)
    if not resolved_path.parent.exists():
        raise ValueError("export_path parent must exist")

    cases = _as_replay_cases(replay_cases)
    feature_width = _common_feature_width(cases)
    parameter_values = _as_parameter_vector(
        "initial_params",
        _torch_values_to_numpy((0.25, -0.35) if initial_params is None else initial_params),
        width=feature_width,
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    torch_module = _load_torch()
    export_module = _require_export_module(torch_module)
    module = _build_dynamic_bounded_qnn_module(
        torch_module=torch_module,
        parameter_values=parameter_values,
    )
    batch_sizes = tuple(case.batch_size for case in cases)
    example_case = cases[0]
    example_features = _torch_tensor(torch_module, example_case.feature_matrix())
    example_labels = _torch_tensor(torch_module, example_case.label_vector())
    batch_dim = _dynamic_batch_dim(
        export_module=export_module,
        min_batch=min(batch_sizes),
        max_batch=max(batch_sizes),
    )
    exported_program = export_module.export(
        module,
        (example_features, example_labels),
        dynamic_shapes={
            "features": {0: batch_dim},
            "labels": {0: batch_dim},
        },
    )
    export_module.save(exported_program, str(resolved_path))
    loaded_program = export_module.load(str(resolved_path))
    export_bytes = resolved_path.read_bytes()
    graph_signature = str(getattr(exported_program, "graph_signature", ""))
    range_constraints = str(getattr(exported_program, "range_constraints", ""))
    graph_node_count = _graph_node_count(getattr(exported_program, "graph_module", None))
    records = tuple(
        _record_replay_case(
            torch_module=torch_module,
            exported_program=exported_program,
            loaded_program=loaded_program,
            case=case,
            parameter_values=parameter_values,
            tolerance=tolerance_value,
        )
        for case in cases
    )
    routes = _dynamic_export_routes(
        records=records,
        resolved_path=resolved_path,
        export_size_bytes=len(export_bytes),
        batch_sizes=batch_sizes,
        graph_signature=graph_signature,
        range_constraints=range_constraints,
        loaded_program=loaded_program,
    )
    return PhaseTorchDynamicShapeExportResult(
        matrix_schema=TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA,
        export_path=str(resolved_path),
        export_size_bytes=len(export_bytes),
        export_sha256=hashlib.sha256(export_bytes).hexdigest(),
        feature_width=feature_width,
        batch_sizes=batch_sizes,
        range_constraints=range_constraints,
        graph_signature=graph_signature,
        graph_node_count=graph_node_count,
        records=records,
        routes=routes,
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
    )


def _as_replay_cases(
    replay_cases: Sequence[PhaseTorchDynamicShapeExportReplayCase] | None,
) -> tuple[PhaseTorchDynamicShapeExportReplayCase, ...]:
    """Return validated dynamic-shape replay cases."""
    case_tuple = (
        default_torch_dynamic_shape_export_replay_cases()
        if replay_cases is None
        else tuple(replay_cases)
    )
    if not case_tuple:
        raise ValueError("at least two distinct batch sizes are required")
    names = [case.name for case in case_tuple]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate replay case names: {', '.join(duplicate_names)}")
    batch_sizes = {case.batch_size for case in case_tuple}
    if len(batch_sizes) < 2:
        raise ValueError("at least two distinct batch sizes are required")
    return case_tuple


def _common_feature_width(
    cases: Sequence[PhaseTorchDynamicShapeExportReplayCase],
) -> int:
    """Return the common feature width or fail closed."""
    feature_widths = {case.feature_width for case in cases}
    if len(feature_widths) != 1:
        raise ValueError("all dynamic replay cases must share one feature width")
    return feature_widths.pop()


def _build_dynamic_bounded_qnn_module(
    *,
    torch_module: Any,
    parameter_values: FloatArray,
) -> Any:
    """Return an input-driven bounded phase-QNN ``nn.Module``."""
    module_base, parameter_cls = _torch_nn_module_and_parameter(torch_module)
    parameter_tensor = _torch_tensor(torch_module, parameter_values)

    class _DynamicBoundedPhaseQNNModule(module_base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.params = parameter_cls(parameter_tensor, requires_grad=True)
            self.feature_width = int(parameter_values.shape[0])
            self.host_boundary = False
            self.native_framework_autodiff = True
            self.claim_boundary = "bounded_torch_dynamic_shape_export_replay"

        def forward(self, features: Any, labels: Any) -> Any:
            return _torch_bounded_qnn_loss_tensor(
                torch_module,
                features,
                labels,
                self.params,
            )

    return _DynamicBoundedPhaseQNNModule()


def _dynamic_batch_dim(
    *,
    export_module: object,
    min_batch: int,
    max_batch: int,
) -> object:
    """Return a PyTorch symbolic batch dimension for ``torch.export``."""
    dim_factory = getattr(export_module, "Dim", None)
    if not callable(dim_factory):
        raise RuntimeError("PyTorch torch.export.Dim is unavailable")
    return dim_factory("batch", min=int(min_batch), max=int(max_batch))


def _record_replay_case(
    *,
    torch_module: Any,
    exported_program: object,
    loaded_program: object,
    case: PhaseTorchDynamicShapeExportReplayCase,
    parameter_values: FloatArray,
    tolerance: float,
) -> PhaseTorchDynamicShapeExportRecord:
    """Replay one concrete case against exported and loaded programs."""
    reference_loss = float(
        parameter_shift_qnn_classifier_loss(
            case.feature_matrix(),
            case.label_vector(),
            parameter_values,
        ),
    )
    exported_loss = _exported_program_loss_for_case(
        torch_module=torch_module,
        exported_program=exported_program,
        case=case,
    )
    loaded_loss = _exported_program_loss_for_case(
        torch_module=torch_module,
        exported_program=loaded_program,
        case=case,
    )
    return PhaseTorchDynamicShapeExportRecord(
        case_name=case.name,
        feature_shape=case.feature_shape,
        batch_size=case.batch_size,
        reference_loss=reference_loss,
        exported_loss=exported_loss,
        loaded_loss=loaded_loss,
        original_loss_error=abs(reference_loss - exported_loss),
        loaded_loss_error=abs(reference_loss - loaded_loss),
        tolerance=tolerance,
    )


def _exported_program_loss_for_case(
    *,
    torch_module: Any,
    exported_program: object,
    case: PhaseTorchDynamicShapeExportReplayCase,
) -> float:
    """Return scalar loss from an exported program for one replay case."""
    module_factory = getattr(exported_program, "module", None)
    if not callable(module_factory):
        raise RuntimeError("ExportedProgram must expose module()")
    exported_module = module_factory()
    exported_call = cast(Callable[[object, object], object], exported_module)
    feature_tensor = _torch_tensor(torch_module, case.feature_matrix())
    label_tensor = _torch_tensor(torch_module, case.label_vector())
    return _torch_scalar_to_float(exported_call(feature_tensor, label_tensor))


def _dynamic_export_routes(
    *,
    records: Sequence[PhaseTorchDynamicShapeExportRecord],
    resolved_path: Path,
    export_size_bytes: int,
    batch_sizes: Sequence[int],
    graph_signature: str,
    range_constraints: str,
    loaded_program: object,
) -> tuple[PhaseTorchDynamicShapeExportRoute, ...]:
    """Build fail-closed route statuses for the dynamic-shape export audit."""
    distinct_batch_count = len(set(batch_sizes))
    dynamic_constraints_passed = bool(
        distinct_batch_count >= 2
        and "VR[" in range_constraints
        and "USER_INPUT" in graph_signature
        and "features" in graph_signature
        and "labels" in graph_signature
    )
    loaded_module_factory = getattr(loaded_program, "module", None)
    round_trip_passed = bool(
        resolved_path.is_file() and export_size_bytes > 0 and callable(loaded_module_factory)
    )
    replay_passed = bool(
        records and distinct_batch_count >= 2 and all(record.passed for record in records)
    )
    return (
        PhaseTorchDynamicShapeExportRoute(
            name="dynamic_batch_shape_constraints",
            status="passed" if dynamic_constraints_passed else "failed",
            reason=(
                "ExportedProgram graph signature records input tensors and a "
                "symbolic batch range constraint"
            ),
            requires=()
            if dynamic_constraints_passed
            else ("symbolic_batch_range_constraint", "input_tensor_graph_signature"),
        ),
        PhaseTorchDynamicShapeExportRoute(
            name="dynamic_exported_program_file_round_trip",
            status="passed" if round_trip_passed else "failed",
            reason="torch.export.save wrote an artifact and torch.export.load returned a module factory",
            requires=()
            if round_trip_passed
            else ("torch_export_save_artifact", "torch_export_load_module_factory"),
        ),
        PhaseTorchDynamicShapeExportRoute(
            name="multi_batch_loaded_cpu_replay",
            status="passed" if replay_passed else "failed",
            reason=(
                "one loaded ExportedProgram reproduced parameter-shift reference "
                "losses across multiple concrete batch sizes"
            ),
            requires=()
            if replay_passed
            else ("two_or_more_batch_replay_records", "matching_loaded_exported_values"),
        ),
        PhaseTorchDynamicShapeExportRoute(
            name="aotautograd_gradient_export_persistence",
            status="blocked",
            reason=(
                "AOTAutograd gradient-export persistence is separate from this "
                "local value-route dynamic-batch torch.export artifact"
            ),
            requires=(
                "aotautograd_gradient_export_artifact",
                "gradient_export_replay_contract",
            ),
        ),
        PhaseTorchDynamicShapeExportRoute(
            name="cuda_dynamic_shape_export_replay",
            status="blocked",
            reason="CUDA dynamic-shape export replay requires compatible visible hardware",
            requires=("compatible_cuda_device", "cuda_dynamic_shape_export_replay_artifact"),
        ),
        PhaseTorchDynamicShapeExportRoute(
            name="cross_runtime_dynamic_shape_export_replay",
            status="blocked",
            reason="cross-runtime replay requires a declared target runtime and loader contract",
            requires=("target_runtime_contract", "cross_runtime_replay_artifact"),
        ),
        PhaseTorchDynamicShapeExportRoute(
            name="isolated_benchmark_dynamic_shape_export",
            status="blocked",
            reason="performance promotion needs isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )


__all__ = [
    "PhaseTorchDynamicShapeExportRecord",
    "PhaseTorchDynamicShapeExportReplayCase",
    "PhaseTorchDynamicShapeExportResult",
    "PhaseTorchDynamicShapeExportRoute",
    "TORCH_DYNAMIC_SHAPE_EXPORT_CLAIM_BOUNDARY",
    "TORCH_DYNAMIC_SHAPE_EXPORT_SCHEMA",
    "default_torch_dynamic_shape_export_replay_cases",
    "run_torch_dynamic_shape_export_audit",
]
