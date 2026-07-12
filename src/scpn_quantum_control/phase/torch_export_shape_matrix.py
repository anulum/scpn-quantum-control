# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — torch export shape matrix module
# SCPN Quantum Control -- PyTorch Export Shape Matrix Utilities
"""Static-shape export matrix for bounded PyTorch phase-QNN modules.

This module keeps the bounded ``torch.export`` claim honest by recording
successful local export/load replay over multiple static feature shapes while
leaving dynamic-shape export constraints and replay as explicit blockers. It
wraps ``run_torch_module_export_audit(...)`` instead of expanding the lower
level export bridge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .torch_bridge import _as_non_negative_tolerance
from .torch_export import PhaseTorchExportAuditResult, run_torch_module_export_audit

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_EXPORT_SHAPE_MATRIX_SCHEMA = (
    "scpn_quantum_control.phase.torch_bounded_qnn_export_shape_matrix.v1"
)
TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY = (
    "bounded PyTorch static-shape export matrix for the local phase-QNN "
    "nn.Module route only; multiple static feature shapes are exported, saved, "
    "loaded, and replayed on a local CPU value route with no dynamic-shape, "
    "AOTAutograd gradient-export, CUDA, provider, hardware, isolated benchmark, "
    "cross-runtime deployment, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchExportShapeScenario:
    """One static-shape bounded phase-QNN export scenario."""

    name: str
    features: tuple[tuple[float, ...], ...]
    labels: tuple[float, ...]
    initial_params: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate scenario shape metadata and normalize scalar values."""
        name = str(self.name).strip()
        if not name:
            raise ValueError("scenario name must be non-empty")
        if any(character.isspace() for character in name):
            raise ValueError("scenario name must not contain whitespace")
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("scenario name must be a plain artifact stem")

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

        initial_params = tuple(float(value) for value in self.initial_params)
        if len(initial_params) != len(features[0]):
            raise ValueError("initial_params must match the feature column count")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "initial_params", initial_params)

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Return the static feature matrix shape for this scenario."""
        return (len(self.features), len(self.features[0]))

    @property
    def parameter_width(self) -> int:
        """Return the bounded phase-QNN parameter width for this scenario."""
        return len(self.initial_params)

    def feature_matrix(self) -> FloatArray:
        """Return the scenario features as a float64 matrix."""
        return np.array(self.features, dtype=np.float64)

    def label_vector(self) -> FloatArray:
        """Return the scenario labels as a float64 vector."""
        return np.array(self.labels, dtype=np.float64)

    def parameter_vector(self) -> FloatArray:
        """Return the scenario initial parameters as a float64 vector."""
        return np.array(self.initial_params, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready static export scenario metadata."""
        return {
            "name": self.name,
            "feature_shape": list(self.feature_shape),
            "label_count": len(self.labels),
            "parameter_width": self.parameter_width,
        }


@dataclass(frozen=True)
class PhaseTorchExportShapeMatrixRoute:
    """One route in the bounded PyTorch export shape matrix."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready export-shape route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchExportShapeMatrixRecord:
    """Static-shape export replay evidence for one bounded QNN scenario."""

    scenario_name: str
    feature_shape: tuple[int, int]
    parameter_width: int
    export_path: str
    export_size_bytes: int
    export_sha256: str
    graph_node_count: int
    graph_signature: str
    original_loss_error: float
    loaded_loss_error: float
    torch_version: str
    base_export_open_gaps: tuple[str, ...]
    base_export_passed: bool

    @property
    def passed(self) -> bool:
        """Return whether the static export/load replay passed."""
        return self.base_export_passed

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready static-shape export record evidence."""
        return {
            "scenario_name": self.scenario_name,
            "feature_shape": list(self.feature_shape),
            "parameter_width": self.parameter_width,
            "export_path": self.export_path,
            "export_size_bytes": self.export_size_bytes,
            "export_sha256": self.export_sha256,
            "graph_node_count": self.graph_node_count,
            "graph_signature": self.graph_signature,
            "original_loss_error": self.original_loss_error,
            "loaded_loss_error": self.loaded_loss_error,
            "torch_version": self.torch_version,
            "base_export_open_gaps": list(self.base_export_open_gaps),
            "base_export_passed": self.base_export_passed,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PhaseTorchExportShapeMatrixResult:
    """Static-shape matrix evidence for bounded PyTorch QNN export replay."""

    matrix_schema: str
    records: tuple[PhaseTorchExportShapeMatrixRecord, ...]
    routes: tuple[PhaseTorchExportShapeMatrixRoute, ...]
    tolerance: float
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    dynamic_shape_claim: bool = False
    method: str = "torch_bounded_qnn_export_shape_matrix"
    claim_boundary: str = TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY

    @property
    def scenario_count(self) -> int:
        """Return the number of static-shape export scenarios."""
        return len(self.records)

    @property
    def passed_count(self) -> int:
        """Return the number of static scenarios that passed replay."""
        return sum(1 for record in self.records if record.passed)

    @property
    def passed(self) -> bool:
        """Return whether the local static-shape matrix passed."""
        return (
            self.scenario_count > 0
            and self.passed_count == self.scenario_count
            and self.route_status("static_shape_export_matrix") == "passed"
            and self.route_status("multi_shape_local_replay") == "passed"
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return shape-matrix routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named export-shape matrix route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch export shape matrix route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready export-shape matrix evidence."""
        return {
            "matrix_schema": self.matrix_schema,
            "records": [record.to_dict() for record in self.records],
            "routes": {route.name: route.to_dict() for route in self.routes},
            "tolerance": self.tolerance,
            "scenario_count": self.scenario_count,
            "passed_count": self.passed_count,
            "open_gaps": list(self.open_gaps),
            "passed": self.passed,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "dynamic_shape_claim": self.dynamic_shape_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def default_torch_export_shape_scenarios() -> tuple[PhaseTorchExportShapeScenario, ...]:
    """Return deterministic bounded static-shape export scenarios."""
    return (
        PhaseTorchExportShapeScenario(
            name="one_parameter_static_shape",
            features=((0.0,), (float(np.pi),)),
            labels=(0.0, 1.0),
            initial_params=(0.45,),
        ),
        PhaseTorchExportShapeScenario(
            name="two_parameter_static_shape",
            features=(
                (0.0, 1.0),
                (float(np.pi / 2.0), -0.4),
                (float(np.pi), 0.25),
                (float(3.0 * np.pi / 2.0), 0.75),
            ),
            labels=(0.0, 1.0, 1.0, 0.0),
            initial_params=(0.25, -0.35),
        ),
    )


def run_torch_export_shape_matrix(
    *,
    export_dir: str | Path,
    scenarios: Sequence[PhaseTorchExportShapeScenario] | None = None,
    tolerance: float = 1e-6,
) -> PhaseTorchExportShapeMatrixResult:
    """Audit bounded PyTorch export replay over multiple static shapes.

    Parameters
    ----------
    export_dir:
        Directory where per-scenario ``torch.export.save`` artifacts are
        retained. The directory is created when missing and must not be a file.
    scenarios:
        Optional deterministic static-shape scenarios. When omitted, the
        default matrix covers one- and two-parameter bounded phase-QNN modules.
    tolerance:
        Non-negative absolute tolerance for local export/load value replay.

    Returns
    -------
    PhaseTorchExportShapeMatrixResult
        Matrix evidence for static-shape export replay plus explicit
        dynamic-shape blocker routes.
    """
    scenario_tuple = _as_scenarios(scenarios)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    resolved_dir = Path(export_dir)
    if resolved_dir.exists() and not resolved_dir.is_dir():
        raise ValueError("export_dir must be a directory")
    resolved_dir.mkdir(parents=True, exist_ok=True)

    records = tuple(
        _record_static_export_shape(
            scenario=scenario,
            export_path=resolved_dir / f"{scenario.name}.pt2",
            tolerance=tolerance_value,
        )
        for scenario in scenario_tuple
    )
    unique_shapes = {record.feature_shape for record in records}
    static_matrix_passed = bool(records) and all(record.passed for record in records)
    multi_shape_replay_passed = static_matrix_passed and len(unique_shapes) >= 2
    routes = (
        PhaseTorchExportShapeMatrixRoute(
            name="static_shape_export_matrix",
            status="passed" if static_matrix_passed else "failed",
            reason="each static-shape scenario exported, saved, loaded, and replayed locally",
            requires=() if static_matrix_passed else ("all_static_shape_exports_pass",),
        ),
        PhaseTorchExportShapeMatrixRoute(
            name="multi_shape_local_replay",
            status="passed" if multi_shape_replay_passed else "failed",
            reason="separate static ExportedProgram artifacts replay across distinct shapes",
            requires=()
            if multi_shape_replay_passed
            else ("two_or_more_distinct_static_feature_shapes",),
        ),
        PhaseTorchExportShapeMatrixRoute(
            name="dynamic_shape_constraints",
            status="blocked",
            reason=(
                "current bounded no-input module captures features and labels as "
                "buffers; dynamic-shape export requires input-driven symbolic "
                "shape constraints"
            ),
            requires=("input_driven_export_module", "symbolic_shape_constraint_policy"),
        ),
        PhaseTorchExportShapeMatrixRoute(
            name="dynamic_shape_replay_matrix",
            status="blocked",
            reason=(
                "dynamic-shape promotion requires one constrained ExportedProgram "
                "replayed across multiple input shapes"
            ),
            requires=("constrained_exported_program", "multi_shape_dynamic_replay_artifacts"),
        ),
    )
    return PhaseTorchExportShapeMatrixResult(
        matrix_schema=TORCH_EXPORT_SHAPE_MATRIX_SCHEMA,
        records=records,
        routes=routes,
        tolerance=tolerance_value,
    )


def _as_scenarios(
    scenarios: Sequence[PhaseTorchExportShapeScenario] | None,
) -> tuple[PhaseTorchExportShapeScenario, ...]:
    """Return validated export-shape scenarios."""
    scenario_tuple = (
        default_torch_export_shape_scenarios() if scenarios is None else tuple(scenarios)
    )
    if not scenario_tuple:
        raise ValueError("at least one export shape scenario is required")
    names = [scenario.name for scenario in scenario_tuple]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"duplicate scenario names: {', '.join(duplicate_names)}")
    return scenario_tuple


def _record_static_export_shape(
    *,
    scenario: PhaseTorchExportShapeScenario,
    export_path: Path,
    tolerance: float,
) -> PhaseTorchExportShapeMatrixRecord:
    """Run the base export audit and convert it into shape-matrix evidence."""
    audit = run_torch_module_export_audit(
        features=scenario.feature_matrix(),
        labels=scenario.label_vector(),
        initial_params=scenario.parameter_vector(),
        export_path=export_path,
        tolerance=tolerance,
    )
    return _record_from_audit(scenario=scenario, audit=audit)


def _record_from_audit(
    *,
    scenario: PhaseTorchExportShapeScenario,
    audit: PhaseTorchExportAuditResult,
) -> PhaseTorchExportShapeMatrixRecord:
    """Convert one base export audit into a matrix record."""
    return PhaseTorchExportShapeMatrixRecord(
        scenario_name=scenario.name,
        feature_shape=scenario.feature_shape,
        parameter_width=scenario.parameter_width,
        export_path=audit.export_path,
        export_size_bytes=audit.export_size_bytes,
        export_sha256=audit.export_sha256,
        graph_node_count=audit.graph_node_count,
        graph_signature=audit.graph_signature,
        original_loss_error=audit.original_loss_error,
        loaded_loss_error=audit.loaded_loss_error,
        torch_version=audit.torch_version,
        base_export_open_gaps=audit.open_gaps,
        base_export_passed=audit.passed,
    )


__all__ = [
    "PhaseTorchExportShapeMatrixRecord",
    "PhaseTorchExportShapeMatrixResult",
    "PhaseTorchExportShapeMatrixRoute",
    "PhaseTorchExportShapeScenario",
    "TORCH_EXPORT_SHAPE_MATRIX_CLAIM_BOUNDARY",
    "TORCH_EXPORT_SHAPE_MATRIX_SCHEMA",
    "default_torch_export_shape_scenarios",
    "run_torch_export_shape_matrix",
]
