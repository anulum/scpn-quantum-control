# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — torch training loop matrix module
# SCPN Quantum Control -- PyTorch Training-Loop Matrix Utilities
"""Training-loop matrix utilities for bounded PyTorch phase-QNN modules.

This module expands the bounded ``run_torch_training_loop_audit(...)`` route
into a deterministic matrix of local phase-QNN training scenarios. The matrix
records loss descent, parameter-update magnitude, gradient parity against SCPN
parameter-shift references, and compile-mode coverage while keeping CUDA,
provider, hardware, isolated-benchmark, arbitrary-architecture, and performance
claims blocked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .torch_bridge import run_torch_training_loop_audit
from .torch_bridge_contracts import PhaseTorchTrainingLoopAuditResult

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_TRAINING_LOOP_MATRIX_SCHEMA = (
    "scpn_quantum_control.phase.torch_bounded_qnn_training_loop_matrix.v1"
)
TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY = (
    "bounded PyTorch training-loop matrix for the local phase-QNN nn.Module "
    "route only; deterministic local CPU scenarios record loss descent, "
    "parameter updates, compile-mode coverage, and gradient parity against "
    "SCPN parameter-shift references with no CUDA, provider, hardware, "
    "arbitrary-architecture, isolated benchmark, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchTrainingLoopScenario:
    """One deterministic bounded PyTorch phase-QNN training-loop scenario."""

    name: str
    features: tuple[tuple[float, ...], ...]
    labels: tuple[float, ...]
    initial_params: tuple[float, ...]
    learning_rate: float
    steps: int
    fullgraph: bool
    dynamic: bool

    def __post_init__(self) -> None:
        """Validate scenario shape and training controls."""
        if not self.name.strip():
            raise ValueError("training-loop scenario name must be non-empty")
        if any(character.isspace() for character in self.name):
            raise ValueError("training-loop scenario name must not contain whitespace")
        if not self.features:
            raise ValueError("training-loop scenario features must be non-empty")
        width = len(self.features[0])
        if width == 0:
            raise ValueError("training-loop scenario feature rows must be non-empty")
        if any(len(row) != width for row in self.features):
            raise ValueError("training-loop scenario feature rows must share one width")
        if len(self.labels) != len(self.features):
            raise ValueError("training-loop scenario labels must match feature rows")
        if len(self.initial_params) != width:
            raise ValueError("training-loop scenario initial_params must match feature width")
        if self.learning_rate <= 0.0:
            raise ValueError("training-loop scenario learning_rate must be positive")
        if self.steps <= 0 or isinstance(self.steps, bool):
            raise ValueError("training-loop scenario steps must be a positive integer")

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Return the scenario feature-matrix shape."""
        return (len(self.features), len(self.features[0]))

    @property
    def parameter_width(self) -> int:
        """Return the scenario parameter-vector width."""
        return len(self.initial_params)

    def feature_matrix(self) -> FloatArray:
        """Return the scenario features as a float64 matrix."""
        return cast(FloatArray, np.asarray(self.features, dtype=np.float64))

    def label_vector(self) -> FloatArray:
        """Return the scenario labels as a float64 vector."""
        return cast(FloatArray, np.asarray(self.labels, dtype=np.float64))

    def parameter_vector(self) -> FloatArray:
        """Return the scenario initial parameters as a float64 vector."""
        return cast(FloatArray, np.asarray(self.initial_params, dtype=np.float64))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready scenario metadata."""
        return {
            "name": self.name,
            "features": [list(row) for row in self.features],
            "labels": list(self.labels),
            "initial_params": list(self.initial_params),
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
        }


@dataclass(frozen=True)
class PhaseTorchTrainingLoopMatrixRoute:
    """One route in the bounded PyTorch training-loop matrix."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready training-loop route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchTrainingLoopMatrixRecord:
    """One bounded PyTorch training-loop matrix record."""

    scenario_name: str
    feature_shape: tuple[int, int]
    parameter_width: int
    steps: int
    learning_rate: float
    fullgraph: bool
    dynamic: bool
    initial_loss: float
    final_loss: float
    loss_drop: float
    max_abs_gradient_error: float
    l2_gradient_error: float
    parameter_update_norm: float
    parameter_update_supported: bool
    compile_supported: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready matrix record metadata."""
        return {
            "scenario_name": self.scenario_name,
            "feature_shape": list(self.feature_shape),
            "parameter_width": self.parameter_width,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_drop": self.loss_drop,
            "max_abs_gradient_error": self.max_abs_gradient_error,
            "l2_gradient_error": self.l2_gradient_error,
            "parameter_update_norm": self.parameter_update_norm,
            "parameter_update_supported": self.parameter_update_supported,
            "compile_supported": self.compile_supported,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PhaseTorchTrainingLoopMatrixResult:
    """Matrix evidence for bounded PyTorch phase-QNN training loops."""

    matrix_schema: str
    records: tuple[PhaseTorchTrainingLoopMatrixRecord, ...]
    routes: tuple[PhaseTorchTrainingLoopMatrixRoute, ...]
    tolerance: float
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_qnn_training_loop_matrix"
    claim_boundary: str = TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY

    @property
    def scenario_count(self) -> int:
        """Return the number of matrix scenarios."""
        return len(self.records)

    @property
    def passed_count(self) -> int:
        """Return the number of passing local scenario records."""
        return sum(1 for record in self.records if record.passed)

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return matrix routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    @property
    def passed(self) -> bool:
        """Return whether local training-loop matrix evidence passed."""
        return (
            self.route_status("multi_scenario_training_loop") == "passed"
            and self.route_status("training_loop_gradient_parity") == "passed"
            and self.route_status("training_loop_loss_descent") == "passed"
            and self.route_status("compile_mode_matrix") == "passed"
            and all(route.status != "failed" for route in self.routes)
        )

    def route_status(self, name: str) -> str:
        """Return the status for a named training-loop matrix route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch training-loop matrix route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready training-loop matrix evidence."""
        return {
            "matrix_schema": self.matrix_schema,
            "scenario_count": self.scenario_count,
            "passed_count": self.passed_count,
            "records": [record.to_dict() for record in self.records],
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "tolerance": self.tolerance,
            "passed": self.passed,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def default_torch_training_loop_scenarios() -> tuple[PhaseTorchTrainingLoopScenario, ...]:
    """Return deterministic bounded phase-QNN training-loop scenarios."""
    return (
        PhaseTorchTrainingLoopScenario(
            name="one_parameter_fullgraph_static",
            features=((0.0,), (np.pi,)),
            labels=(0.0, 1.0),
            initial_params=(0.45,),
            learning_rate=0.2,
            steps=3,
            fullgraph=True,
            dynamic=False,
        ),
        PhaseTorchTrainingLoopScenario(
            name="two_parameter_non_fullgraph_dynamic_request",
            features=((0.0, 1.0), (np.pi / 2.0, -0.4), (np.pi, 0.25)),
            labels=(0.0, 1.0, 1.0),
            initial_params=(0.25, -0.35),
            learning_rate=0.08,
            steps=2,
            fullgraph=False,
            dynamic=True,
        ),
    )


def run_torch_training_loop_matrix(
    *,
    scenarios: Sequence[PhaseTorchTrainingLoopScenario] | None = None,
    tolerance: float = 1e-6,
) -> PhaseTorchTrainingLoopMatrixResult:
    """Run a bounded PyTorch training-loop evidence matrix.

    Parameters
    ----------
    scenarios:
        Optional deterministic training-loop scenarios. When omitted, the
        default matrix covers a one-parameter fullgraph-static route and a
        two-parameter non-fullgraph dynamic-request route.
    tolerance:
        Non-negative absolute tolerance passed to each bounded training-loop
        audit and used when classifying aggregate loss-descent and gradient
        parity routes.

    Returns
    -------
    PhaseTorchTrainingLoopMatrixResult
        Per-scenario records and fail-closed route classification for local
        bounded training-loop evidence.
    """
    tolerance_value = _as_non_negative_tolerance(tolerance)
    selected = tuple(default_torch_training_loop_scenarios() if scenarios is None else scenarios)
    _validate_scenarios(selected)
    records = tuple(
        _record_from_audit(
            scenario,
            run_torch_training_loop_audit(
                features=scenario.feature_matrix(),
                labels=scenario.label_vector(),
                initial_params=scenario.parameter_vector(),
                learning_rate=scenario.learning_rate,
                steps=scenario.steps,
                tolerance=tolerance_value,
                fullgraph=scenario.fullgraph,
                dynamic=scenario.dynamic,
            ),
        )
        for scenario in selected
    )
    return PhaseTorchTrainingLoopMatrixResult(
        matrix_schema=TORCH_TRAINING_LOOP_MATRIX_SCHEMA,
        records=records,
        routes=_classify_routes(records, tolerance=tolerance_value),
        tolerance=tolerance_value,
    )


def _record_from_audit(
    scenario: PhaseTorchTrainingLoopScenario,
    audit: PhaseTorchTrainingLoopAuditResult,
) -> PhaseTorchTrainingLoopMatrixRecord:
    """Convert a single training-loop audit result into a matrix record."""
    parameter_delta = audit.final_params - audit.initial_params
    parameter_update_norm = float(np.linalg.norm(parameter_delta))
    compile_supported = bool(
        audit.torch_compile_supported
        and audit.compiled_loss_supported
        and audit.func_grad_supported
    )
    return PhaseTorchTrainingLoopMatrixRecord(
        scenario_name=scenario.name,
        feature_shape=scenario.feature_shape,
        parameter_width=scenario.parameter_width,
        steps=audit.steps,
        learning_rate=audit.learning_rate,
        fullgraph=scenario.fullgraph,
        dynamic=scenario.dynamic,
        initial_loss=audit.initial_loss,
        final_loss=audit.final_loss,
        loss_drop=audit.initial_loss - audit.final_loss,
        max_abs_gradient_error=audit.max_abs_gradient_error,
        l2_gradient_error=audit.l2_gradient_error,
        parameter_update_norm=parameter_update_norm,
        parameter_update_supported=audit.parameter_update_supported,
        compile_supported=compile_supported,
        passed=bool(audit.passed and compile_supported and parameter_update_norm > 0.0),
    )


def _classify_routes(
    records: tuple[PhaseTorchTrainingLoopMatrixRecord, ...],
    *,
    tolerance: float,
) -> tuple[PhaseTorchTrainingLoopMatrixRoute, ...]:
    """Classify local and blocked training-loop matrix routes."""
    multi_scenario_passed = len(records) >= 2 and all(record.passed for record in records)
    gradient_parity_passed = all(record.max_abs_gradient_error <= tolerance for record in records)
    loss_descent_passed = all(record.loss_drop >= -tolerance for record in records)
    fullgraph_values = {record.fullgraph for record in records}
    dynamic_values = {record.dynamic for record in records}
    compile_matrix_passed = (
        all(record.compile_supported for record in records)
        and fullgraph_values == {False, True}
        and dynamic_values == {False, True}
    )
    return (
        PhaseTorchTrainingLoopMatrixRoute(
            name="multi_scenario_training_loop",
            status="passed" if multi_scenario_passed else "failed",
            reason="multiple bounded phase-QNN training scenarios passed locally",
            requires=()
            if multi_scenario_passed
            else ("two_or_more_passing_training_loop_scenarios",),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="training_loop_gradient_parity",
            status="passed" if gradient_parity_passed else "failed",
            reason="final and per-step gradients match SCPN parameter-shift references",
            requires=() if gradient_parity_passed else ("matching_parameter_shift_gradients",),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="training_loop_loss_descent",
            status="passed" if loss_descent_passed else "failed",
            reason="scenario loss histories do not increase beyond tolerance",
            requires=() if loss_descent_passed else ("bounded_loss_descent",),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="compile_mode_matrix",
            status="passed" if compile_matrix_passed else "failed",
            reason="fullgraph/static and non-fullgraph/dynamic-request compile modes are covered",
            requires=()
            if compile_matrix_passed
            else ("fullgraph_and_non_fullgraph_compile_modes", "dynamic_request_row"),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="cuda_training_loop",
            status="blocked",
            reason="CUDA training-loop execution requires compatible CUDA smoke artefacts",
            requires=("compatible_cuda_training_loop_artifact",),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="provider_hardware_training_loop",
            status="blocked",
            reason="provider and hardware training loops require live-ticketed execution evidence",
            requires=("provider_job_policy", "hardware_execution_ticket"),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="isolated_benchmark_training_loop",
            status="blocked",
            reason="training-loop performance promotion requires isolated benchmark artefacts",
            requires=("isolated_affinity_training_loop_benchmark",),
        ),
        PhaseTorchTrainingLoopMatrixRoute(
            name="arbitrary_architecture_training_loop",
            status="blocked",
            reason="the matrix covers bounded phase-QNN modules only",
            requires=("arbitrary_qnn_architecture_training_evidence",),
        ),
    )


def _validate_scenarios(scenarios: tuple[PhaseTorchTrainingLoopScenario, ...]) -> None:
    """Validate scenario set cardinality and unique names."""
    if not scenarios:
        raise ValueError("at least one training-loop scenario is required")
    names = tuple(scenario.name for scenario in scenarios)
    if len(set(names)) != len(names):
        raise ValueError("duplicate training-loop scenario name")


def _as_non_negative_tolerance(tolerance: float) -> float:
    """Return a finite non-negative tolerance."""
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    return value


__all__ = [
    "PhaseTorchTrainingLoopMatrixRecord",
    "PhaseTorchTrainingLoopMatrixResult",
    "PhaseTorchTrainingLoopMatrixRoute",
    "PhaseTorchTrainingLoopScenario",
    "TORCH_TRAINING_LOOP_MATRIX_CLAIM_BOUNDARY",
    "TORCH_TRAINING_LOOP_MATRIX_SCHEMA",
    "default_torch_training_loop_scenarios",
    "run_torch_training_loop_matrix",
]
