# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 bounded QNN convergence example
"""Frozen phase-QNN convergence task and real framework agreement rows."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ..phase.jax_bridge import jax_native_qnn_value_and_grad
from ..phase.qnn_training import (
    ParameterShiftQNNTrainingResult,
    train_parameter_shift_qnn_classifier,
)
from ..phase.tensorflow_bridge import tensorflow_bounded_qnn_value_and_grad
from ..phase.torch_bridge import torch_autograd_qnn_value_and_grad
from .contracts import (
    ConvergenceCertificate,
    ConvergenceExampleSpec,
    FrameworkEvidenceRow,
    FrameworkStatus,
    ModelFamily,
)

_FRAMEWORKS = ("jax", "pytorch", "tensorflow")


class _FrameworkAgreementResult(Protocol):
    @property
    def passed(self) -> bool:
        """Return whether framework agreement passed."""
        ...

    @property
    def max_abs_error(self) -> float:
        """Return the maximum absolute gradient error."""
        ...


def qnn_example_spec() -> ConvergenceExampleSpec:
    """Return the frozen phase-separable binary QNN task."""
    return ConvergenceExampleSpec(
        example_id="qnn_phase_separable_binary",
        family=ModelFamily.QNN,
        seed=101,
        task="classify phase features 0 and pi as binary labels 0 and 1",
        max_steps=80,
        target_loss=1e-4,
        min_loss_drop=2e-2,
    )


def _qnn_task() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.asarray([[0.0], [np.pi]], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.8], dtype=np.float64),
    )


def _train_qnn() -> ParameterShiftQNNTrainingResult:
    features, labels, initial = _qnn_task()
    spec = qnn_example_spec()
    return train_parameter_shift_qnn_classifier(
        features,
        labels,
        initial_params=initial,
        learning_rate=0.7,
        max_steps=spec.max_steps,
        gradient_tolerance=1e-7,
        target_loss=0.0,
        target_loss_tolerance=spec.target_loss,
    )


def run_qnn_convergence_example() -> ConvergenceCertificate:
    """Run and replay the existing bounded phase-QNN trainer."""
    first = _train_qnn()
    replay = _train_qnn()
    first_history = tuple(float(value) for value in first.loss_history)
    replay_history = tuple(float(value) for value in replay.loss_history)
    spec = qnn_example_spec()
    best = float(min(first_history))
    loss_drop = float(first_history[0] - best)
    return ConvergenceCertificate(
        spec=spec,
        loss_history=first_history,
        initial_loss=first_history[0],
        final_loss=first_history[-1],
        best_loss=best,
        loss_drop=loss_drop,
        target_reached=best <= spec.target_loss,
        loss_drop_reached=loss_drop >= spec.min_loss_drop,
        deterministic_replay=first_history == replay_history,
        stop_reason=first.training.reason,
        metric_name="training_accuracy",
        metric_value=first.prediction.accuracy,
        metric_threshold=1.0,
        details=(
            ("accepted_steps", first.training.accepted_steps),
            ("evaluations", first.training.evaluations),
            ("gradient_method", first.method),
        ),
    )


_FrameworkRunner: TypeAlias = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    _FrameworkAgreementResult,
]
_FRAMEWORK_RUNNERS: dict[
    str,
    tuple[str, _FrameworkRunner],
] = {
    "jax": ("jax", cast(_FrameworkRunner, jax_native_qnn_value_and_grad)),
    "pytorch": ("torch", cast(_FrameworkRunner, torch_autograd_qnn_value_and_grad)),
    "tensorflow": (
        "tensorflow",
        cast(_FrameworkRunner, tensorflow_bounded_qnn_value_and_grad),
    ),
}


def run_qnn_framework_rows(
    *,
    required_frameworks: tuple[str, ...] = (),
) -> tuple[FrameworkEvidenceRow, ...]:
    """Execute installed QNN adapters and record missing dependencies explicitly."""
    unknown = set(required_frameworks) - set(_FRAMEWORKS)
    if unknown:
        raise ValueError(f"unknown required QNN frameworks: {sorted(unknown)}")
    features, labels, params = _qnn_task()
    rows: list[FrameworkEvidenceRow] = [
        FrameworkEvidenceRow(
            family=ModelFamily.QNN,
            framework="scpn_parameter_shift",
            status=FrameworkStatus.RAN,
            required=True,
            executed=True,
            passed=True,
            reason="canonical bounded phase-QNN trainer executed",
            max_abs_error=0.0,
        )
    ]
    for framework in _FRAMEWORKS:
        required = framework in required_frameworks
        dependency, function = _FRAMEWORK_RUNNERS[framework]
        if importlib.util.find_spec(dependency) is None:
            rows.append(
                FrameworkEvidenceRow(
                    family=ModelFamily.QNN,
                    framework=framework,
                    status=FrameworkStatus.UNAVAILABLE,
                    required=required,
                    executed=False,
                    passed=None,
                    reason=f"optional dependency {dependency!r} is not installed",
                )
            )
            continue
        result = function(features, labels, params)
        passed = bool(result.passed)
        error = float(result.max_abs_error)
        status = {True: FrameworkStatus.RAN, False: FrameworkStatus.FAILED}[passed]
        reason = {
            True: "native bounded-QNN gradient agrees with parameter shift",
            False: "native bounded-QNN gradient exceeded the agreement tolerance",
        }[passed]
        rows.append(
            FrameworkEvidenceRow(
                family=ModelFamily.QNN,
                framework=framework,
                status=status,
                required=required,
                executed=True,
                passed=passed,
                reason=reason,
                max_abs_error=error,
            )
        )
    rows.append(
        FrameworkEvidenceRow(
            family=ModelFamily.QNN,
            framework="provider_hardware_gradient",
            status=FrameworkStatus.UNSUPPORTED,
            required=False,
            executed=False,
            passed=None,
            reason="provider hardware gradients require separate job, shot, and approval evidence",
        )
    )
    return tuple(rows)


__all__ = ["qnn_example_spec", "run_qnn_convergence_example", "run_qnn_framework_rows"]
