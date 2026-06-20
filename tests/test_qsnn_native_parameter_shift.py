# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- QSNN native parameter-shift delegation tests
"""Regression tests for QSNN training through native differentiable gradients."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import GradientResult, Parameter, ParameterShiftRule
from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer
from scpn_quantum_control.qsnn.training import QSNNTrainer

FloatArray = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_qsnn_trainer_uses_native_parameter_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    """QSNN training should delegate gradient evaluation to the core primitive."""

    import scpn_quantum_control.qsnn.training as training

    calls: list[tuple[tuple[float, ...], tuple[str, ...]]] = []

    def fake_value_and_grad(
        objective: ScalarObjective,
        values: Sequence[float],
        *,
        parameters: Sequence[Parameter],
        rule: ParameterShiftRule | None = None,
    ) -> GradientResult:
        del rule
        parameter_values = np.asarray(values, dtype=np.float64)
        calls.append(
            (
                tuple(float(value) for value in parameter_values),
                tuple(parameter.name for parameter in parameters),
            )
        )
        value = float(objective(parameter_values))
        return GradientResult(
            value=value,
            gradient=np.full(len(parameter_values), 0.25),
            method="parameter_shift",
            shift=math.pi / 2,
            coefficient=0.5,
            evaluations=1 + 2 * len(parameter_values),
            parameter_names=tuple(parameter.name for parameter in parameters),
            trainable=tuple(parameter.trainable for parameter in parameters),
        )

    monkeypatch.setattr(training, "value_and_parameter_shift_grad", fake_value_and_grad)

    layer = QuantumDenseLayer(1, 2, seed=0)
    trainer = QSNNTrainer(layer)
    gradient = trainer.parameter_shift_gradient(np.array([0.5, 0.25]), np.array([1.0]))

    assert calls
    assert calls[0][1] == ("synapse_0_0", "synapse_0_1")
    _assert_allclose(gradient, [[0.25, 0.25]])
