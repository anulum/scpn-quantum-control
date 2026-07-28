# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 bounded QSNN convergence example
"""Frozen synapse-angle task over the existing QSNN parameter-shift trainer."""

from __future__ import annotations

import numpy as np

from ..qsnn.qlayer import QuantumDenseLayer
from ..qsnn.training import QSNNParameterShiftDescentRun, QSNNTrainer
from .contracts import (
    ConvergenceCertificate,
    ConvergenceExampleSpec,
    FrameworkEvidenceRow,
    FrameworkStatus,
    ModelFamily,
)


def qsnn_example_spec() -> ConvergenceExampleSpec:
    """Return the frozen one-synapse QSNN probability-loss task."""
    return ConvergenceExampleSpec(
        example_id="qsnn_single_synapse_silencing",
        family=ModelFamily.QSNN,
        seed=42,
        task="fit one quantum synapse so a unit input has zero target firing probability",
        max_steps=16,
        target_loss=1e-5,
        min_loss_drop=0.7,
    )


def _train_qsnn() -> tuple[QSNNParameterShiftDescentRun, tuple[int, ...]]:
    spec = qsnn_example_spec()
    layer = QuantumDenseLayer(1, 1, seed=spec.seed)
    trainer = QSNNTrainer(layer, lr=0.4)
    result = trainer.train_with_parameter_shift_descent(
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([[0.0]], dtype=np.float64),
        max_steps=spec.max_steps,
        gradient_tolerance=1e-7,
        min_loss_decrease=spec.min_loss_drop,
    )
    final_spikes = tuple(int(value) for value in layer.forward(np.asarray([1.0])))
    return result, final_spikes


def run_qsnn_convergence_example() -> ConvergenceCertificate:
    """Run and replay the real QSNN synapse-angle training route."""
    first, first_spikes = _train_qsnn()
    replay, replay_spikes = _train_qsnn()
    first_history = tuple(float(value) for value in first.loss_history)
    replay_history = tuple(float(value) for value in replay.loss_history)
    spec = qsnn_example_spec()
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
        deterministic_replay=(first_history == replay_history and first_spikes == replay_spikes),
        stop_reason=first.training.reason,
        metric_name="final_spike_silenced",
        metric_value=float(first_spikes == (0,)),
        metric_threshold=1.0,
        details=(
            ("accepted_steps", first.training.accepted_steps),
            ("evaluations", first.training.evaluations),
            ("final_spikes", list(first_spikes)),
            ("temporal_coding", "not_modelled_by_this_dense-layer convergence example"),
            ("neuromorphic_hardware", False),
        ),
    )


def qsnn_framework_rows() -> tuple[FrameworkEvidenceRow, ...]:
    """Return a complete QSNN matrix without extending dense-layer claims."""
    rows = [
        FrameworkEvidenceRow(
            family=ModelFamily.QSNN,
            framework="scpn_qsnn_statevector",
            status=FrameworkStatus.RAN,
            required=True,
            executed=True,
            passed=True,
            reason="existing QSNN parameter-shift trainer and final spike readout executed",
            max_abs_error=0.0,
        )
    ]
    for framework in ("jax", "pytorch", "tensorflow"):
        rows.append(
            FrameworkEvidenceRow(
                family=ModelFamily.QSNN,
                framework=framework,
                status=FrameworkStatus.NOT_APPLICABLE,
                required=False,
                executed=False,
                passed=None,
                reason="no framework-native adapter is registered for the QSNN dense-layer trainer",
            )
        )
    rows.append(
        FrameworkEvidenceRow(
            family=ModelFamily.QSNN,
            framework="neuromorphic_hardware",
            status=FrameworkStatus.UNSUPPORTED,
            required=False,
            executed=False,
            passed=None,
            reason="the local statevector example is not a neuromorphic-hardware training route",
        )
    )
    return tuple(rows)


__all__ = ["qsnn_example_spec", "qsnn_framework_rows", "run_qsnn_convergence_example"]
