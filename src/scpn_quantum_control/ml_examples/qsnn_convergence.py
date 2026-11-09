# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example bounded QSNN convergence example
"""Frozen synapse-angle task over the existing QSNN parameter-shift trainer."""

from __future__ import annotations

from typing import cast

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


def qsnn_example_spec(*, protocol: str = "exact_mse_v2") -> ConvergenceExampleSpec:
    """Return a versioned, frozen one-synapse probability-loss task.

    Parameters
    ----------
    protocol
        ``exact_mse_v2`` selects the 1024-step exact-gradient budget;
        ``legacy_16_step`` retains the predecessor's budget as a diagnostic.

    Returns
    -------
    ConvergenceExampleSpec
        Identical data, seed and loss gates with an explicit protocol identifier.

    Raises
    ------
    ValueError
        If the protocol is not registered.

    """
    protocols = {
        "exact_mse_v2": ("qsnn_single_synapse_silencing_v2", 1024),
        "legacy_16_step": ("qsnn_single_synapse_silencing", 16),
    }
    if protocol not in protocols:
        raise ValueError("unknown QSNN convergence protocol")
    example_id, max_steps = protocols[protocol]
    return ConvergenceExampleSpec(
        example_id=example_id,
        family=ModelFamily.QSNN,
        seed=42,
        task="fit one quantum synapse so a unit input has zero target firing probability",
        max_steps=max_steps,
        target_loss=1e-5,
        min_loss_drop=0.7,
    )


def _train_qsnn(
    spec: ConvergenceExampleSpec,
) -> tuple[QSNNParameterShiftDescentRun, tuple[int, ...]]:
    """Execute the specified frozen task through the existing public trainer."""
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


def run_qsnn_convergence_example(*, protocol: str = "exact_mse_v2") -> ConvergenceCertificate:
    """Run and replay a frozen QSNN protocol, preserving its failed predecessor.

    Parameters
    ----------
    protocol
        Registered exact-gradient protocol. The current version also records
        a replayed 16-step diagnostic with unchanged accuracy and loss-drop gates.

    Returns
    -------
    ConvergenceCertificate
        Observed loss curve, replay verdict, final spike and actual evaluation
        accounting. Passing loss gates does not assert gradient stationarity.

    Raises
    ------
    ValueError
        If the protocol is unknown or the existing trainer refuses its run.

    """
    spec = qsnn_example_spec(protocol=protocol)
    first, first_spikes = _train_qsnn(spec)
    replay, replay_spikes = _train_qsnn(spec)
    first_history = tuple(float(value) for value in first.loss_history)
    replay_history = tuple(float(value) for value in replay.loss_history)
    best = float(min(first_history))
    loss_drop = float(first_history[0] - best)
    total_evaluations = first.training.evaluations + replay.training.evaluations
    predecessor_details: tuple[tuple[str, object], ...] = ()
    if protocol == "exact_mse_v2":
        predecessor = run_qsnn_convergence_example(protocol="legacy_16_step")
        total_evaluations += cast(int, dict(predecessor.details)["total_statevector_evaluations"])
        predecessor_details = (("predecessor", predecessor.to_dict()),)
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
            ("replay_evaluations", replay.training.evaluations),
            ("total_statevector_evaluations", total_evaluations),
            ("protocol", protocol),
            ("final_spikes", list(first_spikes)),
            ("temporal_coding", "not_modelled_by_this_dense-layer convergence example"),
            ("neuromorphic_hardware", False),
        )
        + predecessor_details,
    )


def qsnn_framework_rows(
    *, certificate: ConvergenceCertificate | None = None
) -> tuple[FrameworkEvidenceRow, ...]:
    """Bind native QSNN status to an executed, registered training certificate.

    Parameters
    ----------
    certificate
        Existing current or predecessor run. If omitted, execute the current
        protocol; a failed certificate produces an executed ``failed`` row.

    Returns
    -------
    tuple[FrameworkEvidenceRow, ...]
        Observed native convergence and explicit inapplicable/unsupported routes.

    Raises
    ------
    ValueError
        If a supplied certificate belongs to another family or task protocol.

    """
    observed = run_qsnn_convergence_example() if certificate is None else certificate
    if observed.spec not in (
        qsnn_example_spec(),
        qsnn_example_spec(protocol="legacy_16_step"),
    ):
        raise ValueError("QSNN certificate must match a registered frozen protocol")
    passed = observed.passed
    rows = [
        FrameworkEvidenceRow(
            family=ModelFamily.QSNN,
            framework="scpn_qsnn_statevector",
            status=FrameworkStatus.RAN if passed else FrameworkStatus.FAILED,
            required=True,
            executed=True,
            passed=passed,
            reason=f"executed QSNN protocol {observed.spec.example_id}; convergence passed={passed}",
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
