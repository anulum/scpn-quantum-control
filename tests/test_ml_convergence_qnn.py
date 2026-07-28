# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 QNN convergence tests
"""Real trainer and installed-framework tests for the QNN example."""

from __future__ import annotations

import importlib.util

import pytest

import scpn_quantum_control.ml_examples.qnn_convergence as qnn_convergence
from scpn_quantum_control.ml_examples import (
    FrameworkStatus,
    qnn_example_spec,
    run_qnn_convergence_example,
    run_qnn_framework_rows,
)


def test_qnn_example_converges_and_replays_exactly() -> None:
    """Certify the frozen phase-separable task through the public trainer."""
    certificate = run_qnn_convergence_example()

    assert certificate.spec == qnn_example_spec()
    assert certificate.passed
    assert certificate.best_loss <= 1e-4
    assert certificate.loss_drop >= 2e-2
    assert certificate.metric_value == 1.0
    assert certificate.deterministic_replay
    assert dict(certificate.details)["gradient_method"] == (
        "multi_frequency_parameter_shift_qnn_classifier"
    )


def test_qnn_framework_rows_execute_installed_adapters_and_expose_absence() -> None:
    """Run every installed adapter and retain every unavailable matrix cell."""
    rows = {row.framework: row for row in run_qnn_framework_rows()}

    assert rows["scpn_parameter_shift"].gate_passed
    for framework, dependency in (
        ("jax", "jax"),
        ("pytorch", "torch"),
        ("tensorflow", "tensorflow"),
    ):
        row = rows[framework]
        if importlib.util.find_spec(dependency) is None:
            assert row.status is FrameworkStatus.UNAVAILABLE
            assert row.max_abs_error is None
        else:
            assert row.status is FrameworkStatus.RAN
            assert row.max_abs_error is not None
            assert row.max_abs_error <= 1e-6
        assert row.gate_passed
    assert rows["provider_hardware_gradient"].status is FrameworkStatus.UNSUPPORTED


def test_qnn_required_missing_framework_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turn a simulated absent TensorFlow dependency into a failed required gate."""
    actual_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        qnn_convergence.importlib.util,
        "find_spec",
        lambda dependency: None if dependency == "tensorflow" else actual_find_spec(dependency),
    )
    rows = {
        row.framework: row for row in run_qnn_framework_rows(required_frameworks=("tensorflow",))
    }

    assert rows["tensorflow"].required
    assert not rows["tensorflow"].gate_passed
    with pytest.raises(ValueError, match="unknown required"):
        run_qnn_framework_rows(required_frameworks=("missing",))
