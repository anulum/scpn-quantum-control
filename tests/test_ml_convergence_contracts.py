# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — BL-42 convergence contract tests
"""Validation and serialisation tests for BL-42 evidence contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from scpn_quantum_control.ml_examples.contracts import (
    ML_CONVERGENCE_CLAIM_BOUNDARY,
    ML_CONVERGENCE_SCHEMA,
    ConvergenceCertificate,
    ConvergenceExampleSpec,
    ConvergenceSuiteEvidence,
    FrameworkEvidenceRow,
    FrameworkStatus,
    ModelFamily,
)


def _spec(family: ModelFamily = ModelFamily.QNN) -> ConvergenceExampleSpec:
    return ConvergenceExampleSpec("example", family, 1, "synthetic task", 2, 0.1, 0.2)


def _certificate(family: ModelFamily = ModelFamily.QNN) -> ConvergenceCertificate:
    return ConvergenceCertificate(
        spec=_spec(family),
        loss_history=(0.5, 0.05),
        initial_loss=0.5,
        final_loss=0.05,
        best_loss=0.05,
        loss_drop=0.45,
        target_reached=True,
        loss_drop_reached=True,
        deterministic_replay=True,
        stop_reason="target",
        metric_name="accuracy",
        metric_value=1.0,
        metric_threshold=0.9,
        details=(("steps", 1),),
    )


def _row(family: ModelFamily = ModelFamily.QNN) -> FrameworkEvidenceRow:
    return FrameworkEvidenceRow(
        family=family,
        framework="native",
        status=FrameworkStatus.RAN,
        required=True,
        executed=True,
        passed=True,
        reason="executed",
        max_abs_error=0.0,
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(_spec(), example_id=""), "non-empty"),
        (lambda: replace(_spec(), task=""), "non-empty"),
        (lambda: replace(_spec(), backend=""), "non-empty"),
        (lambda: replace(_spec(), seed=True), "seed"),
        (lambda: replace(_spec(), seed=-1), "seed"),
        (lambda: replace(_spec(), max_steps=True), "max_steps"),
        (lambda: replace(_spec(), max_steps=0), "max_steps"),
        (lambda: replace(_spec(), target_loss=float("nan")), "target_loss"),
        (lambda: replace(_spec(), min_loss_drop=-1.0), "min_loss_drop"),
        (lambda: replace(_spec(), hardware=True), "simulator-only"),
    ],
)
def test_example_spec_rejects_invalid_values(factory: Callable[[], object], message: str) -> None:
    """Reject malformed or hardware-enabled frozen task specs."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_certificate_serialises_and_applies_optional_metric() -> None:
    """Expose exact convergence arithmetic and metric acceptance."""
    certificate = _certificate()
    payload = certificate.to_dict()

    assert certificate.passed
    assert certificate.metric_reached
    assert payload["details"] == {"steps": 1}
    assert payload["spec"] == certificate.spec.to_dict()
    assert payload["claim_boundary"] == ML_CONVERGENCE_CLAIM_BOUNDARY
    assert not replace(certificate, metric_value=0.8).passed
    no_metric = replace(
        certificate,
        metric_name=None,
        metric_value=None,
        metric_threshold=None,
    )
    assert no_metric.metric_reached


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(_certificate(), loss_history=()), "loss_history"),
        (lambda: replace(_certificate(), stop_reason=""), "stop_reason"),
        (lambda: replace(_certificate(), loss_history=(0.5, float("nan"))), "loss_history"),
        (lambda: replace(_certificate(), loss_history=(0.5, -0.1)), "loss_history"),
        (lambda: replace(_certificate(), initial_loss=0.4), "initial_loss"),
        (lambda: replace(_certificate(), final_loss=0.1), "final_loss"),
        (lambda: replace(_certificate(), best_loss=0.1), "best_loss"),
        (lambda: replace(_certificate(), loss_drop=0.4), "loss_drop"),
        (lambda: replace(_certificate(), target_reached=False), "target_reached"),
        (lambda: replace(_certificate(), loss_drop_reached=False), "loss_drop_reached"),
        (lambda: replace(_certificate(), metric_name=None), "provided together"),
        (lambda: replace(_certificate(), metric_name=""), "metric_name"),
        (lambda: replace(_certificate(), metric_value=float("inf")), "metric_value"),
        (lambda: replace(_certificate(), metric_threshold=float("nan")), "metric_threshold"),
        (lambda: replace(_certificate(), details=(("x", 1), ("x", 2))), "unique"),
        (lambda: replace(_certificate(), details=(("", 1),)), "non-empty"),
    ],
)
def test_certificate_rejects_arithmetic_or_schema_drift(
    factory: Callable[[], object], message: str
) -> None:
    """Reject certificates whose booleans or summaries disagree with the curve."""
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: replace(_row(), framework=""), "non-empty"),
        (lambda: replace(_row(), reason=""), "non-empty"),
        (lambda: replace(_row(), executed=False), "require"),
        (lambda: replace(_row(), passed=None), "require"),
        (
            lambda: FrameworkEvidenceRow(
                ModelFamily.QNN,
                "optional",
                FrameworkStatus.UNAVAILABLE,
                False,
                True,
                None,
                "missing",
            ),
            "cannot carry",
        ),
        (lambda: replace(_row(), passed=False), "must pass"),
        (
            lambda: FrameworkEvidenceRow(
                ModelFamily.QNN,
                "failed",
                FrameworkStatus.FAILED,
                False,
                True,
                True,
                "failed",
            ),
            "cannot pass",
        ),
        (lambda: replace(_row(), max_abs_error=-1.0), "max_abs_error"),
    ],
)
def test_framework_row_rejects_inconsistent_status(
    factory: Callable[[], object], message: str
) -> None:
    """Reject blank or semantically inconsistent framework matrix rows."""
    with pytest.raises(ValueError, match=message):
        factory()


def test_framework_row_gate_semantics_cover_optional_and_required_failures() -> None:
    """Keep optional absence visible while required absence fails closed."""
    optional = FrameworkEvidenceRow(
        ModelFamily.QNN,
        "tensorflow",
        FrameworkStatus.UNAVAILABLE,
        False,
        False,
        None,
        "not installed",
    )
    required = replace(optional, required=True)
    failed = FrameworkEvidenceRow(
        ModelFamily.QNN,
        "jax",
        FrameworkStatus.FAILED,
        False,
        True,
        False,
        "runtime failure",
    )

    assert optional.gate_passed
    assert not required.gate_passed
    assert not failed.gate_passed
    assert optional.to_dict()["status"] == "unavailable"


def test_suite_requires_complete_family_rows_and_pointers() -> None:
    """Require exact three-family coverage and fixed claim boundaries."""
    certificates = tuple(_certificate(family) for family in ModelFamily)
    rows = tuple(_row(family) for family in ModelFamily)
    pointers = tuple((family, f"{family.value} pointer") for family in ModelFamily)
    suite = ConvergenceSuiteEvidence(certificates, rows, pointers)

    assert suite.passed
    assert suite.to_payload()["schema"] == ML_CONVERGENCE_SCHEMA
    assert not replace(
        suite,
        certificates=(replace(certificates[0], deterministic_replay=False), *certificates[1:]),
    ).passed
    assert not replace(
        suite,
        framework_rows=(
            replace(rows[0], required=False, status=FrameworkStatus.FAILED, passed=False),
            *rows[1:],
        ),
    ).passed
    with pytest.raises(ValueError, match="one certificate"):
        replace(suite, certificates=certificates[:2])
    with pytest.raises(ValueError, match="repeat"):
        replace(suite, certificates=(certificates[0], certificates[0], certificates[2]))
    with pytest.raises(ValueError, match="framework matrix"):
        replace(suite, framework_rows=rows[:2])
    with pytest.raises(ValueError, match="notebook pointers"):
        replace(suite, notebook_pointers=pointers[:2])
    with pytest.raises(ValueError, match="pointer values"):
        replace(suite, notebook_pointers=(pointers[0], pointers[1], (ModelFamily.QSNN, "")))
    with pytest.raises(ValueError, match="schema"):
        replace(suite, schema="v2")
    with pytest.raises(ValueError, match="schema"):
        replace(suite, claim_boundary="expanded")
    with pytest.raises(ValueError, match="provider"):
        replace(suite, provider_execution=True)
