# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example convergence contract tests
"""Validation and serialisation tests for convergence-example evidence contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from scpn_quantum_control.ml_examples import (
    qsnn_framework_rows,
    run_ml_convergence_suite,
    run_qsnn_convergence_example,
)
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


def test_real_training_evidence_refuses_a_textual_replay_flag() -> None:
    """A truthy string cannot replace an observed Boolean on real training evidence."""
    result = run_qsnn_convergence_example(protocol="legacy_16_step")
    with pytest.raises(ValueError, match="deterministic_replay.*boolean"):
        replace(result, deterministic_replay=cast(bool, "false"))


@pytest.fixture(scope="module")
def observed_certificate() -> ConvergenceCertificate:
    """Train the real retained QSNN reference for corruption checks.

    Returns
    -------
    ConvergenceCertificate
        Actual sixteen-step simulator result, including its failed loss gate.

    """
    return run_qsnn_convergence_example(protocol="legacy_16_step")


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("target_reached", lambda c, v: replace(c, target_reached=v)),
        ("loss_drop_reached", lambda c, v: replace(c, loss_drop_reached=v)),
        ("deterministic_replay", lambda c, v: replace(c, deterministic_replay=v)),
    ],
)
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_observed_certificate_refuses_non_boolean_verdicts(
    observed_certificate: ConvergenceCertificate,
    field: str,
    mutate: Callable[[ConvergenceCertificate, bool], ConvergenceCertificate],
    value: object,
) -> None:
    """Reject truth-value coercion on a real training certificate.

    Parameters
    ----------
    observed_certificate
        Real public-trainer result before evidence corruption.
    field
        Acceptance flag being corrupted.
    mutate
        Typed constructor call for the flag under test.
    value
        Non-Boolean representation supplied by a caller.

    """
    with pytest.raises(ValueError, match=field + ".*boolean"):
        mutate(observed_certificate, cast(bool, value))


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_observed_spec_refuses_non_boolean_hardware(
    observed_certificate: ConvergenceCertificate, value: object
) -> None:
    """Keep the real simulator task's hardware marker strictly Boolean.

    Parameters
    ----------
    observed_certificate
        Real public-trainer result supplying the frozen task.
    value
        Non-Boolean execution marker.

    """
    with pytest.raises(ValueError, match="hardware.*boolean"):
        replace(observed_certificate.spec, hardware=cast(bool, value))


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("required", lambda r, v: replace(r, required=v)),
        ("executed", lambda r, v: replace(r, executed=v)),
        ("passed", lambda r, v: replace(r, passed=v)),
    ],
)
@pytest.mark.parametrize("value", ["false", 0, 1])
def test_observed_framework_row_refuses_non_boolean_flags(
    observed_certificate: ConvergenceCertificate,
    field: str,
    mutate: Callable[[FrameworkEvidenceRow, bool], FrameworkEvidenceRow],
    value: object,
) -> None:
    """Reject coerced framework verdicts derived from actual failed training.

    Parameters
    ----------
    observed_certificate
        Actual failed predecessor used by the public framework matrix.
    field
        Framework execution or acceptance flag being corrupted.
    mutate
        Typed constructor call for the flag under test.
    value
        Non-Boolean representation supplied by a caller.

    """
    row = qsnn_framework_rows(certificate=observed_certificate)[0]
    assert row.status is FrameworkStatus.FAILED
    assert row.passed is False
    with pytest.raises(ValueError, match=field + ".*boolean"):
        mutate(row, cast(bool, value))


def test_observed_evidence_refuses_untyped_family_and_status() -> None:
    """Reject string enum substitutes before accepting real evidence records."""
    certificate = run_qsnn_convergence_example(protocol="legacy_16_step")
    row = qsnn_framework_rows(certificate=certificate)[0]
    with pytest.raises(ValueError, match="family.*ModelFamily"):
        replace(certificate.spec, family=cast(ModelFamily, "qsnn"))
    with pytest.raises(ValueError, match="family.*ModelFamily"):
        replace(row, family=cast(ModelFamily, "qsnn"))
    with pytest.raises(ValueError, match="status.*FrameworkStatus"):
        replace(row, status=cast(FrameworkStatus, "failed"))


def test_observed_suite_refuses_non_boolean_execution_flags() -> None:
    """Reject coerced provider and hardware markers on the real public suite."""
    suite = run_ml_convergence_suite()
    assert suite.passed
    for value in ("false", 0, 1, None):
        with pytest.raises(ValueError, match="provider_execution.*boolean"):
            replace(suite, provider_execution=cast(bool, value))
        with pytest.raises(ValueError, match="hardware_execution.*boolean"):
            replace(suite, hardware_execution=cast(bool, value))
