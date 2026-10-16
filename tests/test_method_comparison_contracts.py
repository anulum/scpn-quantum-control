# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Explicit method-comparison input boundary tests
"""Immutable public scientific meanings, budget and estimator association tests."""

from dataclasses import replace
from typing import Any

import pytest

from scpn_quantum_control.analysis.sync_uncertainty import order_parameter_shot_noise
from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)


@pytest.fixture
def meaning() -> ComparisonMeaning:
    """Declare a deterministic analytic scalar with no inferred measurement policy."""
    return ComparisonMeaning(
        "problem",
        "observable",
        "1",
        "model",
        "float64",
        None,
        "none",
        "fixed",
        "none",
        "none",
        "exact",
    )


@pytest.fixture
def repetition(meaning: ComparisonMeaning) -> MethodRun:
    """One immutable completed repetition with measured full resource accounting."""
    return MethodRun(
        "repeat",
        "method",
        "native",
        0,
        meaning,
        ComparisonBudget(2.0, 3.0, 5.0, 128, 256, 10.0),
        "completed",
        -0.25,
    )


def test_budget_retains_unknown_and_advisory_measurements() -> None:
    """An advisory timer cannot fill unknown compilation or data-movement cost."""
    original = ComparisonBudget(None, None, None, None, None, None, 25.0)
    assert original.total_cost_ms is None
    assert original.within_budget is False
    assert original.to_dict()["observed_wall_ms"] == 25.0
    assert original.to_dict()["compilation_ms"] is None
    assert ComparisonBudget(0.0, 0.0, 0.0, 0, 0, 0.0).within_budget is True
    assert ComparisonBudget(1e308, 1e308, 0.0, 0, 0, 1e308).total_cost_ms is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("compilation_ms", -1.0),
        ("transfer_ms", float("inf")),
        ("execution_ms", float("nan")),
        ("wall_limit_ms", True),
        ("observed_wall_ms", "25"),
        pytest.param("compilation_ms", 10**1000, id="cost-exceeds-binary64"),
        ("peak_bytes", 1.5),
        ("memory_limit_bytes", -1),
        ("peak_bytes", False),
    ],
)
def test_budget_refuses_malformed_units_and_types(field: str, value: object) -> None:
    """Malformed resource records never become a finite or zero-filled budget."""
    with pytest.raises(ValueError, match=field):
        # Deliberately cross the typed constructor with malformed runtime data.
        changes: dict[str, Any] = {field: value}
        replace(ComparisonBudget(0.0, 0.0, 0.0, 0, 0, 0.0), **changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", ""),
        ("model", " "),
        ("precision", 64),
        pytest.param("unit", "x" * 2049, id="unit-too-long"),
        ("shots", True),
        ("shots", -1),
        ("shots", 1.5),
    ],
)
def test_meaning_refuses_incomplete_or_coerced_semantics(
    meaning: ComparisonMeaning,
    field: str,
    value: object,
) -> None:
    """No missing scientific meaning or float-to-integer shot coercion is admitted."""
    with pytest.raises(ValueError, match=field):
        # Deliberately cross the typed constructor with malformed runtime data.
        changes: dict[str, Any] = {field: value}
        replace(meaning, **changes)


def test_meaning_retains_exact_large_integer_shots(meaning: ComparisonMeaning) -> None:
    """Integral counts survive above the JavaScript safe-integer boundary."""
    exact = 2**53 + 1
    assert replace(meaning, shots=exact).shots == exact


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", ""),
        ("method", None),
        ("backend", " "),
        ("seed", -1),
        ("seed", None),
        ("seed", 1.0),
        ("status", "cancelled"),
        ("value", float("nan")),
        ("value", float("inf")),
        pytest.param("value", 10**1000, id="value-exceeds-binary64"),
        ("value", True),
        ("value", None),
        ("reason", "contradictory failure"),
    ],
)
def test_repetition_refuses_invalid_original_associations(
    repetition: MethodRun,
    field: str,
    value: object,
) -> None:
    """Completed rows require finite values, honest availability and exact identity."""
    with pytest.raises(ValueError):
        # Deliberately cross the typed constructor with malformed runtime data.
        changes: dict[str, Any] = {field: value}
        replace(repetition, **changes)


def test_failed_and_missing_repetitions_require_retained_reason(repetition: MethodRun) -> None:
    """Failure evidence is explicit, immutable and cannot carry a completed value."""
    for status in ("failed", "missing"):
        missing = replace(repetition, status=status, value=None, reason="original unavailable")
        assert missing.value is None
        assert missing.reason == "original unavailable"
        with pytest.raises(ValueError, match="unavailable"):
            replace(missing, value=1.0)
        with pytest.raises(ValueError, match="reason"):
            replace(missing, reason=None)


def test_existing_estimator_interval_requires_exact_original_association(
    repetition: MethodRun,
) -> None:
    """Real analysis-owner intervals stay attached to their point and shot count."""
    interval = order_parameter_shot_noise({"0": 75, "1": 25})
    original = replace(
        repetition,
        value=interval.point,
        interval=interval,
        meaning=replace(repetition.meaning, shots=100, estimator=interval.method),
    )
    assert original.interval is interval
    for changed in [
        replace(interval, point=0.9),
        replace(interval, n_shots=99),
        replace(interval, method="unknown"),
        replace(interval, coverage=1.0),
        replace(interval, low=float("nan")),
        replace(interval, low=2.0, high=1.0),
        replace(interval, standard_error=-1.0),
        replace(interval, n_resamples=-1),
    ]:
        with pytest.raises(ValueError, match="interval"):
            replace(original, interval=changed)
    with pytest.raises(ValueError, match="unavailable"):
        replace(original, status="failed", value=None, reason="failed")


def test_interval_estimator_label_and_sampling_protocol_must_match(repetition: MethodRun) -> None:
    """A source interval cannot be relabelled or use impossible analytic/bootstrap counts."""
    from scpn_quantum_control.analysis.sync_uncertainty import order_parameter_bootstrap

    interval = order_parameter_shot_noise({"0": 75, "1": 25})
    original = replace(
        repetition,
        value=interval.point,
        interval=interval,
        meaning=replace(repetition.meaning, shots=100, estimator=interval.method),
    )
    with pytest.raises(ValueError, match="interval"):
        replace(original, meaning=replace(original.meaning, estimator="other estimator"))
    with pytest.raises(ValueError, match="interval"):
        replace(original, interval=replace(interval, n_resamples=1))
    with pytest.raises(ValueError, match="interval"):
        replace(
            original,
            interval=replace(interval, n_shots=1),
            meaning=replace(original.meaning, shots=1),
        )
    bootstrap = order_parameter_bootstrap({"0": 75, "1": 25}, n_resamples=32)
    valid = replace(
        original, interval=bootstrap, meaning=replace(original.meaning, estimator=bootstrap.method)
    )
    assert valid.interval is bootstrap
    with pytest.raises(ValueError, match="interval"):
        replace(valid, interval=replace(bootstrap, n_resamples=0))
