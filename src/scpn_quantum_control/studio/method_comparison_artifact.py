# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Source-bound native method comparison export
"""Adapt native retained-population comparisons into lossless Studio documents.

The numerical owner never imports Studio. Input summaries are not accepted:
this adapter calls the public native owner over every declared repetition.
Canonical digests bind exact content, not producer authenticity or execution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, cast

from ..analysis.sync_uncertainty import UncertaintyInterval
from ..benchmarks.method_comparison import (
    METHOD_COMPARISON_SCHEMA,
    summarise_method_comparison,
)
from ..benchmarks.method_comparison_contracts import (
    MAX_COMPARISON_RUNS,
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)
from ..studio_workspace.canonical import canonical_digest
from ..studio_workspace.json_transport import read_json, write_json

METHOD_COMPARISON_INPUT_SCHEMA = "studio.method_comparison_input.v1"
"""Versioned local input containing every original repetition, never a summary."""
MAX_COMPARISON_BYTES = 4 * 1024 * 1024
"""Bounded native input and Studio export size in UTF-8 bytes."""


def _record(value: object, keys: set[str], path: str) -> dict[str, object]:
    """Require an exact object shape before passing values to their native owners."""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{path}: exact original fields required")
    return cast(dict[str, object], value)


def _text(value: object, path: str) -> str:
    """Read an explicitly bounded original string without coercion."""
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{path}: nonempty bounded text required")
    return value


def _number(value: object, path: str) -> float:
    """Read a finite scalar while preserving original integer/float token identity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: finite scalar required")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{path}: finite scalar required")
    return value


def _count(value: object, path: str) -> int:
    """Read an exact nonnegative integral resource or seed without coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}: nonnegative integer required")
    return value


def _meaning(value: object) -> ComparisonMeaning:
    """Bind the complete original scientific declaration through its native owner."""
    row = _record(
        value,
        {
            "problem",
            "observable",
            "unit",
            "model",
            "precision",
            "shots",
            "shot_protocol",
            "initial_condition",
            "calibration",
            "noise",
            "estimator",
        },
        "meaning",
    )
    return ComparisonMeaning(
        problem=_text(row["problem"], "problem"),
        observable=_text(row["observable"], "observable"),
        unit=_text(row["unit"], "unit"),
        model=_text(row["model"], "model"),
        precision=_text(row["precision"], "precision"),
        shots=None if row["shots"] is None else _count(row["shots"], "shots"),
        shot_protocol=_text(row["shot_protocol"], "shot_protocol"),
        initial_condition=_text(row["initial_condition"], "initial_condition"),
        calibration=_text(row["calibration"], "calibration"),
        noise=_text(row["noise"], "noise"),
        estimator=_text(row["estimator"], "estimator"),
    )


def _budget(value: object) -> ComparisonBudget:
    """Retain unknown components and exact byte ceilings without a cost fallback."""
    row = _record(
        value,
        {
            "compilation_ms",
            "transfer_ms",
            "execution_ms",
            "peak_bytes",
            "memory_limit_bytes",
            "wall_limit_ms",
            "observed_wall_ms",
        },
        "budget",
    )
    return ComparisonBudget(
        compilation_ms=None
        if row["compilation_ms"] is None
        else _number(row["compilation_ms"], "compilation_ms"),
        transfer_ms=None
        if row["transfer_ms"] is None
        else _number(row["transfer_ms"], "transfer_ms"),
        execution_ms=None
        if row["execution_ms"] is None
        else _number(row["execution_ms"], "execution_ms"),
        peak_bytes=None if row["peak_bytes"] is None else _count(row["peak_bytes"], "peak_bytes"),
        memory_limit_bytes=None
        if row["memory_limit_bytes"] is None
        else _count(row["memory_limit_bytes"], "memory_limit_bytes"),
        wall_limit_ms=None
        if row["wall_limit_ms"] is None
        else _number(row["wall_limit_ms"], "wall_limit_ms"),
        observed_wall_ms=None
        if row["observed_wall_ms"] is None
        else _number(row["observed_wall_ms"], "observed_wall_ms"),
    )


def _interval(value: object) -> UncertaintyInterval | None:
    """Read an existing estimator's exact interval; the native run validates association."""
    if value is None:
        return None
    row = _record(
        value,
        {
            "point",
            "standard_error",
            "low",
            "high",
            "coverage",
            "method",
            "n_shots",
            "n_resamples",
        },
        "interval",
    )
    return UncertaintyInterval(
        point=_number(row["point"], "interval.point"),
        standard_error=_number(row["standard_error"], "interval.standard_error"),
        low=_number(row["low"], "interval.low"),
        high=_number(row["high"], "interval.high"),
        coverage=_number(row["coverage"], "interval.coverage"),
        method=_text(row["method"], "interval.method"),
        n_shots=_count(row["n_shots"], "interval.n_shots"),
        n_resamples=_count(row["n_resamples"], "interval.n_resamples"),
    )


def _run(value: object) -> MethodRun:
    """Admit one whole original row without inventing missing evidence."""
    row = _record(
        value,
        {
            "run_id",
            "method",
            "backend",
            "seed",
            "meaning",
            "budget",
            "status",
            "value",
            "reason",
            "interval",
        },
        "run",
    )
    return MethodRun(
        run_id=_text(row["run_id"], "run_id"),
        method=_text(row["method"], "method"),
        backend=_text(row["backend"], "backend"),
        seed=_count(row["seed"], "seed"),
        meaning=_meaning(row["meaning"]),
        budget=_budget(row["budget"]),
        status=cast(Literal["completed", "failed", "missing"], _text(row["status"], "status")),
        value=None if row["value"] is None else _number(row["value"], "value"),
        reason=None if row["reason"] is None else _text(row["reason"], "reason"),
        interval=_interval(row["interval"]),
    )


def read_method_comparison_population(
    text: str,
) -> tuple[tuple[MethodRun, ...], str, Literal["require-complete", "available-case"], str | None]:
    """Read a complete native input population with strict version/field admission.

    Parameters
    ----------
    text : str
        Lossless local input JSON, at most four mebibytes, containing schema,
        runs, baseline_method, missing_data_policy and selected_run_id.

    Returns
    -------
    tuple
        Original immutable runs, reference method, declared missing-data policy
        and display-only selection. The public native owner validates identity,
        population/resource limits and policy before this returns.

    Raises
    ------
    ValueError
        Unsupported fields/version, malformed original evidence, nonfinite or
        duplicate identities, resource bounds or declared reference fails.

    """
    if len(text.encode("utf-8")) > MAX_COMPARISON_BYTES:
        raise ValueError("method comparison input exceeds four mebibytes")
    unit = _record(
        read_json(text),
        {
            "schema",
            "runs",
            "baseline_method",
            "missing_data_policy",
            "selected_run_id",
        },
        "input",
    )
    if unit["schema"] != METHOD_COMPARISON_INPUT_SCHEMA:
        raise ValueError("unsupported method comparison input schema")
    original = unit["runs"]
    if not isinstance(original, list) or not 1 <= len(original) <= MAX_COMPARISON_RUNS:
        raise ValueError("bounded nonempty original repetition list required")
    runs = tuple(_run(value) for value in original)
    baseline = _text(unit["baseline_method"], "baseline_method")
    policy = cast(
        Literal["require-complete", "available-case"],
        _text(unit["missing_data_policy"], "missing_data_policy"),
    )
    selected = (
        None
        if unit["selected_run_id"] is None
        else _text(unit["selected_run_id"], "selected_run_id")
    )
    summarise_method_comparison(
        runs, baseline_method=baseline, missing_data_policy=policy, selected_run_id=selected
    )
    return runs, baseline, policy, selected


def export_method_comparison(
    runs: Sequence[MethodRun],
    *,
    baseline_method: str,
    missing_data_policy: Literal["require-complete", "available-case"] = "require-complete",
    selected_run_id: str | None = None,
) -> str:
    """Export the full native population and source-owned summary without writes.

    Parameters
    ----------
    runs : sequence of MethodRun
        Every declared original repetition, including missing and failed rows.
    baseline_method : str
        Explicit reference identity retained by the native comparison owner.
    missing_data_policy : {"require-complete", "available-case"}
        Original denominator policy, never inferred by the view.
    selected_run_id : str or None
        Display-only selected repetition; aggregate always uses all declared runs.

    Returns
    -------
    str
        Lossless schema/body/extensions/sha256 document, bounded to four
        mebibytes. Native comparison retains all input values and intervals.
        Digest proves content identity, not authenticity or solver execution.

    Raises
    ------
    ValueError
        Native admission, canonical representation or output byte budget fails.

    """
    comparison = summarise_method_comparison(
        runs,
        baseline_method=baseline_method,
        missing_data_policy=missing_data_policy,
        selected_run_id=selected_run_id,
    )
    unit: dict[str, object] = {
        "schema": METHOD_COMPARISON_SCHEMA,
        "body": {
            "producer": "scpn_quantum_control.benchmarks.method_comparison.summarise_method_comparison",
            "comparison": comparison,
            "no_submit": True,
        },
        "extensions": {},
    }
    unit["sha256"] = canonical_digest(METHOD_COMPARISON_SCHEMA, unit)
    text = write_json(unit)
    if len(text.encode("utf-8")) > MAX_COMPARISON_BYTES:
        raise ValueError("method comparison export exceeds four mebibytes")
    return text
