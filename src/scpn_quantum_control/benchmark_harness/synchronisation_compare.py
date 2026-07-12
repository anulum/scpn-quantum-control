# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation compare module
# scpn-quantum-control -- synchronisation benchmark comparator
"""Tolerance comparator for synchronisation benchmark result artefacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmark_harness.synchronisation import RESULT_SCHEMA


@dataclass(frozen=True, slots=True)
class ObservableComparison:
    """One observable comparison result."""

    backend: str
    name: str
    expected: float
    actual: float
    absolute_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable comparison row."""
        return asdict(self)


REQUIRED_ROW_FIELDS = set(RESULT_SCHEMA["required_fields"])
REQUIRED_OBSERVABLE_FIELDS = set(RESULT_SCHEMA["observable_row_fields"])

DEFAULT_BENCHMARK_ARTIFACTS: tuple[str, ...] = (
    "data/synchronisation_benchmarks/kuramoto_ring_n4_linear_omega_reference_rows.json",
    "data/synchronisation_benchmarks/kuramoto_chain_n8_decay_omega_reference_rows.json",
)


def load_payload(path: Path) -> dict[str, Any]:
    """Load a synchronisation benchmark result payload."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def validate_payload_shape(payload: dict[str, Any]) -> None:
    """Validate the synchronisation benchmark result schema shape."""
    if payload.get("schema") != "synchronisation_benchmark_run_v1":
        raise ValueError("unexpected synchronisation benchmark run schema")
    if payload.get("result_schema") != RESULT_SCHEMA:
        raise ValueError("result_schema drift detected")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("payload must contain at least one result row")
    for index, row in enumerate(rows):
        missing = REQUIRED_ROW_FIELDS - set(row)
        if missing:
            raise ValueError(f"row {index} missing required fields: {sorted(missing)}")
        if row.get("hardware_submission") is not False:
            raise ValueError(f"row {index} must be no-QPU hardware_submission=false")
        observables = row.get("observables")
        if not isinstance(observables, list) or not observables:
            raise ValueError(f"row {index} must contain observables")
        for obs_index, observable in enumerate(observables):
            missing_obs = REQUIRED_OBSERVABLE_FIELDS - set(observable)
            if missing_obs:
                raise ValueError(
                    f"row {index} observable {obs_index} missing fields: {sorted(missing_obs)}"
                )


def observable_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index payload observables by backend and name."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        backend = str(row["backend"])
        for observable in row["observables"]:
            key = (backend, str(observable["name"]))
            if key in out:
                raise ValueError(f"duplicate observable row: {key}")
            out[key] = observable
    return out


def compare_payloads(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Compare actual synchronisation rows against expected committed rows."""
    validate_payload_shape(expected)
    validate_payload_shape(actual)
    if actual.get("benchmark_id") != expected.get("benchmark_id"):
        raise ValueError(
            f"benchmark_id mismatch: {actual.get('benchmark_id')} != {expected.get('benchmark_id')}"
        )
    expected_rows = observable_index(expected)
    actual_rows = observable_index(actual)
    missing = sorted(set(expected_rows) - set(actual_rows))
    extra = sorted(set(actual_rows) - set(expected_rows))
    comparisons: list[ObservableComparison] = []
    for key in sorted(set(expected_rows).intersection(actual_rows)):
        expected_obs = expected_rows[key]
        actual_obs = actual_rows[key]
        tolerance = float(expected_obs["tolerance"])
        expected_value = float(expected_obs["value"])
        actual_value = float(actual_obs["value"])
        absolute_error = abs(actual_value - expected_value)
        comparisons.append(
            ObservableComparison(
                backend=key[0],
                name=key[1],
                expected=expected_value,
                actual=actual_value,
                absolute_error=absolute_error,
                tolerance=tolerance,
                passed=absolute_error <= tolerance and bool(actual_obs["passed"]),
            )
        )
    blockers = [f"missing observable row: {backend}/{name}" for backend, name in missing] + [
        f"unexpected observable row: {backend}/{name}" for backend, name in extra
    ]
    blockers.extend(
        f"{row.backend}/{row.name}: error {row.absolute_error} exceeds tolerance {row.tolerance}"
        for row in comparisons
        if not row.passed
    )
    return {
        "schema": "synchronisation_benchmark_comparison_v1",
        "benchmark_id": expected.get("benchmark_id"),
        "valid": not blockers,
        "blockers": blockers,
        "comparisons": [row.to_dict() for row in comparisons],
    }


def compare_files(expected_path: Path, actual_path: Path) -> dict[str, Any]:
    """Compare two synchronisation benchmark JSON artefacts."""
    return compare_payloads(load_payload(expected_path), load_payload(actual_path))


def compare_default_artifacts(repo_root: Path) -> dict[str, Any]:
    """Compare every committed synchronisation benchmark artefact to itself.

    This is a release gate for schema and tolerance stability. Regeneration
    commands should run before this comparator when checking drift.
    """
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for rel_path in DEFAULT_BENCHMARK_ARTIFACTS:
        path = repo_root / rel_path
        if not path.exists():
            blockers.append(f"missing synchronisation benchmark artefact: {rel_path}")
            continue
        result = compare_files(path, path)
        result["path"] = rel_path
        results.append(result)
        blockers.extend(f"{rel_path}: {blocker}" for blocker in result["blockers"])
    return {
        "schema": "synchronisation_benchmark_multi_comparison_v1",
        "valid": not blockers,
        "blockers": blockers,
        "artifact_count": len(results),
        "results": results,
    }
