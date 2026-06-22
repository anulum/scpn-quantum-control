# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the synchronisation benchmark comparator
"""Guard and branch tests for the synchronisation benchmark comparator.

Covers the payload schema, row-field, observable and duplicate guards, the
benchmark-id mismatch guard and the missing-artefact blocker branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.benchmark_harness.synchronisation import RESULT_SCHEMA
from scpn_quantum_control.benchmark_harness.synchronisation_compare import (
    compare_default_artifacts,
    compare_payloads,
    observable_index,
    validate_payload_shape,
)


def _observable(name: str = "sync_order") -> dict[str, Any]:
    return {
        "name": name,
        "value": 0.5,
        "uncertainty": 0.01,
        "units": "dimensionless",
        "tolerance": 0.1,
        "passed": True,
    }


def _row(backend: str = "ref", observables: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "benchmark_id": "bench-1",
        "backend": backend,
        "backend_version": "1.0",
        "command": "scpn-bench",
        "commit": "abc",
        "dependency_lock": "lock",
        "hardware_submission": False,
        "wall_time_s": 1.0,
        "observables": _observable() if observables is None else observables,
        "claim_boundary": "no-QPU",
    }


def _payload(benchmark_id: str = "bench-1") -> dict[str, Any]:
    return {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "benchmark_id": benchmark_id,
        "rows": [_row(observables=[_observable()])],
    }


def test_validate_rejects_unexpected_schema() -> None:
    """An unexpected run schema is rejected."""
    with pytest.raises(ValueError, match="unexpected synchronisation benchmark run schema"):
        validate_payload_shape({"schema": "v0"})


def test_validate_rejects_empty_rows() -> None:
    """A payload with no result rows is rejected."""
    payload = {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "rows": [],
    }
    with pytest.raises(ValueError, match="payload must contain at least one result row"):
        validate_payload_shape(payload)


def test_validate_rejects_missing_row_fields() -> None:
    """A row missing required fields is rejected."""
    payload = {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "rows": [{"backend": "ref"}],
    }
    with pytest.raises(ValueError, match="missing required fields"):
        validate_payload_shape(payload)


def test_validate_rejects_empty_observables() -> None:
    """A row without observables is rejected."""
    payload = {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "rows": [_row(observables=[])],
    }
    with pytest.raises(ValueError, match="must contain observables"):
        validate_payload_shape(payload)


def test_validate_rejects_observable_missing_fields() -> None:
    """An observable missing required fields is rejected."""
    payload = {
        "schema": "synchronisation_benchmark_run_v1",
        "result_schema": RESULT_SCHEMA,
        "rows": [_row(observables=[{"name": "sync_order"}])],
    }
    with pytest.raises(ValueError, match="missing fields"):
        validate_payload_shape(payload)


def test_observable_index_rejects_duplicates() -> None:
    """A duplicate backend/observable pair is rejected."""
    payload = {
        "rows": [
            _row(backend="ref", observables=[_observable("x")]),
            _row(backend="ref", observables=[_observable("x")]),
        ]
    }
    with pytest.raises(ValueError, match="duplicate observable row"):
        observable_index(payload)


def test_compare_payloads_rejects_benchmark_id_mismatch() -> None:
    """Comparing payloads with mismatched benchmark ids is rejected."""
    with pytest.raises(ValueError, match="benchmark_id mismatch"):
        compare_payloads(_payload("bench-A"), _payload("bench-B"))


def test_compare_default_artifacts_flags_missing(tmp_path: Path) -> None:
    """A repository root with no committed artefacts reports missing-artefact blockers."""
    report = compare_default_artifacts(tmp_path)
    assert report["valid"] is False
    assert any("missing synchronisation benchmark artefact" in b for b in report["blockers"])
