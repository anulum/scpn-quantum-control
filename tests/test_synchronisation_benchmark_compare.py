# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation benchmark compare tests
# scpn-quantum-control -- synchronisation benchmark comparator tests
"""Tests for synchronisation benchmark tolerance comparison."""

from __future__ import annotations

import copy

import pytest

from scpn_quantum_control.benchmark_harness.synchronisation_compare import compare_payloads
from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
    run_kuramoto_ring_n4_linear_omega,
)


def _payload() -> dict:
    return run_kuramoto_ring_n4_linear_omega(command="test", commit="abc123")


def test_compare_payloads_accepts_identical_reference() -> None:
    """Identical committed/regenerated rows pass."""

    payload = _payload()

    result = compare_payloads(payload, copy.deepcopy(payload))

    assert result["valid"] is True
    assert result["blockers"] == []
    assert result["comparisons"]


def test_compare_payloads_rejects_tolerance_drift() -> None:
    """Observable drift beyond tolerance fails closed."""

    expected = _payload()
    actual = copy.deepcopy(expected)
    actual["rows"][0]["observables"][0]["value"] += 0.1

    result = compare_payloads(expected, actual)

    assert result["valid"] is False
    assert any("exceeds tolerance" in blocker for blocker in result["blockers"])


def test_compare_payloads_rejects_schema_drift() -> None:
    """Schema drift is a hard error, not a soft comparison failure."""

    expected = _payload()
    actual = copy.deepcopy(expected)
    actual["result_schema"] = {"schema": "different"}

    with pytest.raises(ValueError, match="result_schema drift"):
        compare_payloads(expected, actual)


def test_compare_payloads_rejects_hardware_submission_rows() -> None:
    """The synchronisation gate remains no-QPU by default."""

    expected = _payload()
    actual = copy.deepcopy(expected)
    actual["rows"][0]["hardware_submission"] = True

    with pytest.raises(ValueError, match="hardware_submission=false"):
        compare_payloads(expected, actual)


def test_compare_default_artifacts_checks_all_committed_paths(tmp_path) -> None:
    """The multi-instance gate reports every configured benchmark artefact."""

    import json

    from scpn_quantum_control.benchmark_harness import synchronisation_compare
    from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
        CHAIN_N8_BENCHMARK_ID,
        RING_N4_BENCHMARK_ID,
        run_kuramoto_chain_n8_decay_omega,
        run_kuramoto_ring_n4_linear_omega,
    )

    ring = tmp_path / "ring.json"
    chain = tmp_path / "chain.json"
    ring.write_text(
        json.dumps(run_kuramoto_ring_n4_linear_omega(command="test", commit="abc123")),
        encoding="utf-8",
    )
    chain.write_text(
        json.dumps(run_kuramoto_chain_n8_decay_omega(command="test", commit="abc123")),
        encoding="utf-8",
    )
    synchronisation_compare.DEFAULT_BENCHMARK_ARTIFACTS = (
        ring.name,
        chain.name,
    )

    result = synchronisation_compare.compare_default_artifacts(tmp_path)

    assert result["valid"] is True
    assert result["artifact_count"] == 2
    assert {item["benchmark_id"] for item in result["results"]} == {
        RING_N4_BENCHMARK_ID,
        CHAIN_N8_BENCHMARK_ID,
    }
