# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- synchronisation benchmark registry tests
"""Tests for the synchronisation benchmark registry contract."""

from __future__ import annotations

from scpn_quantum_control.benchmark_harness.synchronisation import (
    RESULT_SCHEMA,
    list_synchronisation_benchmarks,
    synchronisation_benchmark_registry_payload,
)


def test_registry_contains_required_canonical_instances() -> None:
    """The registry exposes stable starter instances and replay rows."""

    rows = list_synchronisation_benchmarks()
    ids = {row.benchmark_id for row in rows}

    assert "kuramoto_ring_n4_linear_omega" in ids
    assert "kuramoto_chain_n8_decay_omega" in ids
    assert "phase1_dla_parity_n4_ibm_kingston" in ids
    assert "bkt_finite_size_grid_planned" in ids


def test_registry_filters_planned_rows() -> None:
    """Planned rows are visible by default but removable for implemented-only views."""

    rows = list_synchronisation_benchmarks(include_planned=False)

    assert rows
    assert all(row.evidence_class != "planned" for row in rows)


def test_registry_payload_has_result_schema_and_counts() -> None:
    """The exported payload carries the result schema and deterministic counts."""

    payload = synchronisation_benchmark_registry_payload()

    assert payload["schema"] == "synchronisation_benchmark_registry_v1"
    assert payload["result_schema"] == RESULT_SCHEMA
    assert payload["implemented_or_replay_count"] == 3
    assert payload["planned_count"] == 1
    assert "hardware_submission" in payload["result_schema"]["required_fields"]


def test_kuramoto_ring_runner_emits_schema_compatible_rows() -> None:
    """The first no-QPU runner emits rows matching required schema fields."""

    from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
        run_kuramoto_ring_n4_linear_omega,
    )

    payload = run_kuramoto_ring_n4_linear_omega(command="test", commit="abc123")
    required = set(payload["result_schema"]["required_fields"])

    assert payload["benchmark_id"] == "kuramoto_ring_n4_linear_omega"
    assert payload["hardware_submission"] is False
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert required.issubset(row)
        assert row["hardware_submission"] is False
        assert row["observables"]


def test_kuramoto_ring_references_are_finite_and_bounded() -> None:
    """Reference rows must remain physical for the canonical n=4 smoke case."""

    import math

    from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
        run_kuramoto_ring_n4_linear_omega,
    )

    payload = run_kuramoto_ring_n4_linear_omega(command="test", commit="abc123")
    values = {
        observable["name"]: observable["value"]
        for row in payload["rows"]
        for observable in row["observables"]
    }

    assert 0.0 <= values["order_parameter_t1"] <= 1.0
    assert abs(values["state_norm_t1"] - 1.0) <= 1e-10
    assert math.isfinite(values["energy_expectation_t1"])


def test_kuramoto_chain_n8_runner_emits_schema_compatible_rows() -> None:
    """The n=8 decaying-chain runner emits finite no-QPU reference rows."""

    from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
        run_kuramoto_chain_n8_decay_omega,
    )

    payload = run_kuramoto_chain_n8_decay_omega(command="test", commit="abc123")
    required = set(payload["result_schema"]["required_fields"])

    assert payload["benchmark_id"] == "kuramoto_chain_n8_decay_omega"
    assert payload["hardware_submission"] is False
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert required.issubset(row)
        assert row["hardware_submission"] is False
        assert row["observables"]


def test_kuramoto_chain_n8_references_are_finite_and_bounded() -> None:
    """The n=8 decaying-chain reference values stay physically bounded."""

    import math

    from scpn_quantum_control.benchmark_harness.synchronisation_runner import (
        run_kuramoto_chain_n8_decay_omega,
    )

    payload = run_kuramoto_chain_n8_decay_omega(command="test", commit="abc123")
    values = {
        observable["name"]: observable["value"]
        for row in payload["rows"]
        for observable in row["observables"]
    }

    assert 0.0 <= values["order_parameter_t1"] <= 1.0
    assert abs(values["state_norm_t1"] - 1.0) <= 1e-10
    assert math.isfinite(values["energy_expectation_t1"])
