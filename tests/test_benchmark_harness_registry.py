# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- benchmark harness registry tests
"""Tests for S5 benchmark-harness registry metadata."""

from __future__ import annotations

from scpn_quantum_control.benchmark_harness.registry import (
    benchmark_registry_payload,
    list_benchmark_families,
)


def test_registry_lists_phase1_as_only_implemented_family() -> None:
    rows = list_benchmark_families(include_planned=False)

    assert [row.benchmark_id for row in rows] == ["phase1_dla_parity"]
    assert rows[0].status == "implemented"
    assert rows[0].command == "scpn-bench s5-benchmark-suite"


def test_registry_planned_rows_are_explicitly_not_available() -> None:
    rows = list_benchmark_families()
    planned = [row for row in rows if row.status == "planned"]

    assert {row.benchmark_id for row in planned} == {
        "chsh_hardware",
        "bkt_phase_transition",
        "otoc_scrambling",
        "dla_dimension",
    }
    assert all(row.command is None for row in planned)
    assert all(row.generated_artifact is None for row in planned)
    assert all(row.blocker for row in planned)


def test_registry_payload_counts_statuses() -> None:
    payload = benchmark_registry_payload()

    assert payload["schema"] == "benchmark_harness_registry_v1"
    assert payload["implemented_count"] == 1
    assert payload["planned_count"] == 4
    assert payload["blocked_count"] == 0
    assert len(payload["families"]) == 5
