# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- logical DLA parity roadmap tests
"""Tests for the S7 logical-DLA parity roadmap model."""

from __future__ import annotations

import pytest

from scpn_quantum_control.qec.logical_dla_parity import (
    LOGICAL_DLA_PARITY_SCHEMA,
    compare_flat_surface_code_to_multiscale,
    estimate_logical_dla_parity_row,
    estimate_s7_resource_table,
    logical_dla_parity_markdown,
    logical_dla_parity_payload,
    repetition_scaffold_physical_qubits,
    surface_code_physical_qubits,
)


def test_surface_code_qubit_formula_for_s7_default() -> None:
    assert surface_code_physical_qubits(16, 3) == 16 * 17
    assert surface_code_physical_qubits(16, 5) == 16 * 49
    assert surface_code_physical_qubits(16, 7) == 16 * 97


def test_repetition_scaffold_is_marked_as_comparison_only() -> None:
    row = estimate_logical_dla_parity_row(n_oscillators=16, code_distance=5)

    assert row.physical_qubits_repetition_scaffold == repetition_scaffold_physical_qubits(16, 5)
    assert row.physical_qubits_flat_surface_code > row.physical_qubits_repetition_scaffold
    assert row.parity_survival_claim_allowed is False
    assert row.status == "theory_required_before_simulation_or_hardware_promotion"


def test_expected_fidelity_improves_with_distance() -> None:
    rows = estimate_s7_resource_table(code_distances=(3, 5, 7), p_physical=0.003)
    fidelities = [row.expected_step_fidelity for row in rows]
    logical_rates = [row.logical_error_rate_per_round for row in rows]

    assert fidelities[0] < fidelities[1] < fidelities[2]
    assert logical_rates[0] > logical_rates[1] > logical_rates[2]


def test_multiscale_comparison_keeps_theory_review_blocker() -> None:
    comparison = compare_flat_surface_code_to_multiscale(
        n_oscillators=16,
        flat_distance=7,
        multiscale_distances=(3, 3, 3, 3, 3),
    )

    assert comparison.flat_surface_code_physical_qubits == 16 * 97
    assert (
        comparison.flat_logical_error_rate_per_round < comparison.multiscale_effective_logical_rate
    )
    assert comparison.multiscale_total_physical_qubits > 0
    assert comparison.multiscale_below_threshold is True
    assert comparison.conclusion == "hierarchical_lower_qubit_overhead_but_logical_rate_not_viable"


def test_payload_blocks_hardware_and_survival_claims() -> None:
    payload = logical_dla_parity_payload()

    assert payload["schema"] == LOGICAL_DLA_PARITY_SCHEMA
    assert payload["no_qpu_submission"] is True
    assert payload["hardware_submission_allowed"] is False
    assert payload["parity_survival_claim_allowed"] is False
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["code_distance"] == 3


def test_markdown_contains_regeneration_gate() -> None:
    markdown = logical_dla_parity_markdown(logical_dla_parity_payload())

    assert "# Logical DLA Parity Roadmap" in markdown
    assert "scpn-bench s7-logical-dla-roadmap" in markdown
    assert "no claim that DLA parity survives" in markdown


@pytest.mark.parametrize("distance", [0, 2, 4])
def test_invalid_distances_fail_closed(distance: int) -> None:
    with pytest.raises(ValueError, match="odd integer"):
        surface_code_physical_qubits(16, distance)


def test_invalid_physical_error_rate_fails_closed() -> None:
    with pytest.raises(ValueError, match="p_physical"):
        estimate_logical_dla_parity_row(code_distance=3, p_physical=1.0)
