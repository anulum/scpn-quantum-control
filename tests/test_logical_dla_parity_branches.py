# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the logical DLA-parity resource model
"""Guard and branch tests for the logical DLA-parity resource estimator.

Covers the syndrome-round and empty-distance guards, the qubit-count validator,
and the hierarchical-candidate and flat-surface-code comparison conclusions.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.qec.logical_dla_parity import (
    _validate_n_oscillators,
    compare_flat_surface_code_to_multiscale,
    estimate_logical_dla_parity_row,
    estimate_s7_resource_table,
)


def test_row_rejects_non_positive_syndrome_round() -> None:
    """A non-positive syndrome-round duration is rejected."""
    with pytest.raises(ValueError, match="syndrome_round_us must be finite and positive"):
        estimate_logical_dla_parity_row(code_distance=3, syndrome_round_us=0.0)


def test_resource_table_rejects_empty_distances() -> None:
    """An empty code-distance bundle is rejected."""
    with pytest.raises(ValueError, match="code_distances must not be empty"):
        estimate_s7_resource_table(code_distances=())


def test_validate_n_oscillators_rejects_below_two() -> None:
    """A qubit count below two is rejected."""
    with pytest.raises(ValueError, match="n_oscillators must be at least 2"):
        _validate_n_oscillators(1)


def test_comparison_reports_hierarchical_candidate() -> None:
    """A regime where the hierarchical scheme wins on qubits and error is reported."""
    comparison = compare_flat_surface_code_to_multiscale(
        n_oscillators=8, flat_distance=11, multiscale_distances=(7, 7)
    )
    assert comparison.conclusion.startswith("hierarchical_candidate")


def test_comparison_reports_flat_surface_code_advantage() -> None:
    """A regime where the flat surface code wins on qubit overhead is reported."""
    comparison = compare_flat_surface_code_to_multiscale(
        flat_distance=3, multiscale_distances=(7, 7, 7, 7, 7)
    )
    assert (
        comparison.conclusion == "flat_surface_code_lower_qubit_overhead_for_this_distance_bundle"
    )
