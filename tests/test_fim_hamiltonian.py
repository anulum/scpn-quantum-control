# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FIM Hamiltonian Tests
"""Tests for the FIM Hamiltonian offline diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.fim_hamiltonian import (
    add_fim_feedback,
    adjacent_gap_ratio,
    bipartite_entropy_from_statevector,
    commutator_frobenius_norm_with_diagonal,
    computational_magnetisations,
    fim_diagonal,
    magnetisation_operator_diagonal,
    magnetisation_sector_indices,
    sector_coupling_rows,
    sector_spectrum_rows,
    summarise_spectrum,
)


def test_computational_magnetisations_follow_z_convention() -> None:
    assert computational_magnetisations(2).tolist() == [2, 0, 0, -2]


def test_computational_magnetisations_rejects_non_positive_qubit_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        computational_magnetisations(0)


def test_fim_diagonal_matches_minus_lambda_m_squared_over_n() -> None:
    diagonal = fim_diagonal(2, 0.5)
    np.testing.assert_allclose(diagonal, [-1.0, 0.0, 0.0, -1.0])


def test_fim_feedback_rejects_invalid_hamiltonian_dimensions() -> None:
    with pytest.raises(ValueError, match="square"):
        add_fim_feedback(np.ones((2, 3)), 1.0)
    with pytest.raises(ValueError, match="power of two"):
        add_fim_feedback(np.eye(3), 1.0)


def test_add_fim_feedback_adds_only_diagonal_term() -> None:
    base = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    shifted = add_fim_feedback(base, 2.0)
    np.testing.assert_allclose(shifted, [[-2.0, 1.0], [1.0, -2.0]])


def test_magnetisation_sector_indices_group_basis_states() -> None:
    sectors = magnetisation_sector_indices(2)
    assert sectors[2].tolist() == [0]
    assert sectors[0].tolist() == [1, 2]
    assert sectors[-2].tolist() == [3]


def test_adjacent_gap_ratio_uses_nonzero_spacings() -> None:
    stats = adjacent_gap_ratio(np.array([0.0, 1.0, 3.0, 6.0]))
    assert stats["n_spacings"] == 3
    np.testing.assert_allclose(stats["mean_r"], (0.5 + 2.0 / 3.0) / 2.0)


def test_adjacent_gap_ratio_reports_insufficient_nonzero_spacings() -> None:
    stats = adjacent_gap_ratio(np.array([0.0, 0.0, 1e-12, 2.0]))

    assert stats["n_spacings"] == 1
    assert stats["mean_r"] is None
    assert stats["median_r"] is None
    assert stats["min_spacing"] == pytest.approx(2.0)
    assert stats["max_spacing"] == pytest.approx(2.0)


def test_spectrum_summary_records_singleton_gap_as_unavailable() -> None:
    summary = summarise_spectrum(np.array([1.25]), n_qubits=1, lambda_fim=0.5)

    assert summary.dimension == 1
    assert summary.spectral_gap is None
    assert summary.spectral_width == 0.0
    assert summary.ground_energy == pytest.approx(1.25)


def test_bipartite_entropy_product_and_bell_states() -> None:
    product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    assert bipartite_entropy_from_statevector(product, 2, keep=[0]) == 0.0
    assert bipartite_entropy_from_statevector(product, 2) == 0.0
    np.testing.assert_allclose(bipartite_entropy_from_statevector(bell, 2, keep=[0]), 1.0)


def test_bipartite_entropy_rejects_bad_state_shape_and_keep_indices() -> None:
    state = np.array([1.0, 0.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="length"):
        bipartite_entropy_from_statevector(state, 2)
    with pytest.raises(ValueError, match="invalid qubit"):
        bipartite_entropy_from_statevector(np.array([1.0, 0.0, 0.0, 0.0]), 2, keep=[2])
    assert bipartite_entropy_from_statevector(np.array([1.0, 0.0]), 1, keep=[0]) == 0.0


def test_commutator_norm_vanishes_for_diagonal_operator_commuting_with_diagonal_h() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128)
    magnetisation = magnetisation_operator_diagonal(2)
    assert commutator_frobenius_norm_with_diagonal(hamiltonian, magnetisation) == 0.0


def test_commutator_norm_detects_off_diagonal_sector_mixing_and_rejects_mismatch() -> None:
    hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    magnetisation = magnetisation_operator_diagonal(1)

    assert commutator_frobenius_norm_with_diagonal(hamiltonian, magnetisation) == pytest.approx(
        np.sqrt(8.0)
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        commutator_frobenius_norm_with_diagonal(hamiltonian, np.ones(3))


def test_sector_coupling_rows_detect_no_off_sector_coupling_for_diagonal_h() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128)
    rows = sector_coupling_rows(hamiltonian, lambda_fim=0.0)
    assert all(row["off_sector_frobenius_norm"] == 0.0 for row in rows)
    assert all(row["ideal_unitary_sector_leakage"] == 0.0 for row in rows)


def test_sector_diagnostics_preserve_fim_energy_shifts_and_detect_coupling() -> None:
    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    hamiltonian[0, 1] = hamiltonian[1, 0] = 0.25

    spectrum_rows = sector_spectrum_rows(add_fim_feedback(np.zeros((4, 4)), 0.5), 0.5)
    shift_by_magnetisation = {
        row["magnetisation"]: row["fim_energy_shift"] for row in spectrum_rows
    }
    assert shift_by_magnetisation[2] == pytest.approx(-1.0)
    assert shift_by_magnetisation[0] == pytest.approx(0.0)
    assert shift_by_magnetisation[-2] == pytest.approx(-1.0)

    coupling_rows = sector_coupling_rows(hamiltonian, lambda_fim=0.0)
    coupled_sector = next(row for row in coupling_rows if row["magnetisation"] == 2)
    assert coupled_sector["off_sector_frobenius_norm"] == pytest.approx(0.25)
