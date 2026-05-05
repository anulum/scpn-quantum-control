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
)


def test_computational_magnetisations_follow_z_convention() -> None:
    assert computational_magnetisations(2).tolist() == [2, 0, 0, -2]


def test_fim_diagonal_matches_minus_lambda_m_squared_over_n() -> None:
    diagonal = fim_diagonal(2, 0.5)
    np.testing.assert_allclose(diagonal, [-1.0, 0.0, 0.0, -1.0])


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


def test_bipartite_entropy_product_and_bell_states() -> None:
    product = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    assert bipartite_entropy_from_statevector(product, 2, keep=[0]) == 0.0
    np.testing.assert_allclose(bipartite_entropy_from_statevector(bell, 2, keep=[0]), 1.0)


def test_commutator_norm_vanishes_for_diagonal_operator_commuting_with_diagonal_h() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128)
    magnetisation = magnetisation_operator_diagonal(2)
    assert commutator_frobenius_norm_with_diagonal(hamiltonian, magnetisation) == 0.0


def test_sector_coupling_rows_detect_no_off_sector_coupling_for_diagonal_h() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0]).astype(np.complex128)
    rows = sector_coupling_rows(hamiltonian, lambda_fim=0.0)
    assert all(row["off_sector_frobenius_norm"] == 0.0 for row in rows)
    assert all(row["ideal_unitary_sector_leakage"] == 0.0 for row in rows)
