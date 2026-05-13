# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 NTHS spin-glass validation tests
"""Executable simulator fixture tests for the Paper 0 NTHS spin-glass anchor."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.nths_spin_glass_validation import (
    SpinGlassValidationConfig,
    edwards_anderson_parameter,
    hamming_distance_matrix,
    spin_glass_energy,
    ultrametric_violation,
    validate_nths_spin_glass_fixture,
)


def _fixture_problem() -> tuple[np.ndarray, np.ndarray]:
    J_ij = np.array(
        [
            [0.0, 0.84, 0.71, -0.42, -0.16, -0.21],
            [0.84, 0.0, 0.63, -0.38, -0.28, -0.18],
            [0.71, 0.63, 0.0, -0.44, -0.24, -0.33],
            [-0.42, -0.38, -0.44, 0.0, 0.78, 0.69],
            [-0.16, -0.28, -0.24, 0.78, 0.0, 0.73],
            [-0.21, -0.18, -0.33, 0.69, 0.73, 0.0],
        ],
        dtype=np.float64,
    )
    h_i = np.array([0.09, -0.04, 0.03, -0.07, 0.05, -0.02], dtype=np.float64)
    return J_ij, h_i


def test_spin_glass_energy_matches_paper0_hamiltonian() -> None:
    J_ij, h_i = _fixture_problem()
    spins = np.array([1, 1, 1, -1, -1, -1], dtype=np.int8)

    energy = spin_glass_energy(spins, J_ij, h_i)
    manual = 0.0
    for i in range(spins.size):
        for j in range(i + 1, spins.size):
            manual -= J_ij[i, j] * spins[i] * spins[j]
    manual -= float(np.dot(h_i, spins))

    assert energy == pytest.approx(manual)


def test_replica_statistics_and_ultrametric_diagnostic_are_explicit() -> None:
    replicas = np.array(
        [
            [1, 1, 1, -1, -1, -1],
            [1, 1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1],
        ],
        dtype=np.int8,
    )

    q_ea = edwards_anderson_parameter(replicas)
    distances = hamming_distance_matrix(replicas)
    violation = ultrametric_violation(distances)

    assert 0.0 <= q_ea <= 1.0
    assert distances.shape == (4, 4)
    assert np.allclose(np.diag(distances), 0.0)
    assert violation >= 0.0


def test_spin_glass_fixture_consumes_macro_spec_and_records_controls() -> None:
    J_ij, h_i = _fixture_problem()

    result = validate_nths_spin_glass_fixture(J_ij, h_i)

    assert result.spec_key == "nths.spin_glass_hamiltonian"
    assert result.validation_protocol == "paper0.nths.spin_glass.phase_contrast"
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_equation_ids == ("EQ0113",)
    assert "P0R05557" in result.source_ledger_ids
    assert result.state_count == 64
    assert result.ground_state_energy < result.mean_energy
    assert abs(result.ground_state_magnetisation) <= 1.0
    assert 0.0 <= result.edwards_anderson_q <= 1.0
    assert result.null_controls["shuffled_coupling_energy_delta"] > 0.01
    assert result.null_controls["zero_field_energy_delta"] > 0.0
    assert result.null_controls["ferromagnetic_aligned_magnetisation_abs"] == pytest.approx(1.0)
    assert result.null_controls["matched_disorder_q_EA"] == pytest.approx(
        result.edwards_anderson_q
    )
    assert result.problem_metadata["exact_enumeration"] is True


def test_spin_glass_fixture_rejects_invalid_or_unbounded_inputs() -> None:
    J_ij, h_i = _fixture_problem()

    with pytest.raises(ValueError, match="symmetric"):
        validate_nths_spin_glass_fixture(J_ij + np.triu(np.ones_like(J_ij), 1), h_i)

    bad_h = h_i.copy()
    bad_h[0] = np.inf
    with pytest.raises(ValueError, match="h_i must contain only finite values"):
        validate_nths_spin_glass_fixture(J_ij, bad_h)

    with pytest.raises(ValueError, match="exact state count"):
        validate_nths_spin_glass_fixture(
            np.zeros((13, 13), dtype=np.float64),
            np.zeros(13, dtype=np.float64),
            config=SpinGlassValidationConfig(max_exact_states=4096),
        )
