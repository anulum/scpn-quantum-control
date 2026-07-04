# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Networked Kuramoto linear-stability spectrum tests
"""Multi-angle tests for the networked Kuramoto linear-stability spectrum.

Covers the eigenvalue match against the NumPy reference for both the symmetric (``eigh``) and the
directed (``eig``) Jacobian, the Goldstone-mode identification (zero eigenvalue, uniform
eigenvector, row-sum-zero), the eigenpair reconstruction, the descending-real-part ordering, the
stability classification of a synchronised state (stable) versus an anti-phase state (unstable,
exercising both Goldstone-position branches), the spectral-gap definition, the convenience
wrappers and the input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import oscillatools.accel.kuramoto_stability_spectrum as stab
from oscillatools.accel import (
    StabilitySpectrum,
    is_synchronisation_stable,
    networked_kuramoto_jacobian,
    stability_spectrum,
    synchronisation_rate,
)


def _random_symmetric_coupling(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coupling = rng.uniform(0.2, 1.0, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return coupling


def _sorted_eigenvalues(values: np.ndarray) -> np.ndarray:
    return np.array(sorted(values, key=lambda z: (z.real, z.imag)))


class TestSpectrumCorrectness:
    def test_eigenvalues_match_numpy_symmetric(self) -> None:
        theta = np.linspace(-0.4, 0.4, 6)
        coupling = _random_symmetric_coupling(6, 3)
        spectrum = stability_spectrum(theta, coupling)
        reference = np.linalg.eigvals(networked_kuramoto_jacobian(theta, coupling))
        np.testing.assert_allclose(
            _sorted_eigenvalues(spectrum.eigenvalues),
            _sorted_eigenvalues(reference),
            atol=1e-10,
        )
        assert spectrum.is_symmetric

    def test_eigenvalues_match_numpy_directed(self) -> None:
        rng = np.random.default_rng(11)
        coupling = rng.uniform(0.1, 0.9, size=(5, 5))  # not symmetrised -> directed network
        np.fill_diagonal(coupling, 0.0)
        theta = rng.uniform(-math.pi, math.pi, size=5)
        spectrum = stability_spectrum(theta, coupling)
        reference = np.linalg.eigvals(networked_kuramoto_jacobian(theta, coupling))
        np.testing.assert_allclose(
            _sorted_eigenvalues(spectrum.eigenvalues),
            _sorted_eigenvalues(reference),
            atol=1e-10,
        )
        assert not spectrum.is_symmetric

    def test_eigenpairs_reconstruct_the_jacobian_action(self) -> None:
        theta = np.linspace(0.0, 1.0, 7)
        coupling = _random_symmetric_coupling(7, 8)
        jacobian = networked_kuramoto_jacobian(theta, coupling)
        spectrum = stability_spectrum(theta, coupling)
        for index in range(theta.size):
            vector = spectrum.eigenvectors[:, index]
            np.testing.assert_allclose(
                jacobian @ vector, spectrum.eigenvalues[index] * vector, atol=1e-9
            )

    def test_eigenvalues_ordered_by_descending_real_part(self) -> None:
        theta = np.linspace(-1.0, 1.0, 8)
        coupling = _random_symmetric_coupling(8, 21)
        spectrum = stability_spectrum(theta, coupling)
        real_parts = spectrum.eigenvalues.real
        assert np.all(np.diff(real_parts) <= 1e-12)


class TestGoldstoneMode:
    def test_goldstone_is_zero_and_uniform(self) -> None:
        theta = np.linspace(-0.3, 0.5, 6)
        coupling = _random_symmetric_coupling(6, 4)
        spectrum = stability_spectrum(theta, coupling)
        assert abs(spectrum.goldstone_eigenvalue) < 1e-9
        goldstone_vector = spectrum.eigenvectors[:, spectrum.goldstone_index]
        uniform = np.full(theta.size, 1.0 / math.sqrt(theta.size))
        overlap = abs(np.vdot(uniform, goldstone_vector))
        assert overlap > 1.0 - 1e-9

    def test_jacobian_rows_sum_to_zero(self) -> None:
        theta = np.linspace(0.0, 2.0, 5)
        coupling = _random_symmetric_coupling(5, 7)
        jacobian = networked_kuramoto_jacobian(theta, coupling)
        np.testing.assert_allclose(jacobian.sum(axis=1), 0.0, atol=1e-12)


class TestStabilityClassification:
    def test_synchronised_state_is_stable(self) -> None:
        # All phases equal: J = -L (the negative graph Laplacian of a connected positive network),
        # so every transverse eigenvalue is negative and the Goldstone is the largest -> at index 0
        # after the descending sort (exercises the leading_position >= goldstone_index branch).
        coupling = _random_symmetric_coupling(6, 5)
        theta = np.full(6, 0.3)
        spectrum = stability_spectrum(theta, coupling)
        assert spectrum.goldstone_index == 0
        assert spectrum.is_linearly_stable
        assert spectrum.leading_nontrivial_eigenvalue.real < 0.0
        assert spectrum.spectral_gap > 0.0

    def test_anti_phase_pair_is_unstable(self) -> None:
        # Two positively coupled oscillators in anti-phase: eigenvalues {0, 2k}; the transverse
        # mode 2k > 0 is unstable and sorts above the Goldstone (Goldstone at index 1, exercising
        # the leading_position < goldstone_index branch).
        k = 0.75
        coupling = np.array([[0.0, k], [k, 0.0]])
        theta = np.array([0.0, math.pi])
        spectrum = stability_spectrum(theta, coupling)
        assert spectrum.goldstone_index == 1
        assert not spectrum.is_linearly_stable
        np.testing.assert_allclose(
            spectrum.leading_nontrivial_eigenvalue.real, 2.0 * k, atol=1e-12
        )
        assert spectrum.spectral_gap < 0.0
        # The unstable mode is the splay direction [1, -1].
        unstable = spectrum.eigenvectors[:, 0]
        np.testing.assert_allclose(abs(unstable[0]), abs(unstable[1]), atol=1e-12)

    def test_spectral_gap_is_goldstone_minus_leading(self) -> None:
        theta = np.linspace(-0.2, 0.7, 5)
        coupling = _random_symmetric_coupling(5, 33)
        spectrum = stability_spectrum(theta, coupling)
        assert spectrum.spectral_gap == pytest.approx(
            spectrum.goldstone_eigenvalue.real - spectrum.leading_nontrivial_eigenvalue.real
        )

    def test_stability_tolerance_governs_classification(self) -> None:
        # A marginal transverse eigenvalue just above the default tolerance flips the verdict.
        coupling = _random_symmetric_coupling(4, 14)
        theta = np.full(4, -0.1)
        assert is_synchronisation_stable(theta, coupling)
        assert not is_synchronisation_stable(theta, coupling, stability_tolerance=-1.0e3)


class TestConvenienceWrappers:
    def test_synchronisation_rate_equals_spectral_gap(self) -> None:
        theta = np.linspace(0.0, 0.6, 5)
        coupling = _random_symmetric_coupling(5, 9)
        spectrum = stability_spectrum(theta, coupling)
        assert synchronisation_rate(theta, coupling) == pytest.approx(spectrum.spectral_gap)

    def test_is_synchronisation_stable_equals_flag(self) -> None:
        theta = np.full(5, 0.2)
        coupling = _random_symmetric_coupling(5, 2)
        spectrum = stability_spectrum(theta, coupling)
        assert is_synchronisation_stable(theta, coupling) == spectrum.is_linearly_stable

    def test_result_is_a_frozen_dataclass(self) -> None:
        theta = np.full(3, 0.0)
        coupling = _random_symmetric_coupling(3, 1)
        spectrum = stability_spectrum(theta, coupling)
        assert isinstance(spectrum, StabilitySpectrum)
        with pytest.raises(AttributeError):
            spectrum.spectral_gap = 0.0  # type: ignore[misc]


class TestValidation:
    def test_too_few_oscillators(self) -> None:
        with pytest.raises(ValueError, match="at least 2 oscillators"):
            stability_spectrum(np.array([0.1]), np.array([[0.0]]))
        with pytest.raises(ValueError, match="at least 2 oscillators"):
            stab._stability_jacobian(np.zeros(0), np.zeros((0, 0)))

    def test_non_square_coupling(self) -> None:
        with pytest.raises(ValueError, match="square matrix of order 3"):
            stability_spectrum(np.zeros(3), np.zeros((3, 2)))

    def test_wrappers_validate_too(self) -> None:
        with pytest.raises(ValueError, match="at least 2 oscillators"):
            synchronisation_rate(np.array([0.0]), np.array([[0.0]]))
        with pytest.raises(ValueError, match="square matrix of order 2"):
            is_synchronisation_stable(np.zeros(2), np.zeros((2, 3)))
