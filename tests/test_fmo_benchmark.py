# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Fmo Benchmark
"""Tests for FMO photosynthetic complex benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.fmo_benchmark import (
    FMO_COUPLING,
    FMO_SITE_ENERGIES,
    fmo_benchmark,
    fmo_coupling_matrix,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestFMOData:
    """The built-in FMO reference data."""

    def test_fmo_coupling_symmetric(self) -> None:
        """The FMO coupling is symmetric."""
        np.testing.assert_allclose(FMO_COUPLING, FMO_COUPLING.T, atol=1e-10)

    def test_fmo_coupling_7x7(self) -> None:
        """The FMO coupling is 7 by 7."""
        assert FMO_COUPLING.shape == (7, 7)

    def test_fmo_site_energies_7(self) -> None:
        """There are seven site energies."""
        assert FMO_SITE_ENERGIES.shape == (7,)

    def test_fmo_diagonal_zero(self) -> None:
        """The coupling diagonal is zero."""
        np.testing.assert_allclose(np.diag(FMO_COUPLING), 0.0)

    def test_builtin_reference_requires_explicit_opt_in(self) -> None:
        """The built-in reference needs an explicit opt-in."""
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            fmo_coupling_matrix()

    def test_fmo_coupling_matrix_units(self) -> None:
        """The coupling matrix uses absolute couplings for seven sites."""
        K, omega = fmo_coupling_matrix(allow_builtin_reference=True)
        assert K.shape == (7, 7)
        assert omega.shape == (7,)
        assert np.all(K >= 0)  # absolute values
        assert np.all(np.diag(K) == 0)


class TestFMOBenchmark:
    """The SCPN-versus-FMO benchmark."""

    def test_benchmark_returns_result(self) -> None:
        """The benchmark reports seven oscillators and a correlation in [-1, 1]."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_oscillators == 7
        assert -1 <= result.topology_correlation <= 1
        assert result.topology_similarity_proxy == result.topology_correlation
        assert -1 <= result.frequency_correlation <= 1
        assert result.coupling_ratio > 0
        assert result.frequency_ratio > 0

    def test_benchmark_self_comparison(self) -> None:
        """FMO compared against itself should have perfect correlation."""
        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)
        result = fmo_benchmark(K_fmo, omega_fmo, allow_builtin_reference=True)
        assert result.topology_correlation == pytest.approx(1.0, abs=0.01)

    def test_benchmark_4_oscillators(self) -> None:
        """Works with fewer than 7 oscillators."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert result.n_oscillators == 4

    def test_low_topology_match_reports_no_correlation(self) -> None:
        """A poor topology match reports no proxy agreement."""
        rng = np.random.default_rng(1)
        K = rng.random((7, 7))
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 0.0)
        result = fmo_benchmark(K, OMEGA_N_16[:7], allow_builtin_reference=True)
        assert abs(result.topology_correlation) <= 0.3
        assert "no proxy agreement" in result.summary

    def test_benchmark_refuses_implicit_builtin_reference(self) -> None:
        """An implicit built-in reference is refused."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        with pytest.raises(RuntimeError, match="allow_builtin_reference"):
            fmo_benchmark(K, omega)

    def test_result_labels_builtin_reference_source_mode(self) -> None:
        """A built-in reference is labelled as such and not publication-safe."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert result.source_mode == "builtin_literature_reference"
        assert result.publication_safe is False

    def test_result_labels_measured_reference_source_mode(self) -> None:
        """A measured reference is labelled as measured and publication-safe."""
        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)
        result = fmo_benchmark(
            K_fmo,
            omega_fmo,
            fmo_coupling=K_fmo,
            fmo_frequencies=omega_fmo,
        )
        assert result.source_mode == "measured"
        assert result.publication_safe is True

    def test_reference_matrix_and_frequency_vector_must_be_supplied_together(self) -> None:
        """The reference coupling and frequencies must be supplied together."""
        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)
        with pytest.raises(ValueError, match="fmo_coupling and fmo_frequencies"):
            fmo_benchmark(K_fmo, omega_fmo, fmo_coupling=K_fmo)

    def test_scpn_vs_fmo_topology(self) -> None:
        """Record SCPN vs FMO topology correlation — this is Gap 1 data."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        # The correlation value IS the finding — record it
        print(f"\n  SCPN vs FMO topology correlation: {result.topology_correlation:.3f}")
        print(f"  SCPN vs FMO frequency correlation: {result.frequency_correlation:.3f}")
        print(f"  {result.summary}")
        # No assertion on the value — this is measurement, not validation
        assert isinstance(result.topology_correlation, float)

    def test_summary_string(self) -> None:
        """The summary names the comparison and the topology."""
        K = build_knm_paper27(L=7)
        omega = OMEGA_N_16[:7]
        result = fmo_benchmark(K, omega, allow_builtin_reference=True)
        assert "SCPN vs FMO" in result.summary
        assert "topology" in result.summary

    def test_rejects_non_square_scpn_coupling(self) -> None:
        """A non-square SCPN coupling is refused."""
        K = np.ones((2, 3))
        omega = np.ones(2)
        with pytest.raises(ValueError, match="K_scpn must be a square"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_scpn_frequency_shape_mismatch(self) -> None:
        """SCPN frequencies of the wrong shape are refused."""
        K = build_knm_paper27(L=4)
        omega = np.ones(3)
        with pytest.raises(ValueError, match="omega_scpn must match"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_non_finite_scpn_coupling(self) -> None:
        """A non-finite SCPN coupling is refused."""
        K = build_knm_paper27(L=4)
        K[0, 1] = np.nan
        omega = OMEGA_N_16[:4]
        with pytest.raises(ValueError, match="K_scpn must contain only finite"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_too_small_comparison_system(self) -> None:
        """Fewer than two coupled sites are refused."""
        K = np.array([[0.0]])
        omega = np.array([1.0])
        with pytest.raises(ValueError, match="at least two coupled sites"):
            fmo_benchmark(K, omega, allow_builtin_reference=True)

    def test_rejects_fmo_coupling_asymmetry(self) -> None:
        """An asymmetric FMO coupling is refused."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fmo_K = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.2, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        fmo_omega = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="fmo_coupling must be symmetric"):
            fmo_benchmark(K, omega, fmo_coupling=fmo_K, fmo_frequencies=fmo_omega)

    def test_rejects_fmo_coupling_nonzero_diagonal(self) -> None:
        """An FMO coupling with a non-zero diagonal is refused."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fmo_K = np.array(
            [
                [1.0, 0.8, 0.1],
                [0.8, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        fmo_omega = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="fmo_coupling diagonal must be zero"):
            fmo_benchmark(K, omega, fmo_coupling=fmo_K, fmo_frequencies=fmo_omega)

    def test_rejects_fmo_frequency_shape_mismatch(self) -> None:
        """FMO frequencies of the wrong shape are refused."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        fmo_K = np.array(
            [
                [0.0, 0.8, 0.1],
                [0.8, 0.0, 0.3],
                [0.1, 0.3, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="fmo_frequencies must match"):
            fmo_benchmark(K, omega, fmo_coupling=fmo_K, fmo_frequencies=np.ones(2))


# ---------------------------------------------------------------------------
# FMO physics: biological coupling structure
# ---------------------------------------------------------------------------


class TestFMOPhysics:
    """Physical properties of the FMO data."""

    def test_fmo_coupling_non_negative(self) -> None:
        """FMO coupling (absolute) must be non-negative."""
        K, _ = fmo_coupling_matrix(allow_builtin_reference=True)
        assert np.all(K >= 0)

    def test_fmo_site_energies_ordered(self) -> None:
        """FMO site energies span a range (not all equal)."""
        assert FMO_SITE_ENERGIES.max() > FMO_SITE_ENERGIES.min()


# ---------------------------------------------------------------------------
# Pipeline: FMO coupling → benchmark → correlation → wired
# ---------------------------------------------------------------------------


class TestFMOPipeline:
    """The FMO-to-benchmark pipeline."""

    def test_pipeline_fmo_to_benchmark(self) -> None:
        """Full pipeline: FMO coupling → benchmark → topology correlation.

        Verifies FMO benchmark is wired and produces cross-domain data.
        """
        import time

        K_fmo, omega_fmo = fmo_coupling_matrix(allow_builtin_reference=True)

        t0 = time.perf_counter()
        result = fmo_benchmark(K_fmo, omega_fmo, allow_builtin_reference=True)
        dt = (time.perf_counter() - t0) * 1000

        assert result.topology_correlation == pytest.approx(1.0, abs=0.01)

        print(f"\n  PIPELINE FMO→Benchmark (7 sites): {dt:.1f} ms")
        print(f"  ρ_topo={result.topology_correlation:.4f}")
