"""Tests for ssgf.quantum_spectral."""

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.ssgf.quantum_spectral import (
    SpectralBridgeResult,
    entrainment_criterion,
    laplacian_spectrum,
    qpe_resource_estimate,
    spectral_bridge_analysis,
)


def _small_system():
    return build_knm_paper27(L=3), OMEGA_N_16[:3]


class TestLaplacianSpectrum:
    def test_first_eigenvalue_zero(self):
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert abs(spec[0]) < 1e-10

    def test_sorted_ascending(self):
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert all(spec[i] <= spec[i + 1] + 1e-12 for i in range(len(spec) - 1))

    def test_length_matches_n(self):
        K, _ = _small_system()
        spec = laplacian_spectrum(K)
        assert len(spec) == 3


class TestEntrainmentCriterion:
    def test_strong_coupling_stable(self):
        K = build_knm_paper27(L=3, K_base=5.0)
        omega = OMEGA_N_16[:3]
        stable, margin = entrainment_criterion(K, omega)
        assert stable is True
        assert margin > 0

    def test_returns_tuple(self):
        K, omega = _small_system()
        result = entrainment_criterion(K, omega)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestQPEResourceEstimate:
    def test_returns_positive(self):
        K, _ = _small_system()
        n_bits, depth = qpe_resource_estimate(K)
        assert n_bits >= 1
        assert depth > 0

    def test_higher_precision_more_bits(self):
        K, _ = _small_system()
        n1, _ = qpe_resource_estimate(K, epsilon=0.1)
        n2, _ = qpe_resource_estimate(K, epsilon=0.001)
        assert n2 >= n1


class TestSpectralBridgeAnalysis:
    def test_returns_result(self):
        K, omega = _small_system()
        result = spectral_bridge_analysis(K, omega)
        assert isinstance(result, SpectralBridgeResult)
        assert result.fiedler_value >= 0
        assert result.qpe_bits_needed >= 1
        assert len(result.laplacian_spectrum) == 3
