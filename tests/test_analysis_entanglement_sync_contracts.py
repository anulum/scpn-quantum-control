# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Analysis entanglement and synchronisation contract tests
"""Contract tests for entanglement, synchronisation, OTOC, QFI, and self-consistency analysis behaviours."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_hamiltonian,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

OMEGA_2 = np.array([1.0, 1.2])
K_TOPO_2 = build_knm_paper27(L=2)


def _dense_H(K: float = 1.0) -> NDArray[np.complex128]:
    H = knm_to_hamiltonian(K_TOPO_2 * K, OMEGA_2).to_matrix()
    if hasattr(H, "toarray"):
        H = H.toarray()
    return np.array(H)


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


def test_entanglement_entropy_single_qubit_partition() -> None:
    """Verifies 105: n_A = 1 when n//2 == 0 (single-qubit system)."""
    from scpn_quantum_control.analysis.entanglement_entropy import entanglement_at_coupling

    omega = np.array([1.0])
    K_topology = np.array([[1.0]])
    result = entanglement_at_coupling(omega, K_topology, K_base=0.5)
    assert hasattr(result, "entropy")


def test_qfi_precision_zero_diagonal() -> None:
    """Verifies 49: zero QFI diagonal gives infinite precision bound."""
    from scpn_quantum_control.analysis.qfi import QFIResult

    result = QFIResult(
        qfi_matrix=np.zeros((1, 1)),
        coupling_pairs=[(0, 1)],
        precision_bounds=np.array([float("inf")]),
        spectral_gap=0.1,
        n_qubits=2,
    )
    assert result.precision_for(0, 1) == float("inf")


def test_otoc_default_times() -> None:
    """Verifies 99: times defaults to linspace when None."""
    from scpn_quantum_control.analysis.otoc import compute_otoc

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = compute_otoc(K, omega, w_qubit=0, times=None)
    assert len(result.times) == 30


def test_otoc_lyapunov_zero_f0() -> None:
    """Verifies 152: _estimate_lyapunov returns None when f0 near zero."""
    from scpn_quantum_control.analysis.otoc import _estimate_lyapunov

    times = np.linspace(0, 1, 10)
    otoc = np.zeros(10)  # F(0) ≈ 0
    result = _estimate_lyapunov(times, otoc)
    assert result is None


def test_otoc_lyapunov_few_positive() -> None:
    """Verifies 163: _estimate_lyapunov returns None when < 3 positive points."""
    from scpn_quantum_control.analysis.otoc import _estimate_lyapunov

    times = np.linspace(0, 1, 10)
    # All values close to F0 — decay barely positive
    otoc = np.ones(10) * 1.0
    otoc[1] = 0.9999
    result = _estimate_lyapunov(times, otoc)
    assert result is None


def test_otoc_scrambling_time_zero_f0() -> None:
    """Verifies 176: _estimate_scrambling_time returns None for f0~0."""
    from scpn_quantum_control.analysis.otoc import _estimate_scrambling_time

    times = np.linspace(0, 1, 10)
    otoc = np.zeros(10)
    result = _estimate_scrambling_time(times, otoc)
    assert result is None


def test_self_consistency_zero_counts() -> None:
    """Verifies 74-75: _two_point_from_counts returns zeros for empty counts."""
    from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
        _two_point_from_counts,
    )

    result = _two_point_from_counts({}, 3)
    np.testing.assert_array_equal(result, np.zeros((3, 3)))


def test_self_consistency_from_counts() -> None:
    """Verifies 137-144: self_consistency_from_counts with synthetic counts."""
    from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
        self_consistency_from_counts,
    )

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    x_counts = {"00": 400, "01": 100, "10": 100, "11": 400}
    y_counts = {"00": 400, "01": 100, "10": 100, "11": 400}
    result = self_consistency_from_counts(K, omega, x_counts, y_counts, maxiter=5)
    assert hasattr(result, "frobenius_error")


class TestEntanglementEntropyJaxPath:
    """Entanglement scans when the JAX path is not taken."""

    def test_scan_falls_through_without_jax(self) -> None:
        """A three-point coupling scan returns one entropy per coupling."""
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_vs_coupling

        K_topo = np.array([[0, 1], [1, 0]], dtype=float)
        omega = OMEGA_N_16[:2]
        result = entanglement_vs_coupling(omega, K_topo, k_range=np.array([0.5, 1.0, 2.0]))
        assert len(result.entropy) == 3


class TestSyncWitnessTopological:
    """Topological synchronisation witness from correlator matrices."""

    def test_topological_witness_no_ripser(self) -> None:
        """Without ripser the witness is still named but carries no value."""
        from scpn_quantum_control.analysis.sync_witness import (
            topological_witness_from_correlator,
        )

        corr = np.array([[1, 0.5], [0.5, 1]])
        with patch.dict(sys.modules, {"ripser": None}):
            result = topological_witness_from_correlator(corr)
        assert result.witness_name == "topological"

    def test_topological_witness_empty_h1(self) -> None:
        """An empty H1 diagram gives a persistent fraction of zero."""
        from scpn_quantum_control.analysis.sync_witness import (
            topological_witness_from_correlator,
        )

        mock_ripser = MagicMock()
        mock_ripser.ripser.return_value = {"dgms": [np.array([[0, 1]]), np.empty((0, 2))]}
        with patch.dict(sys.modules, {"ripser": mock_ripser}):
            result = topological_witness_from_correlator(np.eye(3))
        assert result.raw_observable == 0.0

    def test_topological_witness_counts_bars_longer_than_the_median(self) -> None:
        """Only H1 bars outliving the median lifetime count towards the persistent fraction.

        ripser is an optional extra with no CI lane, so the documented result shape stands in.
        """
        from scpn_quantum_control.analysis.sync_witness import (
            topological_witness_from_correlator,
        )

        mock_ripser = MagicMock()
        h1 = np.array([[0.0, 0.1], [0.0, 0.3], [0.0, 0.5]])
        mock_ripser.ripser.return_value = {"dgms": [np.array([[0.0, np.inf]]), h1]}
        corr = np.full((4, 4), 0.25) + 0.75 * np.eye(4)
        with patch.dict(sys.modules, {"ripser": mock_ripser}):
            result = topological_witness_from_correlator(corr, threshold=0.5)

        distance = mock_ripser.ripser.call_args.args[0]
        np.testing.assert_allclose(np.diag(distance), 0.0)
        np.testing.assert_allclose(distance[0, 1], 0.75)
        assert mock_ripser.ripser.call_args.kwargs == {"maxdim": 1, "distance_matrix": True}
        assert result.raw_observable == pytest.approx(1 / 6)
        assert result.expectation_value == pytest.approx(1 / 6 - 0.5)
        assert result.is_synchronized is True


class TestSyncEntanglementWitnessEdge:
    """Edge cases of the synchronisation entanglement witness."""

    def test_certified_entanglement_depth_is_conservative(self) -> None:
        """The certified depth is one without a witness and two with one."""
        from scpn_quantum_control.analysis.sync_entanglement_witness import (
            _certified_entanglement_depth,
        )

        assert _certified_entanglement_depth(False) == 1
        assert _certified_entanglement_depth(True) == 2

    def test_R_from_statevector(self) -> None:
        """The order parameter of a basis state is a float in [0, 1]."""
        from scpn_quantum_control.analysis.sync_entanglement_witness import (
            R_from_statevector,
        )

        psi = np.array([1, 0, 0, 0], dtype=complex)
        r = R_from_statevector(psi, 2)
        assert isinstance(r, float)
        assert 0 <= r <= 1


class TestQFICriticalityDefaults:
    """Default coupling grid of the QFI criticality scan."""

    def test_default_k_range(self) -> None:
        """Without a range the scan uses twenty couplings."""
        from scpn_quantum_control.analysis.qfi_criticality import qfi_vs_coupling

        result = qfi_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 20


class TestEntanglementPercolationEdge:
    """Default grid and the no-entanglement edge of the percolation scan."""

    def test_default_k_range(self) -> None:
        """Without a range the scan uses twenty couplings."""
        from scpn_quantum_control.analysis.entanglement_percolation import percolation_scan

        result = percolation_scan(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 20

    def test_no_entangled_pairs(self) -> None:
        """A vanishing coupling leaves no entangled pair."""
        from scpn_quantum_control.analysis.entanglement_percolation import percolation_scan

        result = percolation_scan(OMEGA_N_16[:2], _ring(2), k_range=np.array([0.001]))
        assert result.n_entangled_pairs[0] == 0


class TestOTOCProbeDefaults:
    """Default coupling grid of the OTOC synchronisation probe."""

    def test_default_k_range(self) -> None:
        """Without a range the probe still scans some couplings."""
        from scpn_quantum_control.analysis.otoc_sync_probe import otoc_sync_scan

        result = otoc_sync_scan(_ring(2), OMEGA_N_16[:2])
        assert len(result.K_base_values) > 0


class TestSelfConsistencyEdge:
    """Shot-noise estimate of the Hamiltonian self-consistency check."""

    def test_correlator_shot_noise(self) -> None:
        """Balanced counts give a positive correlator shot-noise estimate."""
        from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
            correlator_shot_noise,
        )

        x_counts = {"00": 500, "11": 500}
        y_counts = {"00": 500, "11": 500}
        noise = correlator_shot_noise(x_counts, y_counts, 2)
        assert noise > 0


class TestMagicNonstabilizerness:
    """Stabiliser Renyi entropy at one coupling and across a scan."""

    def test_magic_at_coupling(self) -> None:
        """The stabiliser Renyi entropy at one coupling is non-negative."""
        from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_at_coupling

        result = magic_at_coupling(OMEGA_2, K_TOPO_2, K_base=1.0)
        assert result.sre_m2 >= 0

    def test_magic_scan(self) -> None:
        """A three-point scan returns one entropy per coupling."""
        from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_vs_coupling

        scan = magic_vs_coupling(OMEGA_2, K_TOPO_2, k_range=np.array([0.5, 1.0, 2.0]))
        assert len(scan.k_values) == 3
        assert len(scan.sre_m2) == 3


class TestKrylovComplexity:
    """Lanczos coefficients and Krylov complexity of a small Hamiltonian."""

    def test_lanczos_coefficients(self) -> None:
        """Lanczos recursion returns at most the requested number of coefficients."""
        from scpn_quantum_control.analysis.krylov_complexity import lanczos_coefficients

        H = _dense_H()
        op = np.zeros((4, 4), dtype=np.complex128)
        op[0, 1] = 1.0
        b_n = lanczos_coefficients(H, op, max_steps=3)
        assert len(b_n) <= 3

    def test_krylov_at_coupling(self) -> None:
        """Krylov complexity records its Lanczos coefficients."""
        from scpn_quantum_control.analysis.krylov_complexity import krylov_complexity

        H = _dense_H()
        op = np.zeros((4, 4), dtype=np.complex128)
        op[0, 1] = 1.0
        result = krylov_complexity(H, op, t_max=0.5, n_times=5, max_lanczos=3)
        assert result.lanczos_b is not None


class TestLoschmidtEcho:
    """Loschmidt echo after a coupling quench and across quench targets."""

    def test_quench(self) -> None:
        """A quench returns one Loschmidt amplitude per time point."""
        from scpn_quantum_control.analysis.loschmidt_echo import loschmidt_quench

        result = loschmidt_quench(
            OMEGA_2, K_TOPO_2, K_initial=0.5, K_final=2.0, t_max=1.0, n_times=5
        )
        assert len(result.loschmidt_amplitude) == 5

    def test_quench_scan(self) -> None:
        """A quench scan records its final couplings."""
        from scpn_quantum_control.analysis.loschmidt_echo import quench_scan

        scan = quench_scan(
            OMEGA_2,
            K_TOPO_2,
            K_initial=0.5,
            K_final_range=np.array([1.0, 2.0]),
            t_max=0.5,
            n_times=5,
        )
        assert "K_final" in scan


class TestEntanglementEntropy:
    """Entanglement entropy at one coupling and across a scan."""

    def test_at_coupling(self) -> None:
        """The entanglement entropy at one coupling is non-negative."""
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_at_coupling

        result = entanglement_at_coupling(OMEGA_2, K_TOPO_2, K_base=1.0)
        assert result.entropy >= 0

    def test_scan(self) -> None:
        """A three-point scan returns one entropy per coupling."""
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_vs_coupling

        scan = entanglement_vs_coupling(OMEGA_2, K_TOPO_2, k_range=np.array([0.5, 1.0, 2.0]))
        assert len(scan.entropy) == 3


class TestPairingCorrelator:
    """Pairing correlator map and its anisotropy scan."""

    def test_pairing_map(self) -> None:
        """The pairing map at one coupling is produced."""
        from scpn_quantum_control.analysis.pairing_correlator import pairing_map

        result = pairing_map(OMEGA_2, K_TOPO_2, K_base=1.0)
        assert result is not None

    def test_pairing_vs_anisotropy(self) -> None:
        """The anisotropy scan records its anisotropy values."""
        from scpn_quantum_control.analysis.pairing_correlator import pairing_vs_anisotropy

        scan = pairing_vs_anisotropy(
            OMEGA_2, K_TOPO_2, K_base=1.0, delta_range=np.array([0.0, 0.5])
        )
        assert "delta" in scan


class TestOTOCSyncProbe:
    """OTOC synchronisation scan and its comparison with the order parameter."""

    def test_otoc_sync_scan(self) -> None:
        """A two-point scan returns two couplings."""
        from scpn_quantum_control.analysis.otoc_sync_probe import otoc_sync_scan

        scan = otoc_sync_scan(
            K_TOPO_2, OMEGA_2, K_base_range=np.array([0.5, 1.0]), n_time_points=5
        )
        assert len(scan.K_base_values) == 2

    def test_compare_otoc_vs_R(self) -> None:
        """The comparison reports the OTOC critical coupling."""
        from scpn_quantum_control.analysis.otoc_sync_probe import compare_otoc_vs_R, otoc_sync_scan

        scan = otoc_sync_scan(
            K_TOPO_2, OMEGA_2, K_base_range=np.array([0.5, 1.0, 2.0]), n_time_points=5
        )
        result = compare_otoc_vs_R(scan)
        assert "K_c_otoc" in result
