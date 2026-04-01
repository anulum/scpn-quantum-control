# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coverage tests for applications/ module gaps
"""Tests targeting specific uncovered lines in the applications/ subpackage."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- fmo_benchmark.py line 127: topo_verdict == "no correlation" ---


def test_fmo_no_correlation_verdict():
    """Cover line 127: abs(rho_topo) <= 0.3 gives 'no correlation'."""
    from scpn_quantum_control.applications.fmo_benchmark import fmo_benchmark

    # Random coupling matrix unlikely to correlate with FMO
    rng = np.random.default_rng(42)
    K = rng.uniform(0, 0.01, (7, 7))
    K = (K + K.T) / 2
    np.fill_diagonal(K, 0)
    omega = rng.uniform(10, 20, 7)
    result = fmo_benchmark(K, omega)
    assert hasattr(result, "summary")


# --- eeg_benchmark.py line 80: unknown band raises ValueError ---


def test_eeg_unknown_band():
    """Cover line 80: eeg_coupling_matrix raises for unknown band."""
    import pytest

    from scpn_quantum_control.applications.eeg_benchmark import eeg_coupling_matrix

    with pytest.raises(ValueError, match="Unknown EEG band"):
        eeg_coupling_matrix(band="delta")


# --- eeg_benchmark.py line 106: len(e_flat) < 3 path ---


def test_eeg_benchmark_small_scpn():
    """Cover line 106: topo_corr = 0.0 when < 3 upper-triangle elements."""
    from scpn_quantum_control.applications.eeg_benchmark import eeg_benchmark

    K = np.array([[0, 0.1], [0.1, 0]])
    omega = np.array([10.0, 12.0])
    result = eeg_benchmark(K, omega)
    assert result.topology_correlation == 0.0


# --- eeg_benchmark.py lines 111-113: n < 3 freq_corr path ---


def test_eeg_benchmark_freq_corr_small():
    """Cover lines 111-113: freq_corr = 0.0 when n < 3."""
    from scpn_quantum_control.applications.eeg_benchmark import eeg_benchmark

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([10.0, 12.0])
    result = eeg_benchmark(K, omega)
    assert result.frequency_correlation == 0.0


# --- iter_benchmark.py line 106: len(i_flat) < 3 path ---


def test_iter_benchmark_small_scpn():
    """Cover line 106: topo_corr = 0.0 when < 3 elements."""
    from scpn_quantum_control.applications.iter_benchmark import iter_benchmark

    K = np.array([[0, 0.1], [0.1, 0]])
    omega = np.array([1.0, 2.0])
    result = iter_benchmark(K, omega)
    assert result.topology_correlation == 0.0


# --- iter_benchmark.py lines 111-113: n < 3 freq_corr path ---


def test_iter_benchmark_freq_corr_small():
    """Cover lines 111-113: freq_corr = 0.0 when n < 3."""
    from scpn_quantum_control.applications.iter_benchmark import iter_benchmark

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 2.0])
    result = iter_benchmark(K, omega)
    assert result.frequency_correlation == 0.0


# --- josephson_array.py line 93: unknown topology raises ValueError ---


def test_josephson_unknown_topology():
    """Cover line 93: jja_coupling_matrix raises for unknown topology."""
    import pytest

    from scpn_quantum_control.applications.josephson_array import jja_coupling_matrix

    with pytest.raises(ValueError, match="Unknown topology"):
        jja_coupling_matrix(4, topology="hexagonal")


# --- josephson_array.py line 116: np.sum(mask) < 3 path ---


def test_josephson_benchmark_small():
    """Cover line 116: topo_corr = 0.0 when < 3 nonzero pairs."""
    from scpn_quantum_control.applications.josephson_array import josephson_benchmark

    K = np.array([[0, 0.01], [0.01, 0]])
    omega = np.array([5.0, 6.0])
    result = josephson_benchmark(K, omega)
    assert result.topology_correlation == 0.0


# --- josephson_array.py line 127: freq_corr = 0.0 for n < 3 ---


def test_josephson_freq_corr_small():
    """Cover line 127: freq_corr = 0.0 when n < 3."""
    from scpn_quantum_control.applications.josephson_array import josephson_benchmark

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([5.0, 6.0])
    result = josephson_benchmark(K, omega)
    assert result.frequency_correlation == 0.0


# --- power_grid.py line 109: unknown grid raises ValueError ---


def test_power_grid_unknown_grid():
    """Cover line 109: power_grid_benchmark raises for unknown grid."""
    import pytest

    from scpn_quantum_control.applications.power_grid import power_grid_benchmark

    with pytest.raises(ValueError, match="Unknown grid"):
        K = build_knm_paper27(L=5)
        omega = OMEGA_N_16[:5]
        power_grid_benchmark(K, omega, grid_name="IEEE-9bus")


# --- power_grid.py line 126: len(g_flat) < 3 topo_corr ---


def test_power_grid_small_scpn():
    """Cover line 126: topo_corr = 0.0 when < 3 elements."""
    from scpn_quantum_control.applications.power_grid import power_grid_benchmark

    K = np.array([[0, 0.1], [0.1, 0]])
    omega = np.array([60.0, 60.0])
    result = power_grid_benchmark(K, omega)
    assert result.topology_correlation == 0.0


# --- power_grid.py lines 139-141: freq_corr NaN or n < 3 ---


def test_power_grid_freq_corr_small():
    """Cover lines 139-141: freq_corr = 0.0 when n < 3."""
    from scpn_quantum_control.applications.power_grid import power_grid_benchmark

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([60.0, 60.0])
    result = power_grid_benchmark(K, omega)
    assert result.frequency_correlation == 0.0
