# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Shared Test Fixtures and Hypothesis Strategies
"""Shared fixtures, parametrized system sizes, and hypothesis strategies.

Provides:
  - System-size fixtures (knm_Nq for N in {2,3,4,6,8})
  - Coupling matrix variants (zero, identity, ring, paper27)
  - dt / t_max grids
  - Hypothesis strategies for quantum states, coupling matrices, frequencies
  - AerSimulator runner (sim_runner)
  - Reproducible RNG
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import strategies as st

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_hamiltonian,
)
from scpn_quantum_control.hardware.runner import HardwareRunner

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


# ---------------------------------------------------------------------------
# System sizes — parametrized
# ---------------------------------------------------------------------------

SMALL_SIZES = [2, 3, 4]
MEDIUM_SIZES = [2, 3, 4, 6]
ALL_SIZES = [2, 3, 4, 6, 8]
DT_VALUES = [0.01, 0.05, 0.1]
TMAX_VALUES = [0.1, 0.5, 1.0]


@pytest.fixture(params=SMALL_SIZES, ids=lambda n: f"{n}q")
def n_qubits_small(request):
    """System size from {2, 3, 4}."""
    return request.param


@pytest.fixture(params=MEDIUM_SIZES, ids=lambda n: f"{n}q")
def n_qubits_medium(request):
    """System size from {2, 3, 4, 6}."""
    return request.param


@pytest.fixture(params=ALL_SIZES, ids=lambda n: f"{n}q")
def n_qubits(request):
    """System size from {2, 3, 4, 6, 8}."""
    return request.param


@pytest.fixture(params=DT_VALUES, ids=lambda dt: f"dt={dt}")
def dt(request):
    """Time step from {0.01, 0.05, 0.1}."""
    return request.param


# ---------------------------------------------------------------------------
# Coupling matrices — fixed sizes
# ---------------------------------------------------------------------------


@pytest.fixture
def knm_4q():
    return build_knm_paper27(L=4), OMEGA_N_16[:4]


@pytest.fixture
def knm_8q():
    return build_knm_paper27(L=8), OMEGA_N_16[:8]


def _knm_for_size(n):
    """Build (K, omega) pair for a given system size."""
    return build_knm_paper27(L=n), OMEGA_N_16[:n].copy()


@pytest.fixture
def knm(n_qubits):
    """Parametrized (K, omega) pair — follows n_qubits fixture."""
    return _knm_for_size(n_qubits)


@pytest.fixture
def knm_medium(n_qubits_medium):
    """Parametrized (K, omega) pair — follows n_qubits_medium fixture."""
    return _knm_for_size(n_qubits_medium)


# ---------------------------------------------------------------------------
# Coupling matrix variants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=["paper27", "ring", "zero", "identity"],
    ids=lambda v: f"K={v}",
)
def coupling_variant_4q(request):
    """4-qubit coupling matrix variants for testing coupling sensitivity."""
    n = 4
    omega = OMEGA_N_16[:n].copy()
    if request.param == "paper27":
        K = build_knm_paper27(L=n)
    elif request.param == "ring":
        K, _ = build_kuramoto_ring(n, coupling=1.0)
    elif request.param == "zero":
        K = np.zeros((n, n))
    elif request.param == "identity":
        K = np.ones((n, n)) - np.eye(n)
    return K, omega, request.param


# ---------------------------------------------------------------------------
# Hamiltonian fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hamiltonian_4q(knm_4q):
    K, omega = knm_4q
    return knm_to_hamiltonian(K, omega)


@pytest.fixture
def hamiltonian(knm):
    K, omega = knm
    return knm_to_hamiltonian(K, omega)


# ---------------------------------------------------------------------------
# Quantum state helpers
# ---------------------------------------------------------------------------


def random_statevector(n_qubits, rng=None):
    """Generate a random normalised statevector for n_qubits."""
    if rng is None:
        rng = np.random.default_rng()
    dim = 2**n_qubits
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def computational_basis_state(n_qubits, index=0):
    """Return |index> in the computational basis."""
    dim = 2**n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[index] = 1.0
    return psi


# ---------------------------------------------------------------------------
# Hardware runner
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    return runner


# ---------------------------------------------------------------------------
# Classical reference data
# ---------------------------------------------------------------------------

_REFERENCE_PATH = Path(__file__).parent.parent / "results" / "classical_16q_reference.json"


@pytest.fixture(scope="session")
def classical_reference():
    """Load pre-computed classical reference data (session-scoped for speed)."""
    if not _REFERENCE_PATH.exists():
        pytest.skip(f"Reference file not found: {_REFERENCE_PATH}")
    with open(_REFERENCE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def st_system_size(draw, min_n=2, max_n=8):
    """Draw a system size (number of qubits/oscillators)."""
    return draw(st.integers(min_value=min_n, max_value=max_n))


@st.composite
def st_coupling_matrix(draw, n=None, min_n=2, max_n=6):
    """Draw a symmetric non-negative coupling matrix."""
    if n is None:
        n = draw(st.integers(min_value=min_n, max_value=max_n))
    raw = draw(
        st.lists(
            st.lists(
                st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ),
            min_size=n,
            max_size=n,
        )
    )
    K = np.array(raw)
    K = (K + K.T) / 2.0
    np.fill_diagonal(K, 0.0)
    return K


@st.composite
def st_frequencies(draw, n=None, min_n=2, max_n=6):
    """Draw natural frequencies for n oscillators."""
    if n is None:
        n = draw(st.integers(min_value=min_n, max_value=max_n))
    freqs = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(freqs)


@st.composite
def st_angles(draw, n=None, min_n=2, max_n=8):
    """Draw n angles in [0, 2pi)."""
    if n is None:
        n = draw(st.integers(min_value=min_n, max_value=max_n))
    angles = draw(
        st.lists(
            st.floats(
                min_value=0.0, max_value=2 * np.pi - 1e-10, allow_nan=False, allow_infinity=False
            ),
            min_size=n,
            max_size=n,
        )
    )
    return np.array(angles)


@st.composite
def st_dt(draw, min_val=0.001, max_val=0.5):
    """Draw a time step."""
    return draw(
        st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False)
    )


@st.composite
def st_statevector(draw, n_qubits=None, min_n=2, max_n=4):
    """Draw a normalised random statevector."""
    if n_qubits is None:
        n_qubits = draw(st.integers(min_value=min_n, max_value=max_n))
    dim = 2**n_qubits
    real = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    imag = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    psi = np.array(real) + 1j * np.array(imag)
    norm = np.linalg.norm(psi)
    if norm < 1e-10:
        psi[0] = 1.0
        norm = 1.0
    return psi / norm, n_qubits
