# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the gauge/lattice confinement cross-check
"""Tests for the quantum-vs-lattice confinement cross-check surface."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.gauge.lattice_crosscheck import (
    GaugeLatticeCrosscheck,
    crosscheck_confinement_on_lattice,
)


def _ring_with_chord(n: int = 4) -> NDArray[np.float64]:
    K = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = 0.7
    K[0, 2] = K[2, 0] = 0.4
    return K


_OMEGA4 = np.array([0.15, -0.1, 0.05, -0.02])


def test_crosscheck_reports_both_routes_on_a_plaquette_topology() -> None:
    result = crosscheck_confinement_on_lattice(
        _ring_with_chord(), _OMEGA4, beta=2.0, n_thermalisation=60, seed=7
    )

    assert isinstance(result, GaugeLatticeCrosscheck)
    assert result.quantum.n_qubits == 4
    assert result.lattice_n_plaquettes == 2
    assert result.lattice_string_tension is not None
    assert result.lattice_string_tension > 0.0
    assert 0.0 <= result.hmc_acceptance_rate <= 1.0
    assert np.isfinite(result.lattice_topological_charge)
    assert 0.0 <= result.lattice_average_link_magnitude <= 1.0
    assert result.beta == 2.0
    assert result.n_thermalisation_steps == 60


def test_crosscheck_is_reproducible_with_a_seed() -> None:
    first = crosscheck_confinement_on_lattice(
        _ring_with_chord(), _OMEGA4, n_thermalisation=30, seed=11
    )
    second = crosscheck_confinement_on_lattice(
        _ring_with_chord(), _OMEGA4, n_thermalisation=30, seed=11
    )
    assert first.lattice_string_tension == second.lattice_string_tension
    assert first.lattice_topological_charge == second.lattice_topological_charge
    assert first.hmc_acceptance_rate == second.hmc_acceptance_rate


def test_tree_topology_has_no_plaquettes_and_reports_none_tension() -> None:
    """A plaquette-free graph must not fabricate a lattice string tension."""
    K = np.zeros((4, 4))
    for i in range(3):
        K[i, i + 1] = K[i + 1, i] = 0.6
    result = crosscheck_confinement_on_lattice(K, _OMEGA4, n_thermalisation=10, seed=3)

    assert result.lattice_n_plaquettes == 0
    assert result.lattice_string_tension is None
    assert result.both_tensions_available is False


def test_both_tensions_available_requires_both_routes() -> None:
    result = crosscheck_confinement_on_lattice(
        _ring_with_chord(), _OMEGA4, beta=2.0, n_thermalisation=30, seed=7
    )
    expected = (
        result.quantum.string_tension is not None and result.lattice_string_tension is not None
    )
    assert result.both_tensions_available is expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta": 0.0}, "beta"),
        ({"beta": -1.0}, "beta"),
        ({"n_thermalisation": 0}, "n_thermalisation"),
        ({"n_leapfrog": 0}, "n_leapfrog"),
        ({"step_size": 0.0}, "step_size"),
    ],
)
def test_sampling_parameters_fail_closed(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        crosscheck_confinement_on_lattice(_ring_with_chord(), _OMEGA4, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("K", "omega", "match"),
    [
        (np.zeros((2, 3)), np.zeros(2), "square"),
        (np.array([[0.0, 1.0], [0.5, 0.0]]), np.zeros(2), "symmetric"),
        (np.zeros((3, 3)), np.zeros(2), "omega shape"),
    ],
)
def test_malformed_topology_fails_closed(
    K: NDArray[np.float64], omega: NDArray[np.float64], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        crosscheck_confinement_on_lattice(K, omega)


def test_gauge_subpackage_exports_the_crosscheck() -> None:
    from scpn_quantum_control import gauge

    assert hasattr(gauge, "crosscheck_confinement_on_lattice")
    assert hasattr(gauge, "GaugeLatticeCrosscheck")
    assert "crosscheck_confinement_on_lattice" in gauge.__all__
    assert "GaugeLatticeCrosscheck" in gauge.__all__
