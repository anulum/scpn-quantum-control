# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the master stability function and multiplex decomposition
"""Module-specific tests for :mod:`multiplex_master_stability`.

The contracts: the master stability function returns the maximal transverse growth rate of a
coupling block; the multiplex decomposition reproduces the full multiplex synchronous-state spectrum
exactly from the per-structure eigenvalues (the ``LN`` eigenproblem replaced by an ``N`` + ``L`` one);
attractive coupling gives a stable synchronous state and repulsive inter-layer coupling an unstable
one; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.multiplex_kuramoto import multiplex_jacobian
from oscillatools.accel.multiplex_master_stability import (
    MultiplexSynchronisationStability,
    master_stability_function,
    multiplex_synchronisation_stability,
)

_L = 3
_N = 5


def _graphs(seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(_N, _N))
    intra = 0.5 * (raw + raw.T)
    np.fill_diagonal(intra, 0.0)
    raw_inter = rng.uniform(0.0, 1.0, size=(_L, _L))
    inter = 0.4 * (raw_inter + raw_inter.T)
    np.fill_diagonal(inter, 0.0)
    return intra, inter


def test_master_stability_function_returns_max_real_eigenvalue() -> None:
    node = np.zeros((2, 2), dtype=np.float64)
    coupling = np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float64)
    # spec(0 + alpha * diag(-1, -2)) = {-alpha, -2 alpha}; the max real part is -alpha
    assert master_stability_function(node, coupling, 2.0) == pytest.approx(-2.0)
    assert master_stability_function(node, coupling, -1.0) == pytest.approx(2.0)


def test_decomposition_reproduces_the_full_multiplex_spectrum() -> None:
    intra, inter = _graphs()
    result = multiplex_synchronisation_stability(intra, inter)
    assert isinstance(result, MultiplexSynchronisationStability)
    assert result.spectrum.shape == (_L * _N,)

    full = multiplex_jacobian(
        np.zeros((_L, _N)), np.zeros((_L, _N)), np.stack([intra] * _L), inter
    )
    full_spectrum = np.sort(np.linalg.eigvals(full).real)
    decomposed_spectrum = np.sort(result.spectrum.real)
    assert decomposed_spectrum == pytest.approx(full_spectrum, abs=1e-9)


def test_attractive_coupling_is_stable_repulsive_is_not() -> None:
    intra, inter = _graphs(1)
    attractive = multiplex_synchronisation_stability(intra, inter)
    assert attractive.is_stable
    assert attractive.transverse_decay < 0.0
    repulsive = multiplex_synchronisation_stability(intra, -inter)
    assert not repulsive.is_stable
    assert repulsive.transverse_decay > 0.0


def test_transverse_decay_drops_only_the_global_zero_mode() -> None:
    intra, inter = _graphs(2)
    result = multiplex_synchronisation_stability(intra, inter)
    ordered = np.sort(result.spectrum.real)
    # the largest eigenvalue is the global phase-shift zero mode
    assert ordered[-1] == pytest.approx(0.0, abs=1e-9)
    # the transverse decay is the next eigenvalue down
    assert result.transverse_decay == pytest.approx(ordered[-2])


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("msf", {"node_jacobian": np.zeros((2, 3))}, "node_jacobian must be a non-empty square"),
        ("msf", {"node_jacobian": np.zeros(4)}, "node_jacobian must be a non-empty square"),
        ("msf", {"coupling_jacobian": np.zeros((3, 3))}, "coupling_jacobian must have shape"),
        ("multiplex", {"intra_coupling": np.zeros((2, 3))}, "intra_coupling must be an"),
        ("multiplex", {"inter_coupling": np.zeros((1, 1))}, "inter_coupling must be an"),
        ("multiplex", {"intra_coupling": np.full((_N, _N), np.nan)}, "must be finite"),
        ("multiplex", {"stability_tolerance": 0.0}, "stability_tolerance must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    intra, inter = _graphs()
    with pytest.raises(ValueError, match=message):
        if call == "msf":
            args: dict[str, Any] = {
                "node_jacobian": np.zeros((2, 2)),
                "coupling_jacobian": np.zeros((2, 2)),
            }
            args.update(kwargs)
            master_stability_function(args["node_jacobian"], args["coupling_jacobian"], 1.0)
        else:
            args = {"intra_coupling": intra, "inter_coupling": inter, "stability_tolerance": 1e-9}
            args.update(kwargs)
            multiplex_synchronisation_stability(
                args["intra_coupling"],
                args["inter_coupling"],
                stability_tolerance=args["stability_tolerance"],
            )
