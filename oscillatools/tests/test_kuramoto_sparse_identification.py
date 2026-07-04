# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for sparse identification of phase dynamics
"""Module-specific tests for :mod:`kuramoto_sparse_identification`.

The contracts: sequentially-thresholded least squares discovers the exact sparse structure of a
directed Kuramoto network — recovering the coupling and frequencies exactly while driving every absent
coupling, every cosine term and every higher harmonic to hard zero; it discovers the Sakaguchi
(sine+cosine) and biharmonic (second-harmonic) functional forms; the threshold rejects a sub-threshold
coupling; the discovered model evaluates its own field; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_sparse_identification import (
    SparseDynamicsModel,
    discover_phase_dynamics,
)

_N = 5
_OMEGA = np.array([0.4, -0.5, 0.6, -0.7, 0.3])  # all above the test threshold


def _coupling() -> NDArray[np.float64]:
    coupling = np.zeros((_N, _N), dtype=np.float64)
    coupling[0, 1] = 0.8
    coupling[1, 2] = 0.6
    coupling[2, 3] = 0.9
    coupling[3, 4] = 0.5
    coupling[4, 0] = 0.7
    return coupling


def _snapshots(seed: int = 0, n: int = 120) -> NDArray[np.float64]:
    return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=(n, _N))


def _kuramoto_derivatives(
    snapshots: NDArray[np.float64],
    coupling: NDArray[np.float64],
    shift: float = 0.0,
    harmonic: int = 1,
) -> NDArray[np.float64]:
    derivatives = np.empty_like(snapshots)
    for sample in range(snapshots.shape[0]):
        difference = snapshots[sample][None, :] - snapshots[sample][:, None]
        derivatives[sample] = _OMEGA + np.sum(
            coupling * np.sin(harmonic * difference - shift), axis=1
        )
    return derivatives


def test_discovers_sparse_kuramoto_structure_exactly() -> None:
    coupling = _coupling()
    snapshots = _snapshots()
    derivatives = _kuramoto_derivatives(snapshots, coupling)
    model = discover_phase_dynamics(snapshots, derivatives, n_harmonics=2, threshold=0.05)

    assert isinstance(model, SparseDynamicsModel)
    assert model.sine_coupling[0] == pytest.approx(coupling, abs=1e-6)
    assert model.frequencies == pytest.approx(_OMEGA, abs=1e-6)
    # the discovery: absent couplings, all cosine terms and all second harmonics are hard zero
    assert np.all(model.sine_coupling[0][coupling == 0.0] == 0.0)
    assert np.all(model.cosine_coupling == 0.0)
    assert np.all(model.sine_coupling[1] == 0.0)
    # exactly the true number of active terms (5 couplings + 5 frequencies)
    assert model.active_terms == 10
    assert model.residual < 1e-9


def test_discovered_model_reproduces_the_field() -> None:
    snapshots = _snapshots(seed=1)
    coupling = _coupling()
    derivatives = _kuramoto_derivatives(snapshots, coupling)
    model = discover_phase_dynamics(snapshots, derivatives, n_harmonics=2, threshold=0.05)
    for sample in range(snapshots.shape[0]):
        assert model.field(snapshots[sample]) == pytest.approx(derivatives[sample], abs=1e-6)


def test_discovers_sakaguchi_functional_form() -> None:
    beta = 0.4
    coupling = np.zeros((_N, _N), dtype=np.float64)
    coupling[0, 1] = 1.0
    snapshots = _snapshots(seed=2)
    derivatives = _kuramoto_derivatives(snapshots, coupling, shift=beta)
    model = discover_phase_dynamics(snapshots, derivatives, n_harmonics=2, threshold=0.05)
    # sin(d - beta) = cos(beta) sin d - sin(beta) cos d → a paired sine/cosine first harmonic
    assert model.sine_coupling[0, 0, 1] == pytest.approx(np.cos(beta), abs=1e-6)
    assert model.cosine_coupling[0, 0, 1] == pytest.approx(-np.sin(beta), abs=1e-6)
    assert np.all(model.sine_coupling[1] == 0.0)


def test_discovers_biharmonic_coupling() -> None:
    coupling = np.zeros((_N, _N), dtype=np.float64)
    coupling[0, 1] = 0.7
    snapshots = _snapshots(seed=3)
    # a pure second-harmonic (Hansel–Daido) coupling sin(2(θ_j − θ_i))
    derivatives = _kuramoto_derivatives(snapshots, coupling, harmonic=2)
    model = discover_phase_dynamics(snapshots, derivatives, n_harmonics=2, threshold=0.05)
    assert model.sine_coupling[1, 0, 1] == pytest.approx(0.7, abs=1e-6)
    assert np.all(model.sine_coupling[0] == 0.0)  # the first harmonic is absent


def test_threshold_rejects_subthreshold_coupling() -> None:
    coupling = _coupling()
    coupling[0, 1] = 0.02  # below the 0.05 threshold → must be discovered as absent
    snapshots = _snapshots(seed=4)
    derivatives = _kuramoto_derivatives(snapshots, coupling)
    model = discover_phase_dynamics(snapshots, derivatives, n_harmonics=2, threshold=0.05)
    assert model.sine_coupling[0, 0, 1] == 0.0
    # the supra-threshold couplings survive
    assert model.sine_coupling[0, 2, 3] == pytest.approx(0.9, abs=1e-2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"phases": np.zeros((10, 1))}, "phases must be a"),
        ({"phases": np.zeros(10)}, "phases must be a"),
        ({"derivatives": np.zeros((10, 4))}, "derivatives must match"),
        ({"n_harmonics": 0}, "n_harmonics must be positive"),
        ({"threshold": 0.0}, "threshold must be positive"),
        ({"max_iterations": 0}, "max_iterations must be positive"),
    ],
)
def test_validation_errors(mutation: dict[str, Any], message: str) -> None:
    snapshots = _snapshots(n=20)
    call: dict[str, Any] = {
        "phases": snapshots,
        "derivatives": _kuramoto_derivatives(snapshots, _coupling()),
        "n_harmonics": 2,
        "threshold": 0.05,
        "max_iterations": 10,
    }
    call.update(mutation)
    with pytest.raises(ValueError, match=message):
        discover_phase_dynamics(
            call["phases"],
            call["derivatives"],
            n_harmonics=call["n_harmonics"],
            threshold=call["threshold"],
            max_iterations=call["max_iterations"],
        )
