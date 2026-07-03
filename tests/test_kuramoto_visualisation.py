# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Kuramoto matplotlib visualisation layer
r"""Tests for the Kuramoto matplotlib visualisation layer.

Assert on the matplotlib artists each renderer produces (images, lines, collections, labels and colour
limits) rather than on rendered pixels, so the tests are deterministic and need no display. Cover the
axis-supplied and axis-created branches, the default and explicit sample-time branches, the
single-timestep raster extent, the with-edges and no-edges network embedding, the anchoring of the
order-parameter curve to the shipped ``order_parameter`` surface, every input-validation error, and the
helpful error raised when matplotlib is absent.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scpn_quantum_control.accel import kuramoto_visualisation as viz
from scpn_quantum_control.accel.order_parameter_observables import order_parameter


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    """Close every figure a test opens so a long session does not accumulate open figures."""
    yield
    plt.close("all")


def _trajectory(steps: int = 24, count: int = 6, seed: int = 20260703) -> NDArray[np.float64]:
    """Return a reproducible ``(steps, count)`` phase trajectory in radians."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (steps, count))


def test_phase_raster_creates_figure_image_and_labels() -> None:
    phases = _trajectory()
    axis = viz.phase_raster(phases)
    assert len(axis.images) == 1
    image = axis.images[0]
    # The raster is transposed to oscillator-by-time and clamped to the cyclic phase range.
    assert image.get_array().shape == (phases.shape[1], phases.shape[0])
    assert image.get_clim() == pytest.approx((-np.pi, np.pi))
    assert axis.get_xlabel() == "time"
    assert axis.get_ylabel() == "oscillator index"
    # The colour bar adds a second axis to the created figure.
    assert len(axis.figure.axes) == 2


def test_phase_raster_draws_into_a_supplied_axis() -> None:
    phases = _trajectory()
    _, supplied = plt.subplots()
    returned = viz.phase_raster(phases, ax=supplied)
    assert returned is supplied
    assert len(supplied.images) == 1


def test_phase_raster_single_timestep_widens_the_extent() -> None:
    phases = _trajectory(steps=1)
    axis = viz.phase_raster(phases)
    left, right, _, _ = axis.images[0].get_extent()
    assert right == pytest.approx(left + 1.0)


def test_phase_raster_uses_supplied_times_for_the_extent() -> None:
    phases = _trajectory(steps=10)
    times = np.linspace(2.0, 6.5, 10)
    axis = viz.phase_raster(phases, times=times)
    left, right, _, _ = axis.images[0].get_extent()
    assert left == pytest.approx(2.0)
    assert right == pytest.approx(6.5)


def test_order_parameter_timeseries_matches_the_shipped_order_parameter() -> None:
    phases = _trajectory()
    axis = viz.order_parameter_timeseries(phases)
    assert len(axis.lines) == 1
    plotted = axis.lines[0].get_ydata()
    expected = np.array([order_parameter(row) for row in phases])
    np.testing.assert_allclose(plotted, expected, atol=1e-12)
    assert axis.get_ylim() == pytest.approx((0.0, 1.05))
    assert axis.get_ylabel() == "order parameter $r$"


def test_order_parameter_timeseries_draws_into_a_supplied_axis() -> None:
    phases = _trajectory()
    _, supplied = plt.subplots()
    returned = viz.order_parameter_timeseries(phases, ax=supplied)
    assert returned is supplied
    assert len(supplied.lines) == 1


def test_chimera_snapshot_draws_one_curve_per_community_with_the_index_title() -> None:
    phases = _trajectory(count=6)
    communities = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    axis = viz.chimera_snapshot(phases, communities)
    assert len(axis.lines) == 2
    assert "chimera index" in axis.get_title()


def test_chimera_snapshot_curves_are_the_shipped_community_order_parameters() -> None:
    from scpn_quantum_control.accel.kuramoto_chimera import community_order_parameters

    phases = _trajectory(count=6)
    communities = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    axis = viz.chimera_snapshot(phases, communities)
    expected = community_order_parameters(phases, communities)
    for community, line in enumerate(axis.lines):
        np.testing.assert_allclose(line.get_ydata(), expected[:, community], atol=1e-12)


def test_network_phase_embedding_draws_edges_and_nodes() -> None:
    rng = np.random.default_rng(1)
    phases = rng.uniform(-np.pi, np.pi, 5)
    coupling = rng.uniform(0.3, 0.9, (5, 5))
    np.fill_diagonal(coupling, 0.0)
    axis = viz.network_phase_embedding(phases, coupling)
    kinds = [type(collection).__name__ for collection in axis.collections]
    assert "LineCollection" in kinds  # the coupling edges
    assert "PathCollection" in kinds  # the phase-coloured nodes
    assert axis.get_aspect() == 1.0


def test_network_phase_embedding_without_coupling_omits_the_edge_collection() -> None:
    phases = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    coupling = np.zeros((4, 4))
    axis = viz.network_phase_embedding(phases, coupling)
    kinds = [type(collection).__name__ for collection in axis.collections]
    assert "LineCollection" not in kinds
    assert kinds.count("PathCollection") == 1


def test_network_phase_embedding_draws_into_a_supplied_axis() -> None:
    phases = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    coupling = np.eye(4) * 0.0 + 0.5
    np.fill_diagonal(coupling, 0.0)
    _, supplied = plt.subplots()
    returned = viz.network_phase_embedding(phases, coupling, ax=supplied)
    assert returned is supplied


def test_phase_raster_rejects_a_non_two_dimensional_trajectory() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        viz.phase_raster(np.zeros(5))


def test_phase_raster_rejects_an_empty_trajectory() -> None:
    with pytest.raises(ValueError, match="at least one step and one oscillator"):
        viz.phase_raster(np.zeros((0, 4)))


def test_order_parameter_timeseries_rejects_mismatched_times() -> None:
    with pytest.raises(ValueError, match="times must have shape"):
        viz.order_parameter_timeseries(_trajectory(steps=10), times=np.zeros(9))


def test_network_phase_embedding_rejects_a_non_one_dimensional_snapshot() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        viz.network_phase_embedding(np.zeros((3, 3)), np.zeros((3, 3)))


def test_network_phase_embedding_rejects_a_mismatched_coupling_shape() -> None:
    with pytest.raises(ValueError, match="coupling must have shape"):
        viz.network_phase_embedding(np.zeros(4), np.zeros((3, 3)))


def test_missing_matplotlib_raises_a_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import: Callable[..., object] = builtins.__import__

    def _fail_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("matplotlib"):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_matplotlib)
    with pytest.raises(ImportError, match=r"\[viz\]"):
        viz._require_pyplot()
