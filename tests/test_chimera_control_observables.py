# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multiscale observable tests
"""Production-surface tests for hierarchy-aware order parameters."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.chimera_control.observables import (
    LevelOrderParameterSummary,
    MultiscaleOrderParameterReport,
    measure_multiscale_order_parameters,
)
from scpn_quantum_control.chimera_control.schema import two_population_hierarchy


def test_multiscale_measurement_resolves_partial_and_global_synchrony() -> None:
    """Resolve partial population coherence and ensemble coherence separately."""
    hierarchy = two_population_hierarchy(2)
    phases = np.array(
        [
            [0.0, 0.0, 0.0, np.pi],
            [0.1, 0.1, 0.0, np.pi],
            [0.2, 0.2, np.pi / 2.0, -np.pi / 2.0],
        ]
    )
    report = measure_multiscale_order_parameters(phases, hierarchy)

    population = report.level("population")
    ensemble = report.level("ensemble")
    np.testing.assert_allclose(population.mean_by_community[0], 1.0)
    assert population.mean_by_community[1] < 0.01
    assert population.chimera_index > 0.24
    assert ensemble.chimera_index == 0.0
    np.testing.assert_allclose(
        ensemble.community_order_parameters[:, 0],
        report.global_order_parameter,
    )
    assert (
        report.content_digest
        == measure_multiscale_order_parameters(phases, hierarchy).content_digest
    )
    assert not report.global_order_parameter.flags.writeable
    assert not population.community_order_parameters.flags.writeable
    assert not population.mean_by_community.flags.writeable
    with pytest.raises(KeyError, match="unknown report level"):
        report.level("missing")


def test_measurement_rejects_wrong_rank_width_and_non_finite_phases() -> None:
    """Reject phase arrays with invalid rank, width, or finite-value custody."""
    hierarchy = two_population_hierarchy(2)
    with pytest.raises(ValueError, match="2-dimensional"):
        measure_multiscale_order_parameters(np.zeros(4), hierarchy)
    with pytest.raises(ValueError, match="node width"):
        measure_multiscale_order_parameters(np.zeros((2, 3)), hierarchy)
    bad = np.zeros((2, 4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        measure_multiscale_order_parameters(bad, hierarchy)


def test_level_summary_contract_rejects_invalid_shapes_and_scalars() -> None:
    """Reject level summaries with inconsistent shapes or invalid diagnostics."""
    values = np.ones((2, 2))
    with pytest.raises(ValueError, match="level_name"):
        LevelOrderParameterSummary(" ", values, np.ones(2), 0.0, 0.0)
    with pytest.raises(ValueError, match="mean_by_community shape"):
        LevelOrderParameterSummary("fine", values, np.ones(1), 0.0, 0.0)
    with pytest.raises(ValueError, match="chimera_index"):
        LevelOrderParameterSummary("fine", values, np.ones(2), -1.0, 0.0)
    with pytest.raises(ValueError, match="community_metastability"):
        LevelOrderParameterSummary("fine", values, np.ones(2), 0.0, np.nan)


def test_report_contract_rejects_misaligned_levels_time_and_digest() -> None:
    """Reject reports whose scales, time axis, or digest are inconsistent."""
    hierarchy = two_population_hierarchy(2)
    phases = np.zeros((2, 4))
    valid = measure_multiscale_order_parameters(phases, hierarchy)
    with pytest.raises(ValueError, match="hierarchy order"):
        MultiscaleOrderParameterReport(
            hierarchy,
            valid.global_order_parameter,
            tuple(reversed(valid.levels)),
            valid.content_digest,
        )
    short = LevelOrderParameterSummary("population", np.ones((1, 2)), np.ones(2), 0.0, 0.0)
    with pytest.raises(ValueError, match="global time length"):
        MultiscaleOrderParameterReport(
            hierarchy,
            valid.global_order_parameter,
            (short, valid.level("ensemble")),
            valid.content_digest,
        )
    with pytest.raises(ValueError, match="content_digest"):
        MultiscaleOrderParameterReport(
            hierarchy,
            valid.global_order_parameter,
            valid.levels,
            "bad",
        )
