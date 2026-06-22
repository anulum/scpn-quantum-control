# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the power-grid benchmark
"""Guard tests for the power-grid functional-coupling benchmark.

Covers the IEEE 14-bus reactance integrity guard, the built-in reference refusal,
the frequency-vector finiteness guard and the reference source-mode guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications import power_grid as pg
from scpn_quantum_control.applications.power_grid import (
    _validated_frequency_vector,
    ieee_14bus_admittance_coupling_matrix,
    ieee_14bus_susceptance_matrix,
    power_grid_benchmark,
)


def test_susceptance_matrix_rejects_non_positive_reactance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-positive branch reactance breaks the susceptance integrity guard."""
    monkeypatch.setattr(pg, "IEEE_14BUS_BRANCH_REACTANCE", ((1, 2, -0.5),))
    with pytest.raises(ValueError, match="branch reactance must be positive"):
        ieee_14bus_susceptance_matrix()


def test_admittance_refuses_builtin_without_opt_in() -> None:
    """The built-in IEEE 14-bus reference is refused without explicit opt-in."""
    with pytest.raises(RuntimeError, match="Refusing built-in IEEE 14-bus reference"):
        ieee_14bus_admittance_coupling_matrix()


def test_frequency_vector_rejects_non_finite() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _validated_frequency_vector(np.array([0.1, np.inf], dtype=np.float64), 2, "freqs", "K")


def test_susceptance_matrix_builds_from_real_data() -> None:
    """The IEEE 14-bus susceptance matrix builds from the packaged reactances."""
    assert ieee_14bus_susceptance_matrix().shape == (14, 14)


def test_admittance_builtin_with_opt_in() -> None:
    """With explicit opt-in the built-in IEEE 14-bus admittance matrix is returned."""
    coupling, omega = ieee_14bus_admittance_coupling_matrix(allow_builtin_reference=True)
    assert coupling.shape[0] == omega.shape[0]


def test_benchmark_rejects_unknown_grid() -> None:
    """An unknown built-in grid name is rejected."""
    with pytest.raises(ValueError, match="Unknown grid"):
        power_grid_benchmark(
            np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), grid_name="UNKNOWN"
        )


def test_benchmark_small_comparison_zeroes_correlations() -> None:
    """A two-node comparison cannot estimate correlations and reports zero."""
    eye = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    result = power_grid_benchmark(
        eye,
        np.array([0.1, 0.2], dtype=np.float64),
        grid_coupling=eye,
        grid_frequencies=np.array([0.3, 0.4], dtype=np.float64),
        reference_source_mode="curated",
    )
    assert result.topology_correlation == 0.0
    assert result.frequency_correlation == 0.0


def test_benchmark_rejects_unknown_reference_source_mode() -> None:
    """An unknown reference source mode is rejected for measured inputs."""
    eye = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="reference_source_mode must be one of"):
        power_grid_benchmark(
            eye,
            np.zeros(2, dtype=np.float64),
            grid_coupling=eye,
            grid_frequencies=np.zeros(2, dtype=np.float64),
            reference_source_mode="bogus_mode",
        )
