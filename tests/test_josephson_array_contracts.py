# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Josephson-array coupling-matrix contracts
"""Contract tests for the Josephson-array coupling-matrix builder.

Covers the provenance and positivity guards, the measured-vs-illustrative
topology gate, and the explicit coupling-edge path including its index and
non-negativity validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.josephson_array import (
    JosephsonArrayParameters,
    jja_coupling_matrix,
    josephson_benchmark,
)


def _params(
    ej: float = 15.0, ec: float = 0.25, coupling: float = 0.015
) -> JosephsonArrayParameters:
    """Build Josephson-array parameters with an explicit test provenance."""
    return JosephsonArrayParameters(
        ej_ghz=ej, ec_ghz=ec, coupling_ghz=coupling, parameter_source="unit-test"
    )


def test_requires_parameters() -> None:
    """A None parameter set is rejected (provenance is mandatory)."""
    with pytest.raises(ValueError, match="requires JosephsonArrayParameters"):
        jja_coupling_matrix(3)


@pytest.mark.parametrize("bad", [_params(ej=0.0), _params(ec=0.0), _params(coupling=0.0)])
def test_rejects_non_positive_ghz_parameters(bad: JosephsonArrayParameters) -> None:
    """Non-positive GHz parameters are rejected."""
    with pytest.raises(ValueError, match="positive GHz"):
        jja_coupling_matrix(3, parameters=bad, allow_illustrative_topology=True)


def test_measured_topology_requires_coupling_edges() -> None:
    """Without explicit edges, an illustrative topology must be opted into."""
    with pytest.raises(ValueError, match="Measured topology requires coupling_edges"):
        jja_coupling_matrix(3, parameters=_params())


def test_explicit_coupling_edges_build_symmetric_matrix() -> None:
    """Explicit edges populate a symmetric coupling matrix and uniform frequencies."""
    K, omega = jja_coupling_matrix(
        3, parameters=_params(), coupling_edges=[(0, 1, 0.4), (1, 2, 0.6)]
    )
    assert K[0, 1] == pytest.approx(0.4)
    assert K[1, 0] == pytest.approx(0.4)
    assert K[1, 2] == pytest.approx(0.6)
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(omega, _params().ec_ghz)


@pytest.mark.parametrize("edge", [(0, 0, 0.3), (0, 5, 0.3), (-1, 1, 0.3)])
def test_rejects_invalid_coupling_edge(edge: tuple[int, int, float]) -> None:
    """Self-loops and out-of-range indices are rejected."""
    with pytest.raises(ValueError, match="Invalid coupling edge"):
        jja_coupling_matrix(3, parameters=_params(), coupling_edges=[edge])


def test_rejects_negative_coupling_strength() -> None:
    """Negative coupling strengths are rejected."""
    with pytest.raises(ValueError, match="Coupling strengths must be non-negative"):
        jja_coupling_matrix(3, parameters=_params(), coupling_edges=[(0, 1, -0.1)])


def test_rejects_unknown_topology() -> None:
    """An unrecognised illustrative topology is rejected."""
    with pytest.raises(ValueError, match="Unknown topology"):
        jja_coupling_matrix(
            3, parameters=_params(), topology="bogus", allow_illustrative_topology=True
        )


def test_benchmark_reports_zero_correlation_for_too_few_pairs() -> None:
    """With fewer than three coupled pairs the topology correlation is reported as zero."""
    k_scpn = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
    omega_scpn = np.array([1.0, 2.0], dtype=np.float64)
    result = josephson_benchmark(
        k_scpn,
        omega_scpn,
        parameters=_params(),
        coupling_edges=[(0, 1, 0.5)],
    )
    assert result.topology_correlation == 0.0
    assert result.frequency_correlation == 0.0
    assert result.n_junctions == 2
