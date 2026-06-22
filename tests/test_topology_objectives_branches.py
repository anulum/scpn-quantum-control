# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the topology-control objective
"""Guard and branch tests for the coupling-topology objective.

Covers the objective weight guards, the complete-uniform and disconnected
degeneracy classifications and the digest payload with a source matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.topology_control import (
    CouplingTopologyObjective,
    DegeneracyMode,
    NetworkCycleBackend,
    TopologyConstraintLedger,
)
from scpn_quantum_control.topology_control.objectives import (
    classify_degeneracy,
    objective_sha256_payload,
)


def _objective(**overrides: Any) -> CouplingTopologyObjective:
    kwargs: dict[str, Any] = {
        "ph_backend": NetworkCycleBackend(threshold=0.2),
        "ledger": TopologyConstraintLedger(),
        "source_matrix": np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64),
        "allow_approximate_ph_backend": True,
    }
    kwargs.update(overrides)
    return CouplingTopologyObjective(**kwargs)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("h1_weight", "h1_weight must be non-negative"),
        ("source_distance_weight", "source_distance_weight must be non-negative"),
        ("h1_target", "h1_target must be non-negative"),
    ],
)
def test_objective_rejects_negative_weights(field: str, match: str) -> None:
    """Each objective weight must be non-negative."""
    with pytest.raises(ValueError, match=match):
        _objective(**{field: -1.0})


def test_classify_degeneracy_complete_uniform() -> None:
    """A complete uniform graph is classified as complete-uniform."""
    k = np.array([[0.0, 0.3, 0.3], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]], dtype=np.float64)
    assert classify_degeneracy(k) is DegeneracyMode.COMPLETE_UNIFORM


def test_classify_degeneracy_disconnected() -> None:
    """A block-disconnected graph is classified as disconnected."""
    k = np.array(
        [
            [0.0, 0.3, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    assert classify_degeneracy(k) is DegeneracyMode.DISCONNECTED


def test_digest_payload_includes_source_shape() -> None:
    """The digest payload records the source-matrix shape when present."""
    payload = objective_sha256_payload(_objective())
    assert payload["source_shape"] == (2, 2)
