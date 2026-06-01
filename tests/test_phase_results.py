# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- phase result contract tests
"""Module-specific tests for typed phase-dynamics result objects."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from scpn_quantum_control.phase.results import TrajectoryResult


def test_trajectory_result_preserves_legacy_mapping_contract_and_metadata() -> None:
    """TrajectoryResult should expose immutable arrays through the legacy mapping API."""

    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    order_parameter = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    result = TrajectoryResult(times=times, R=order_parameter, metadata={"source": "unit"})

    assert len(result) == 2
    assert list(result) == ["times", "R"]
    np.testing.assert_allclose(result["times"], times)
    np.testing.assert_allclose(result["R"], order_parameter)
    legacy = result.to_dict()
    assert list(legacy) == ["times", "R"]
    assert legacy["times"] is result.times
    assert legacy["R"] is result.R
    assert isinstance(result.metadata, MappingProxyType)
    assert result.metadata["source"] == "unit"


def test_trajectory_result_copies_inputs_and_exposes_read_only_arrays() -> None:
    """Input mutation after construction must not affect stored phase trajectories."""

    times = np.array([0.0, 1.0], dtype=np.float64)
    order_parameter = np.array([0.1, 0.9], dtype=np.float64)
    result = TrajectoryResult(times=times, R=order_parameter)

    times[0] = 99.0
    order_parameter[1] = 99.0

    np.testing.assert_allclose(result.times, np.array([0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(result.R, np.array([0.1, 0.9], dtype=np.float64))
    with pytest.raises(ValueError, match="read-only"):
        result.times[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        result.R[0] = 1.0
    with pytest.raises(TypeError):
        result.metadata["new"] = "forbidden"  # type: ignore[index]


@pytest.mark.parametrize(
    ("times", "order_parameter", "message"),
    [
        (np.array([[0.0, 1.0]], dtype=np.float64), np.array([0.1, 0.9]), "times"),
        (np.array([0.0, 1.0], dtype=np.float64), np.array([[0.1, 0.9]]), "R"),
        (np.array([0.0, 1.0], dtype=np.float64), np.array([0.1]), "identical shape"),
        (np.array([0.0, np.inf], dtype=np.float64), np.array([0.1, 0.9]), "times"),
        (np.array([0.0, 1.0], dtype=np.float64), np.array([0.1, np.nan]), "R"),
    ],
)
def test_trajectory_result_rejects_invalid_phase_trajectory_inputs(
    times: np.ndarray,
    order_parameter: np.ndarray,
    message: str,
) -> None:
    """Typed phase results should fail closed on malformed trajectory arrays."""

    with pytest.raises(ValueError, match=message):
        TrajectoryResult(times=times, R=order_parameter)


def test_trajectory_result_rejects_unknown_legacy_mapping_keys() -> None:
    """Legacy mapping compatibility should remain limited to documented keys."""

    result = TrajectoryResult(
        times=np.array([0.0], dtype=np.float64),
        R=np.array([1.0], dtype=np.float64),
    )

    with pytest.raises(KeyError, match="amplitude"):
        _ = result["amplitude"]
