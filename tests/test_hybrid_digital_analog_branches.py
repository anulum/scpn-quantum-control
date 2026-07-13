# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the hybrid digital-analog backend
"""Branch and guard tests for the hybrid digital-analog compilation backend.

Covers the empty-partition analog fraction, the programme serialiser, the
trotter-order guard, the native-engine partition fallback and the coupling
matrix shape/finiteness/symmetry guards.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.hardware.analog_kuramoto import AnalogKuramotoPlatform
from scpn_quantum_control.hardware.hybrid_digital_analog import (
    HybridRoute,
    compile_hybrid_digital_analog,
    partition_kuramoto_couplings,
)

_K = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
_OMEGA = np.array([0.2, 0.5], dtype=np.float64)


def test_analog_fraction_zero_without_assignments() -> None:
    """A coupling-free matrix yields an empty partition with zero analog fraction."""
    partition = partition_kuramoto_couplings(np.zeros((2, 2), dtype=np.float64))
    assert partition.assignments == ()
    assert partition.analog_fraction == 0.0


def test_compiled_program_serialises_to_dict() -> None:
    """A compiled hybrid programme exposes a JSON-serialisable dictionary."""
    program = compile_hybrid_digital_analog(
        _K, _OMEGA, platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS, duration=1.0
    )
    payload = program.to_dict()
    assert payload["platform"] == AnalogKuramotoPlatform.NEUTRAL_ATOMS.value
    assert payload["duration"] == 1.0
    assert "partition" in payload
    assert "analog_program" in payload


def test_compile_rejects_non_positive_trotter_order() -> None:
    """A trotter order below one is rejected."""
    with pytest.raises(ValueError, match="trotter_order must be at least 1"):
        compile_hybrid_digital_analog(
            _K,
            _OMEGA,
            platform=AnalogKuramotoPlatform.NEUTRAL_ATOMS,
            duration=1.0,
            trotter_order=0,
        )


def test_partition_falls_back_to_numpy_on_engine_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising native partition kernel falls back to the NumPy partition."""

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("engine refused the partition")

    stub = types.ModuleType("scpn_quantum_engine")
    stub.hybrid_coupling_partition = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    partition = partition_kuramoto_couplings(_K)
    assert partition.n_couplings == 1
    assert partition.n_analog_couplings + partition.n_digital_couplings == 1


def test_partition_falls_back_when_native_export_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the NumPy partition when the installed engine lacks the native export."""
    stub = types.ModuleType("scpn_quantum_engine")
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    partition = partition_kuramoto_couplings(_K, max_analog_couplers=0)

    np.testing.assert_array_equal(partition.analog_K_nm, np.zeros_like(_K))
    np.testing.assert_array_equal(partition.digital_K_nm, _K)
    assert partition.assignments[0].route is HybridRoute.DIGITAL


def test_partition_rejects_non_square_matrix() -> None:
    """A non-square coupling matrix is rejected."""
    with pytest.raises(ValueError, match="K_nm must be a square matrix"):
        partition_kuramoto_couplings(np.zeros((2, 3), dtype=np.float64))


def test_partition_rejects_non_finite_matrix() -> None:
    """A non-finite coupling matrix is rejected."""
    matrix = np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="K_nm must contain only finite values"):
        partition_kuramoto_couplings(matrix)


def test_partition_rejects_asymmetric_matrix() -> None:
    """An asymmetric coupling matrix is rejected for hybrid splitting."""
    matrix = np.array([[0.0, 0.5], [0.1, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="K_nm must be symmetric for hybrid splitting"):
        partition_kuramoto_couplings(matrix)
