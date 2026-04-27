# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Control StructuredAnsatz tests
"""Tests for the hardware-campaign StructuredAnsatz builder."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control import StructuredAnsatz


def _coupling_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.25],
            [0.0, 0.25, 0.0],
        ],
        dtype=np.float64,
    )


def test_from_kuramoto_builds_expected_scaled_trotter_circuit():
    ansatz = StructuredAnsatz.from_kuramoto(
        _coupling_matrix(),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        trotter_depth=2,
        time_step=0.25,
        coupling_scale=2.0,
    )

    circuit = ansatz.build_circuit()
    ops = circuit.count_ops()
    assert ops["h"] == 3
    assert ops["rz"] == 6
    assert ops["rzz"] == 4
    assert circuit.num_parameters == 0

    rzz_angles = [
        float(instruction.operation.params[0])
        for instruction in circuit.data
        if instruction.operation.name == "rzz"
    ]
    assert rzz_angles == pytest.approx([0.5, 0.25, 0.5, 0.25])


def test_lambda_fim_is_concrete_float_and_repeats_without_parameter_collision():
    ansatz = StructuredAnsatz.from_kuramoto(
        _coupling_matrix(),
        trotter_depth=3,
        time_step=0.1,
        lambda_fim=8.0,
    )

    circuit = ansatz.build_circuit()
    ops = circuit.count_ops()
    assert ops["rz"] == 9
    assert circuit.num_parameters == 0
    assert ansatz.params["lambda_fim"] == 8.0


def test_build_circuit_returns_copy():
    ansatz = StructuredAnsatz.from_kuramoto(_coupling_matrix(), trotter_depth=1)
    first = ansatz.build_circuit()
    second = ansatz.build_circuit()

    first.x(0)
    assert first.count_ops()["x"] == 1
    assert "x" not in second.count_ops()


def test_compatibility_kwargs_are_accepted_without_changing_core_circuit():
    ansatz = StructuredAnsatz.from_kuramoto(
        _coupling_matrix(),
        trotter_depth=1,
        informed_topology=True,
        non_hermitian_gain=0.25,
        mediated_couplings=True,
    )

    ops = ansatz.build_circuit().count_ops()
    assert ops["rzz"] == 2


def test_from_kuramoto_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="square"):
        StructuredAnsatz.from_kuramoto(np.ones((2, 3)))

    with pytest.raises(ValueError, match="omega shape"):
        StructuredAnsatz.from_kuramoto(np.zeros((3, 3)), omega=np.ones(2))

    invalid = np.zeros((2, 2), dtype=np.float64)
    invalid[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        StructuredAnsatz.from_kuramoto(invalid)
