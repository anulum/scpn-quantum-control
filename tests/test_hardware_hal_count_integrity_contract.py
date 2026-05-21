# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- strict count integrity contract tests
"""Contract tests for strict integer count handling in HAL adapters."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware import (
    hal_azure,
    hal_braket,
    hal_cirq,
    hal_dwave,
    hal_iqm,
    hal_oqc,
    hal_pasqal,
    hal_qbraid,
    hal_quandela,
    hal_quantinuum,
    hal_quera_bloqade,
    hal_strangeworks,
)
from scpn_quantum_control.hardware._count_integrity import strict_shot_conservation


def test_count_normalisers_reject_fractional_counts_without_truncation() -> None:
    with pytest.raises(ValueError, match="integer"):
        hal_azure._extract_counts({"counts": {"0": 1.5}}, n_qubits=1)
    with pytest.raises(ValueError, match="integer"):
        hal_braket._extract_braket_counts(
            type("R", (), {"measurement_counts": {"0": 1.5}})(), n_qubits=1
        )
    with pytest.raises(ValueError, match="integer"):
        hal_qbraid._normalise_counts({"0": 1.5}, n_qubits=1)
    with pytest.raises(ValueError, match="integer"):
        hal_strangeworks._normalise_counts({"0": 1.5}, n_qubits=1)
    with pytest.raises(ValueError, match="integer"):
        hal_pasqal._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_iqm._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_quera_bloqade._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_quantinuum._normalise_counts({"0": 1.5})


def test_count_normalisers_accept_integral_numeric_strings() -> None:
    assert hal_azure._extract_counts({"counts": {"0": "2"}}, n_qubits=1) == {"0": 2}
    assert hal_qbraid._normalise_counts({"0": "2"}, n_qubits=1) == {"0": 2}
    assert hal_strangeworks._normalise_counts({"0": "2"}, n_qubits=1) == {"0": 2}
    assert hal_pasqal._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_iqm._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_quera_bloqade._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_quantinuum._normalise_counts({"0": "2"}) == {"0": 2}


def test_count_normalisers_reject_non_binary_bitstring_keys() -> None:
    with pytest.raises(ValueError):
        hal_azure._extract_counts({"counts": {" 0 1 ": 1}}, n_qubits=3)
    with pytest.raises(ValueError):
        hal_braket._extract_braket_counts(
            type("R", (), {"measurement_counts": {"0a1": 1}})(), n_qubits=3
        )
    with pytest.raises(ValueError):
        hal_qbraid._normalise_counts({" 01 ": 1}, n_qubits=2)
    with pytest.raises(ValueError):
        hal_strangeworks._normalise_counts({"0x1": 1}, n_qubits=3)
    with pytest.raises(ValueError):
        hal_pasqal._normalise_counts({"0-1": 1})
    with pytest.raises(ValueError):
        hal_iqm._normalise_counts({"0 1": 1})


def test_legacy_integer_coercion_paths_reject_fractional_values() -> None:
    with pytest.raises(ValueError, match="integer"):
        hal_cirq._coerce_int(1.5, field_name="count")
    with pytest.raises(ValueError, match="integer"):
        hal_dwave._coerce_int(1.5, field_name="num_occurrences")
    with pytest.raises(ValueError, match="integer"):
        hal_oqc._coerce_int(1.5, field_name="count")
    with pytest.raises(ValueError, match="integer"):
        hal_pasqal._coerce_int(1.5, field_name="Pulser register site")
    with pytest.raises(ValueError, match="integer"):
        hal_quandela._coerce_int(1.5, field_name="mode")
    with pytest.raises(ValueError, match="integer"):
        hal_quera_bloqade._coerce_int(1.5, field_name="Bloqade atom index")


def test_shot_conservation_guard_rejects_mismatched_totals() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        strict_shot_conservation({"00": 3, "11": 2}, expected_shots=4)

    assert strict_shot_conservation({"00": 3, "11": 2}, expected_shots=5) == 5
