# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL count integrity contract tests
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
    hal_qiskit,
    hal_quandela,
    hal_quantinuum,
    hal_quera_bloqade,
    hal_strangeworks,
)
from scpn_quantum_control.hardware._count_integrity import strict_shot_conservation


def test_count_normalisers_reject_fractional_counts_without_truncation() -> None:
    """Legacy count decoders refuse fractional multiplicities instead of truncating them."""
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
    """Legacy decoders retain supported integral numeric-string count compatibility."""
    assert hal_azure._extract_counts({"counts": {"0": "2"}}, n_qubits=1) == {"0": 2}
    assert hal_qbraid._normalise_counts({"0": "2"}, n_qubits=1) == {"0": 2}
    assert hal_strangeworks._normalise_counts({"0": "2"}, n_qubits=1) == {"0": 2}
    assert hal_pasqal._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_iqm._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_quera_bloqade._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_quantinuum._normalise_counts({"0": "2"}) == {"0": 2}


@pytest.mark.parametrize("raw", [None, [], "01", 4])
def test_iqm_count_normaliser_rejects_nonmapping_internal_inputs(raw: object) -> None:
    """Supplement public result-channel conformance with the legacy normalizer's type guard."""
    with pytest.raises(TypeError, match="IQM counts must be a mapping"):
        hal_iqm._normalise_counts(raw)


@pytest.mark.parametrize("raw", [None, [], "01", 4])
def test_pasqal_count_normaliser_rejects_nonmapping_internal_inputs(raw: object) -> None:
    """Supplement public analog channel refusals with the original normalizer's type contract."""
    with pytest.raises(TypeError, match="Pasqal counts must be a mapping"):
        hal_pasqal._normalise_counts(raw)


def test_dwave_legacy_unspecified_domain_count_codec_keeps_its_original_guard() -> None:
    """Supplement declared-domain public tests with the internal optional-domain compatibility contract."""
    assert hal_dwave._sample_bitstring({"a": -1, "b": 1}, ["a", "b"]) == "01"
    with pytest.raises(ValueError, match="binary or spin"):
        hal_dwave._sample_bitstring({"a": 2}, ["a"])


@pytest.mark.parametrize("raw", [None, [], "20", 4])
def test_quandela_count_decoders_reject_nonmapping_internal_inputs(raw: object) -> None:
    """Supplement public channel refusals with the two internal count-decoder type contracts."""
    with pytest.raises(TypeError, match="Quandela counts must be a mapping"):
        hal_quandela._normalise_counts(raw)
    with pytest.raises(ValueError, match="occupation count mapping"):
        hal_quandela._photonic_samples(raw)


def test_count_normalisers_reject_non_binary_bitstring_keys() -> None:
    """Binary legacy routes reject non-binary labels and internal whitespace."""
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
    """Count and native-axis integer contracts reject fractional numeric values."""
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
    """Exact shot conservation rejects a wrong total and admits the requested sum."""
    with pytest.raises(ValueError, match="mismatch"):
        strict_shot_conservation({"00": 3, "11": 2}, expected_shots=4)

    assert strict_shot_conservation({"00": 3, "11": 2}, expected_shots=5) == 5


def test_count_normalisers_reject_empty_count_maps() -> None:
    """An empty provider histogram cannot qualify a sampled result."""
    with pytest.raises(ValueError, match="empty count map|did not contain any counts"):
        hal_azure._extract_counts({"counts": {}}, n_qubits=1)
    with pytest.raises(ValueError, match="empty count map"):
        hal_braket._extract_braket_counts(type("R", (), {"measurement_counts": {}})(), n_qubits=1)
    with pytest.raises(ValueError, match="empty count map"):
        hal_qbraid._normalise_counts({}, n_qubits=1)
    with pytest.raises(ValueError, match="empty count map"):
        hal_strangeworks._normalise_counts({}, n_qubits=1)
    with pytest.raises(ValueError, match="did not contain any counts"):
        hal_pasqal._normalise_counts({})
    with pytest.raises(ValueError, match="did not contain any counts"):
        hal_iqm._normalise_counts({})


def test_qiskit_count_normaliser_accumulates_equivalent_bitstring_keys() -> None:
    """Equivalent supported Qiskit labels accumulate rather than overwrite counts."""
    counts = hal_qiskit._normalise_counts({"01": 2, (0, 1): 3})
    assert counts == {"01": 5}


def test_iqm_oqc_pasqal_count_normalisers_accumulate_equivalent_bitstring_keys() -> None:
    """String and tuple labels merge under their existing binary compatibility codecs."""
    iqm_counts = hal_iqm._normalise_counts({"01": 2, (0, 1): 3})
    assert iqm_counts == {"01": 5}

    oqc_counts = hal_oqc._normalise_counts({"01": 2, (0, 1): 3})
    assert oqc_counts == {"01": 5}

    pasqal_counts = hal_pasqal._normalise_counts({"01": 2, (0, 1): 3})
    assert pasqal_counts == {"01": 5}


def test_quandela_count_normaliser_accumulates_canonical_state_collisions() -> None:
    """Equivalent legacy photonic state labels retain their combined multiplicity."""
    counts = hal_quandela._normalise_counts({"1": 2, 1: 3})
    assert counts == {"1": 5}


def test_quandela_count_normaliser_trims_state_key_padding() -> None:
    """Legacy photonic label padding canonicalises without losing observations."""
    counts = hal_quandela._normalise_counts({" 10 ": 2, "10": 3})
    assert counts == {"10": 5}


def test_qbraid_and_strangeworks_normalisers_accumulate_canonical_collisions() -> None:
    """Padded-width legacy keys accumulate into the same declared binary output."""
    qbraid_counts = hal_qbraid._normalise_counts({"1": 2, "01": 3}, n_qubits=2)
    assert qbraid_counts == {"01": 5}

    strangeworks_counts = hal_strangeworks._normalise_counts({"1": 2, "01": 3}, n_qubits=2)
    assert strangeworks_counts == {"01": 5}


def test_azure_and_braket_normalisers_accumulate_canonical_collisions() -> None:
    """Azure and Braket retain the combined counts of equivalent width-normalised keys."""
    azure_counts = hal_azure._extract_counts({"counts": {"1": 2, "01": 3}}, n_qubits=2)
    assert azure_counts == {"01": 5}

    braket_counts = hal_braket._extract_braket_counts(
        type("R", (), {"measurement_counts": {"1": 2, "01": 3}})(),
        n_qubits=2,
    )
    assert braket_counts == {"01": 5}
