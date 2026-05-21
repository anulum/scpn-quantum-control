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
    hal_iqm,
    hal_pasqal,
    hal_qbraid,
    hal_quantinuum,
    hal_quera_bloqade,
    hal_strangeworks,
)


def test_count_normalisers_reject_fractional_counts_without_truncation() -> None:
    with pytest.raises(ValueError, match="integer"):
        hal_azure._extract_counts({"counts": {"0": 1.5}})
    with pytest.raises(ValueError, match="integer"):
        hal_braket._extract_braket_counts(type("R", (), {"measurement_counts": {"0": 1.5}})())
    with pytest.raises(ValueError, match="integer"):
        hal_qbraid._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_strangeworks._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_pasqal._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_iqm._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_quera_bloqade._normalise_counts({"0": 1.5})
    with pytest.raises(ValueError, match="integer"):
        hal_quantinuum._normalise_counts({"0": 1.5})


def test_count_normalisers_accept_integral_numeric_strings() -> None:
    assert hal_azure._extract_counts({"counts": {"0": "2"}}) == {"0": 2}
    assert hal_qbraid._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_strangeworks._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_pasqal._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_iqm._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_quera_bloqade._normalise_counts({"0": "2"}) == {"0": 2}
    assert hal_quantinuum._normalise_counts({"0": "2"}) == {"0": 2}
