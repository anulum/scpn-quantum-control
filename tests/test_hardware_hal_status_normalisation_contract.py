# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- status normalisation contract tests
"""Contract tests for adapter status canonicalisation fallback policy."""

from __future__ import annotations

from scpn_quantum_control.hardware import (
    hal_azure,
    hal_braket,
    hal_ionq,
    hal_iqm,
    hal_oqc,
    hal_pasqal,
    hal_qbraid,
    hal_qiskit,
    hal_quantinuum,
    hal_quera_bloqade,
    hal_strangeworks,
)


def test_status_normalisers_reject_raw_unknown_status_leakage() -> None:
    unknown_token = "vendor_future_terminal_state"

    assert hal_pasqal._normalise_status(unknown_token) == "unknown"
    assert hal_oqc._normalise_status(unknown_token) == "unknown"
    assert hal_iqm._normalise_status(unknown_token) == "unknown"
    assert hal_quera_bloqade._normalise_status(unknown_token) == "unknown"
    assert hal_quantinuum._normalise_status(unknown_token) == "unknown"

    assert hal_azure._normalise_status(unknown_token) == "unknown"
    assert hal_braket._normalise_status(unknown_token) == "unknown"
    assert hal_qbraid._normalise_status(unknown_token) == "unknown"
    assert hal_qiskit._normalise_status(unknown_token) == "unknown"
    assert hal_ionq._normalise_status(unknown_token) == "unknown"
    assert hal_strangeworks._normalise_status(unknown_token) == "unknown"


def test_status_normalisers_honour_explicit_default_overrides() -> None:
    unknown_token = "vendor_status_v2"

    assert hal_azure._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_braket._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_qbraid._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_qiskit._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_ionq._normalise_status(unknown_token, default="queued") == "queued"
    assert hal_strangeworks._normalise_status(unknown_token, default="queued") == "queued"
