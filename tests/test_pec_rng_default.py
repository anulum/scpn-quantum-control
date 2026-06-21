# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Test for PEC default RNG path
"""Test the probabilistic-error-cancellation default-RNG construction path."""

from __future__ import annotations

from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.pec import PECResult, pec_sample


def test_pec_sample_defaults_rng_when_unspecified() -> None:
    """Omitting the generator falls back to a fresh default RNG and still samples."""
    qc = QuantumCircuit(1)
    qc.h(0)
    result = pec_sample(qc, 0.01, n_samples=4)
    assert isinstance(result, PECResult)
