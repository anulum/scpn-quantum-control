# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum entropy and randomness package
"""Quantum random-number generation with NIST SP 800-22 and FIPS 140-2 health checks.

Public surfaces:

- :class:`~scpn_quantum_control.entropy.qrng_stream.QRNGStream` — streaming QRNG
  with Von Neumann debiasing and periodic health checks.
- :mod:`~scpn_quantum_control.entropy.nist_sp800_22` — the 15 NIST SP 800-22
  Revision 1a statistical tests.
- :mod:`~scpn_quantum_control.entropy.fips_140_2` — the FIPS 140-2 Annex C
  power-up tests.
- :class:`~scpn_quantum_control.entropy.quantum_source.AerQuantumEntropySource` —
  Qiskit Aer quantum measurement entropy.
"""

from __future__ import annotations

from .fips_140_2 import (
    FIPS_SAMPLE_BITS,
    FipsHealthReport,
    enforce_fips_140_2,
    fips_140_2_tests,
)
from .nist_sp800_22 import (
    NistTestResult,
    approximate_entropy_test,
    berlekamp_massey,
    binary_matrix_rank_test,
    block_frequency_test,
    cumulative_sums_test,
    dft_spectral_test,
    frequency_test,
    linear_complexity_test,
    longest_run_of_ones_test,
    maurers_universal_test,
    non_overlapping_template_test,
    overlapping_template_test,
    random_excursions_test,
    random_excursions_variant_test,
    runs_test,
    serial_test,
)
from .qrng_stream import EntropyHealthReport, QRNGStream
from .quantum_source import (
    AerQuantumEntropySource,
    QuantumSourceKind,
    von_neumann_debias,
)

__all__ = [
    "FIPS_SAMPLE_BITS",
    "AerQuantumEntropySource",
    "EntropyHealthReport",
    "FipsHealthReport",
    "NistTestResult",
    "QRNGStream",
    "QuantumSourceKind",
    "approximate_entropy_test",
    "berlekamp_massey",
    "binary_matrix_rank_test",
    "block_frequency_test",
    "cumulative_sums_test",
    "dft_spectral_test",
    "enforce_fips_140_2",
    "fips_140_2_tests",
    "frequency_test",
    "linear_complexity_test",
    "longest_run_of_ones_test",
    "maurers_universal_test",
    "non_overlapping_template_test",
    "overlapping_template_test",
    "random_excursions_test",
    "random_excursions_variant_test",
    "runs_test",
    "serial_test",
    "von_neumann_debias",
]
