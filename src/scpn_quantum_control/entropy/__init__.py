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

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "FIPS_SAMPLE_BITS": ("scpn_quantum_control.entropy.fips_140_2", "FIPS_SAMPLE_BITS"),
    "FipsHealthReport": ("scpn_quantum_control.entropy.fips_140_2", "FipsHealthReport"),
    "enforce_fips_140_2": ("scpn_quantum_control.entropy.fips_140_2", "enforce_fips_140_2"),
    "fips_140_2_tests": ("scpn_quantum_control.entropy.fips_140_2", "fips_140_2_tests"),
    "NistTestResult": ("scpn_quantum_control.entropy.nist_sp800_22", "NistTestResult"),
    "approximate_entropy_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "approximate_entropy_test",
    ),
    "berlekamp_massey": ("scpn_quantum_control.entropy.nist_sp800_22", "berlekamp_massey"),
    "binary_matrix_rank_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "binary_matrix_rank_test",
    ),
    "block_frequency_test": ("scpn_quantum_control.entropy.nist_sp800_22", "block_frequency_test"),
    "cumulative_sums_test": ("scpn_quantum_control.entropy.nist_sp800_22", "cumulative_sums_test"),
    "dft_spectral_test": ("scpn_quantum_control.entropy.nist_sp800_22", "dft_spectral_test"),
    "frequency_test": ("scpn_quantum_control.entropy.nist_sp800_22", "frequency_test"),
    "linear_complexity_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "linear_complexity_test",
    ),
    "longest_run_of_ones_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "longest_run_of_ones_test",
    ),
    "maurers_universal_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "maurers_universal_test",
    ),
    "non_overlapping_template_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "non_overlapping_template_test",
    ),
    "overlapping_template_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "overlapping_template_test",
    ),
    "random_excursions_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "random_excursions_test",
    ),
    "random_excursions_variant_test": (
        "scpn_quantum_control.entropy.nist_sp800_22",
        "random_excursions_variant_test",
    ),
    "runs_test": ("scpn_quantum_control.entropy.nist_sp800_22", "runs_test"),
    "serial_test": ("scpn_quantum_control.entropy.nist_sp800_22", "serial_test"),
    "EntropyHealthReport": ("scpn_quantum_control.entropy.qrng_stream", "EntropyHealthReport"),
    "QRNGStream": ("scpn_quantum_control.entropy.qrng_stream", "QRNGStream"),
    "AerQuantumEntropySource": (
        "scpn_quantum_control.entropy.quantum_source",
        "AerQuantumEntropySource",
    ),
    "QuantumSourceKind": ("scpn_quantum_control.entropy.quantum_source", "QuantumSourceKind"),
    "von_neumann_debias": ("scpn_quantum_control.entropy.quantum_source", "von_neumann_debias"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


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
