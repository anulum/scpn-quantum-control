# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — resilient optional Rust accelerator accessor
"""Single resilient entry point to the optional :mod:`oscillatools` Rust engine.

Several physics kernels (Kuramoto XY/Floquet, Kn-m Hamiltonian, Koopman,
symmetry decay, biological surface code, pulse shaping) opportunistically use the
native ``scpn_quantum_engine`` extension exposed by
:func:`oscillatools.accel.rust_import.optional_rust_engine`, each with a pure
Python/Qiskit fallback. Importing that accelerator at module scope made every one
of those modules — and therefore the ``scpn_quantum_control`` package root —
hard-require a complete :mod:`oscillatools` install: a partial or minimal
environment could not even import the package or run a stdlib-only integrity
tool.

This module centralises the accelerator lookup behind one call-time import that
soft-fails when :mod:`oscillatools` is absent or incomplete. Consumers import
:func:`optional_rust_engine` from here instead of from :mod:`oscillatools`
directly, so importing them never touches :mod:`oscillatools`; the accelerator is
resolved only when a kernel actually asks for it, and its absence degrades
cleanly to the Python path.

``oscillatools`` remains a declared runtime dependency; this only makes the
*import path* resilient, it does not make the accelerator optional at the
functional level beyond the fallbacks the kernels already carry.
"""

from __future__ import annotations

from types import ModuleType

__all__ = ["optional_rust_engine"]


def optional_rust_engine() -> ModuleType | None:
    """Return the optional ``scpn_quantum_engine`` Rust extension, or ``None``.

    The :mod:`oscillatools` accelerator import is deferred to call time and
    soft-fails when :mod:`oscillatools` cannot be imported (a partial or minimal
    install), so importing any consumer module never hard-requires the
    accelerator stack.

    Returns
    -------
    ModuleType | None
        The Rust engine module when both :mod:`oscillatools` and the native
        ``scpn_quantum_engine`` extension are importable; ``None`` when the
        accelerator is unavailable, in which case callers use the Python path.
    """
    try:
        from oscillatools.accel.rust_import import (
            optional_rust_engine as _oscillatools_rust_engine,
        )
    except ModuleNotFoundError:
        return None
    return _oscillatools_rust_engine()
