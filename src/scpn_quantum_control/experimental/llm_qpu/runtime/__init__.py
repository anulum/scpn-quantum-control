# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental attempt runtime
"""Private attempt state for the opt-in LLM-QPU lane."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attempt_journal import AttemptJournal, JobPlan, JournalStateError
    from .provider_recovery import recover_existing_job

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "AttemptJournal": (
        "scpn_quantum_control.experimental.llm_qpu.runtime.attempt_journal",
        "AttemptJournal",
    ),
    "JobPlan": ("scpn_quantum_control.experimental.llm_qpu.runtime.attempt_journal", "JobPlan"),
    "JournalStateError": (
        "scpn_quantum_control.experimental.llm_qpu.runtime.attempt_journal",
        "JournalStateError",
    ),
    "recover_existing_job": (
        "scpn_quantum_control.experimental.llm_qpu.runtime.provider_recovery",
        "recover_existing_job",
    ),
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


__all__ = ("AttemptJournal", "JobPlan", "JournalStateError", "recover_existing_job")
