# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — optional local job service
"""Durable local jobs around the local-only executive handlers.

See :mod:`scpn_quantum_control.studio_workspace.jobs.service`. Exports resolve
lazily, so starting the worker module does not import the service first.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .journal import TERMINAL_STATES, Attempt, Event, JobJournal
    from .registry import build_local_registry, local_verbs
    from .service import (
        JobRefused,
        LocalJobPolicy,
        LocalJobServer,
        LocalJobService,
        admit_request,
        attempt_view,
    )

_EXPORTS: dict[str, str] = {
    "TERMINAL_STATES": "journal",
    "Attempt": "journal",
    "Event": "journal",
    "JobJournal": "journal",
    "build_local_registry": "registry",
    "local_verbs": "registry",
    "JobRefused": "service",
    "LocalJobPolicy": "service",
    "LocalJobServer": "service",
    "LocalJobService": "service",
    "admit_request": "service",
    "attempt_view": "service",
}


def __getattr__(name: str) -> Any:
    """Resolve a public export from its owning module.

    Parameters
    ----------
    name
        Requested export.

    Returns
    -------
    Any
        The exported object.

    Raises
    ------
    AttributeError
        ``name`` is not an export of this package.

    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)


__all__ = [
    "TERMINAL_STATES",
    "Attempt",
    "Event",
    "JobJournal",
    "JobRefused",
    "LocalJobPolicy",
    "LocalJobServer",
    "LocalJobService",
    "admit_request",
    "attempt_view",
    "build_local_registry",
    "local_verbs",
]
