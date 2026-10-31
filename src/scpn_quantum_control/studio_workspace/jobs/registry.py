# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — local-only handler registry of the job service
"""Register only the executive handlers a local job may run.

The command-line registry also carries ``execute``, which submits to provider
hardware once approved. The job service must never expose that route, even to
a caller that sends an approval, so it builds its own registry from the
shipped handlers and keeps a verb only when its federation contract is local:
no approval requirement and no live-hardware side effect.
"""

from __future__ import annotations

from ...studio.executive import ActionRegistry, resolve_verb_contract
from ...studio.executive_cli import build_default_registry

_LIVE_HARDWARE = "LIVE_HARDWARE"


def local_verbs() -> tuple[str, ...]:
    """Return the shipped verbs whose contracts allow a local run.

    Returns
    -------
    tuple of str
        Sorted verbs with no approval requirement and no live-hardware effect.

    """
    return tuple(
        verb
        for verb in build_default_registry().verbs()
        if not resolve_verb_contract(verb).requires_approval
        and resolve_verb_contract(verb).side_effect != _LIVE_HARDWARE
    )


def build_local_registry() -> ActionRegistry:
    """Return a registry holding only the local verbs' handlers.

    Returns
    -------
    ActionRegistry
        A fresh registry without any provider or approval-gated handler.

    """
    shipped = build_default_registry()
    registry = ActionRegistry()
    for verb in local_verbs():
        registry.register(shipped.resolve(verb))
    return registry


__all__ = ["build_local_registry", "local_verbs"]
