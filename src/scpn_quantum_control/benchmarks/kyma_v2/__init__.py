# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 corrected-design composition probe
"""KYMA v2 compositional-generalisation probe (corrected design).

v2 answers the two defects diagnosed by the v1 NEGATIVE
(``KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18.md``, commit ``2f67de12``,
kept as the honest baseline):

1. **Coupling gating** — the control code gates the coupling ``K``, not just the
   frequency drive, so one substrate physically realises in-phase *and*
   anti-phase motifs at once (:mod:`.coupling`, :mod:`.teacher`).
2. **Non-separable, data-dependent readout** — a fixed ambient coupling makes the
   joint state non-factorisable and the label is a quantised *achieved* phase, so
   the answer depends on ``θ0`` and on the interaction of both relations; a
   param-matched MLP cannot compose it by learning each relation alone.

The design constants are fixed by a mechanism-only sanity check (:mod:`.design`,
teacher dynamics only) *before* any model is trained, and the pass/fail contract
is frozen in ``KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21.md``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .task import ProbeConfigV2, TrialBatchV2, build_trials

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ProbeConfigV2": ("scpn_quantum_control.benchmarks.kyma_v2.task", "ProbeConfigV2"),
    "TrialBatchV2": ("scpn_quantum_control.benchmarks.kyma_v2.task", "TrialBatchV2"),
    "build_trials": ("scpn_quantum_control.benchmarks.kyma_v2.task", "build_trials"),
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


__all__ = ["ProbeConfigV2", "TrialBatchV2", "build_trials"]
