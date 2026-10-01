# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Identity Continuity Analysis
"""Identity continuity analysis for coupled oscillator networks.

Quantitative tools for characterizing identity attractor basins,
coherence budgets, entanglement structure, and cryptographic
fingerprinting of coupling topologies.
"""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .binding_spec import (
        ARCANE_SAPIENCE_SPEC,
        ORCHESTRATOR_MAPPING,
        build_identity_attractor,
        orchestrator_to_quantum_phases,
        quantum_to_orchestrator_phases,
        solve_identity,
    )
    from .coherence_budget import coherence_budget, fidelity_at_depth
    from .entanglement_witness import chsh_from_statevector, disposition_entanglement_map
    from .ground_state import IdentityAttractor
    from .identity_key import identity_fingerprint, verify_identity
    from .robustness import (
        RobustnessCertificate,
        compute_robustness_certificate,
        gap_vs_perturbation_scan,
        perturbation_fidelity,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ARCANE_SAPIENCE_SPEC": ("scpn_quantum_control.identity.binding_spec", "ARCANE_SAPIENCE_SPEC"),
    "ORCHESTRATOR_MAPPING": ("scpn_quantum_control.identity.binding_spec", "ORCHESTRATOR_MAPPING"),
    "build_identity_attractor": (
        "scpn_quantum_control.identity.binding_spec",
        "build_identity_attractor",
    ),
    "orchestrator_to_quantum_phases": (
        "scpn_quantum_control.identity.binding_spec",
        "orchestrator_to_quantum_phases",
    ),
    "quantum_to_orchestrator_phases": (
        "scpn_quantum_control.identity.binding_spec",
        "quantum_to_orchestrator_phases",
    ),
    "solve_identity": ("scpn_quantum_control.identity.binding_spec", "solve_identity"),
    "coherence_budget": ("scpn_quantum_control.identity.coherence_budget", "coherence_budget"),
    "fidelity_at_depth": ("scpn_quantum_control.identity.coherence_budget", "fidelity_at_depth"),
    "chsh_from_statevector": (
        "scpn_quantum_control.identity.entanglement_witness",
        "chsh_from_statevector",
    ),
    "disposition_entanglement_map": (
        "scpn_quantum_control.identity.entanglement_witness",
        "disposition_entanglement_map",
    ),
    "IdentityAttractor": ("scpn_quantum_control.identity.ground_state", "IdentityAttractor"),
    "identity_fingerprint": ("scpn_quantum_control.identity.identity_key", "identity_fingerprint"),
    "verify_identity": ("scpn_quantum_control.identity.identity_key", "verify_identity"),
    "RobustnessCertificate": ("scpn_quantum_control.identity.robustness", "RobustnessCertificate"),
    "compute_robustness_certificate": (
        "scpn_quantum_control.identity.robustness",
        "compute_robustness_certificate",
    ),
    "gap_vs_perturbation_scan": (
        "scpn_quantum_control.identity.robustness",
        "gap_vs_perturbation_scan",
    ),
    "perturbation_fidelity": ("scpn_quantum_control.identity.robustness", "perturbation_fidelity"),
}


class _ExportModule(ModuleType):
    """Keep declared object exports when Python publishes child modules."""

    def __setattr__(self, name: str, value: object) -> None:
        """Publish a module attribute while preserving same-named exports.

        Parameters
        ----------
        name
            Attribute assigned by the import system or a caller.
        value
            Value to publish in the package namespace.

        """
        target = _PUBLIC_EXPORTS.get(name)
        if (
            target is not None
            and target[1] is not None
            and isinstance(value, ModuleType)
            and value.__name__ in {target[0], f"{__name__}.{name}"}
        ):
            value = __getattr__(name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _ExportModule


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
    "IdentityAttractor",
    "coherence_budget",
    "fidelity_at_depth",
    "disposition_entanglement_map",
    "chsh_from_statevector",
    "identity_fingerprint",
    "verify_identity",
    "ARCANE_SAPIENCE_SPEC",
    "build_identity_attractor",
    "solve_identity",
    "RobustnessCertificate",
    "compute_robustness_certificate",
    "perturbation_fidelity",
    "gap_vs_perturbation_scan",
    "ORCHESTRATOR_MAPPING",
    "quantum_to_orchestrator_phases",
    "orchestrator_to_quantum_phases",
]
