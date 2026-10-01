# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — L16 Cybernetic Closure
"""L16 quantum indicators and bounded heuristic director evidence."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .director_contracts import (
        L16_DIRECTOR_CLAIM_BOUNDARY,
        L16_DIRECTOR_SCHEMA,
        L16DirectorEvidence,
        L16IndicatorCertificate,
        L16RouteEvidence,
        L16ScenarioSpec,
    )
    from .director_evidence import validate_l16_evidence, write_l16_evidence
    from .director_product import (
        L16DirectorPolicyError,
        frozen_l16_scenarios,
        informative_l16_indicators,
        l16_promotion_blockers,
        observer_inputs_from_l16,
        run_l16_director_suite,
        run_l16_indicator_scenario,
    )
    from .quantum_director import L16Result, compute_l16_lyapunov

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "L16_DIRECTOR_CLAIM_BOUNDARY": (
        "scpn_quantum_control.l16.director_contracts",
        "L16_DIRECTOR_CLAIM_BOUNDARY",
    ),
    "L16_DIRECTOR_SCHEMA": ("scpn_quantum_control.l16.director_contracts", "L16_DIRECTOR_SCHEMA"),
    "L16DirectorEvidence": ("scpn_quantum_control.l16.director_contracts", "L16DirectorEvidence"),
    "L16IndicatorCertificate": (
        "scpn_quantum_control.l16.director_contracts",
        "L16IndicatorCertificate",
    ),
    "L16RouteEvidence": ("scpn_quantum_control.l16.director_contracts", "L16RouteEvidence"),
    "L16ScenarioSpec": ("scpn_quantum_control.l16.director_contracts", "L16ScenarioSpec"),
    "validate_l16_evidence": (
        "scpn_quantum_control.l16.director_evidence",
        "validate_l16_evidence",
    ),
    "write_l16_evidence": ("scpn_quantum_control.l16.director_evidence", "write_l16_evidence"),
    "L16DirectorPolicyError": (
        "scpn_quantum_control.l16.director_product",
        "L16DirectorPolicyError",
    ),
    "frozen_l16_scenarios": ("scpn_quantum_control.l16.director_product", "frozen_l16_scenarios"),
    "informative_l16_indicators": (
        "scpn_quantum_control.l16.director_product",
        "informative_l16_indicators",
    ),
    "l16_promotion_blockers": (
        "scpn_quantum_control.l16.director_product",
        "l16_promotion_blockers",
    ),
    "observer_inputs_from_l16": (
        "scpn_quantum_control.l16.director_product",
        "observer_inputs_from_l16",
    ),
    "run_l16_director_suite": (
        "scpn_quantum_control.l16.director_product",
        "run_l16_director_suite",
    ),
    "run_l16_indicator_scenario": (
        "scpn_quantum_control.l16.director_product",
        "run_l16_indicator_scenario",
    ),
    "L16Result": ("scpn_quantum_control.l16.quantum_director", "L16Result"),
    "compute_l16_lyapunov": ("scpn_quantum_control.l16.quantum_director", "compute_l16_lyapunov"),
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
    "L16_DIRECTOR_CLAIM_BOUNDARY",
    "L16_DIRECTOR_SCHEMA",
    "L16DirectorEvidence",
    "L16DirectorPolicyError",
    "L16IndicatorCertificate",
    "L16Result",
    "L16RouteEvidence",
    "L16ScenarioSpec",
    "compute_l16_lyapunov",
    "frozen_l16_scenarios",
    "informative_l16_indicators",
    "l16_promotion_blockers",
    "observer_inputs_from_l16",
    "run_l16_director_suite",
    "run_l16_indicator_scenario",
    "validate_l16_evidence",
    "write_l16_evidence",
]
