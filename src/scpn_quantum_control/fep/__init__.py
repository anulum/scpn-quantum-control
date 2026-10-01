# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Free Energy Principle
"""Free Energy Principle (FEP) on the SCPN quantum substrate.

Implements Friston's variational free energy framework mapped onto
the SCPN hierarchy: oscillator phases serve as sufficient statistics,
K_nm couplings as precision parameters, and the UPDE as belief dynamics.

Modules:
    variational_free_energy — F, KL divergence, ELBO
    predictive_coding — hierarchical message-passing across SCPN layers
    quantum_belief — quantum state as belief, measurement as update
"""

from __future__ import annotations

import sys as _sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .predictive_coding import (
        PredictiveCodingResult,
        hierarchical_prediction_error,
        predictive_coding_step,
    )
    from .variational_free_energy import (
        COVARIANCE_SYMMETRY_ATOL,
        PRECISION_RIDGE,
        FreeEnergyResult,
        evidence_lower_bound,
        free_energy_gradient,
        kl_divergence_gaussian,
        variational_free_energy,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "PredictiveCodingResult": (
        "scpn_quantum_control.fep.predictive_coding",
        "PredictiveCodingResult",
    ),
    "hierarchical_prediction_error": (
        "scpn_quantum_control.fep.predictive_coding",
        "hierarchical_prediction_error",
    ),
    "predictive_coding_step": (
        "scpn_quantum_control.fep.predictive_coding",
        "predictive_coding_step",
    ),
    "COVARIANCE_SYMMETRY_ATOL": (
        "scpn_quantum_control.fep.variational_free_energy",
        "COVARIANCE_SYMMETRY_ATOL",
    ),
    "PRECISION_RIDGE": ("scpn_quantum_control.fep.variational_free_energy", "PRECISION_RIDGE"),
    "FreeEnergyResult": ("scpn_quantum_control.fep.variational_free_energy", "FreeEnergyResult"),
    "evidence_lower_bound": (
        "scpn_quantum_control.fep.variational_free_energy",
        "evidence_lower_bound",
    ),
    "free_energy_gradient": (
        "scpn_quantum_control.fep.variational_free_energy",
        "free_energy_gradient",
    ),
    "kl_divergence_gaussian": (
        "scpn_quantum_control.fep.variational_free_energy",
        "kl_divergence_gaussian",
    ),
    "variational_free_energy": (
        "scpn_quantum_control.fep.variational_free_energy",
        "variational_free_energy",
    ),
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
    "FreeEnergyResult",
    "variational_free_energy",
    "COVARIANCE_SYMMETRY_ATOL",
    "PRECISION_RIDGE",
    "kl_divergence_gaussian",
    "evidence_lower_bound",
    "free_energy_gradient",
    "PredictiveCodingResult",
    "hierarchical_prediction_error",
    "predictive_coding_step",
]
