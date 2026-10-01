# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — deployment package exports
# scpn-quantum-control -- deployment exports
"""Cloud-native deployment manifest generation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cloud_native import (
        CloudDeploymentSpec,
        CloudManifestBundle,
        ContainerResources,
        generate_cloud_manifests,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "CloudDeploymentSpec": ("scpn_quantum_control.deployment.cloud_native", "CloudDeploymentSpec"),
    "CloudManifestBundle": ("scpn_quantum_control.deployment.cloud_native", "CloudManifestBundle"),
    "ContainerResources": ("scpn_quantum_control.deployment.cloud_native", "ContainerResources"),
    "generate_cloud_manifests": (
        "scpn_quantum_control.deployment.cloud_native",
        "generate_cloud_manifests",
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


__all__ = [
    "CloudDeploymentSpec",
    "CloudManifestBundle",
    "ContainerResources",
    "generate_cloud_manifests",
]
