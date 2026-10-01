# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware code generation package
"""Code generation for FPGA pulse deployment (QUA-C.4)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ultrascale_hls import (
        HLS_ARTIFACT_CLAIM_BOUNDARY,
        HLS_ARTIFACT_SCHEMA_VERSION,
        HLS_CONSUMER_CONTRACT_VERSION,
        HLSArtifactFile,
        HLSArtifactManifest,
        HLSArtifactVerification,
        HLSBundle,
        emit_versioned_hls_artifact,
        pulse_to_vivado_hls,
        quantise_q_format,
        verify_hls_artifact_manifest,
        write_bundle,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "HLS_ARTIFACT_CLAIM_BOUNDARY": (
        "scpn_quantum_control.codegen.ultrascale_hls",
        "HLS_ARTIFACT_CLAIM_BOUNDARY",
    ),
    "HLS_ARTIFACT_SCHEMA_VERSION": (
        "scpn_quantum_control.codegen.ultrascale_hls",
        "HLS_ARTIFACT_SCHEMA_VERSION",
    ),
    "HLS_CONSUMER_CONTRACT_VERSION": (
        "scpn_quantum_control.codegen.ultrascale_hls",
        "HLS_CONSUMER_CONTRACT_VERSION",
    ),
    "HLSArtifactFile": ("scpn_quantum_control.codegen.ultrascale_hls", "HLSArtifactFile"),
    "HLSArtifactManifest": ("scpn_quantum_control.codegen.ultrascale_hls", "HLSArtifactManifest"),
    "HLSArtifactVerification": (
        "scpn_quantum_control.codegen.ultrascale_hls",
        "HLSArtifactVerification",
    ),
    "HLSBundle": ("scpn_quantum_control.codegen.ultrascale_hls", "HLSBundle"),
    "emit_versioned_hls_artifact": (
        "scpn_quantum_control.codegen.ultrascale_hls",
        "emit_versioned_hls_artifact",
    ),
    "pulse_to_vivado_hls": ("scpn_quantum_control.codegen.ultrascale_hls", "pulse_to_vivado_hls"),
    "quantise_q_format": ("scpn_quantum_control.codegen.ultrascale_hls", "quantise_q_format"),
    "verify_hls_artifact_manifest": (
        "scpn_quantum_control.codegen.ultrascale_hls",
        "verify_hls_artifact_manifest",
    ),
    "write_bundle": ("scpn_quantum_control.codegen.ultrascale_hls", "write_bundle"),
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
    "HLS_ARTIFACT_CLAIM_BOUNDARY",
    "HLS_ARTIFACT_SCHEMA_VERSION",
    "HLS_CONSUMER_CONTRACT_VERSION",
    "HLSArtifactFile",
    "HLSArtifactManifest",
    "HLSArtifactVerification",
    "HLSBundle",
    "emit_versioned_hls_artifact",
    "pulse_to_vivado_hls",
    "quantise_q_format",
    "verify_hls_artifact_manifest",
    "write_bundle",
]
