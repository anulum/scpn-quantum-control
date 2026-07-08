# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware code generation package
"""Code generation for FPGA pulse deployment (QUA-C.4)."""

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
