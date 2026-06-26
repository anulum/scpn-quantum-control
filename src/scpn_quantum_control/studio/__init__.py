# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio federation surface
"""The QUANTUM studio's federation surface on the SCPN-STUDIO platform contract.

Exposes the schema-A capability manifest (verbs, evidence schemas, content digest)
that the Hub ingests for federation. See :mod:`scpn_quantum_control.studio.manifest`
and :mod:`scpn_quantum_control.studio.verbs`.
"""

from __future__ import annotations

from .coverage_frontier import (
    ANSWERED_STATUSES,
    CoverageFrontierReport,
    map_claim_status,
    measure_coverage_frontier,
    render_coverage_frontier_markdown,
)
from .evidence_bundle import (
    StudioBundleValidation,
    build_claim_ledger_bundle,
    build_claim_ledger_bundles,
    build_hardware_result_pack_bundle,
    build_hardware_result_pack_bundles,
    evidence_axes,
    validate_bundle,
    validate_bundles,
)
from .federation import (
    build_architecture_map_extension,
    build_federation_document,
    write_federation_document,
)
from .manifest import build_manifest, declared_surface
from .result_pack_seal import (
    build_provider_attestation,
    build_result_pack_unit,
    seal_result_pack,
)
from .verbs import QUANTUM_VERBS, STUDIO_ID, evidence_schemas, verb_substrates

__all__ = [
    "ANSWERED_STATUSES",
    "CoverageFrontierReport",
    "QUANTUM_VERBS",
    "STUDIO_ID",
    "StudioBundleValidation",
    "build_architecture_map_extension",
    "build_claim_ledger_bundle",
    "build_claim_ledger_bundles",
    "build_federation_document",
    "build_hardware_result_pack_bundle",
    "build_hardware_result_pack_bundles",
    "build_manifest",
    "build_provider_attestation",
    "build_result_pack_unit",
    "declared_surface",
    "evidence_schemas",
    "evidence_axes",
    "map_claim_status",
    "measure_coverage_frontier",
    "render_coverage_frontier_markdown",
    "seal_result_pack",
    "validate_bundle",
    "validate_bundles",
    "verb_substrates",
    "write_federation_document",
]
