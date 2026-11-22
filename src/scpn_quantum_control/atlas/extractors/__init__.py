# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Atlas extractors facade
"""Extract Atlas snapshots from committed producer artefacts without importing what they describe."""

from __future__ import annotations

from .snapshot import (
    DEFAULT_INPUTS,
    EXTRACTOR_VERSION,
    AtlasExtraction,
    AtlasExtractionError,
    AtlasFragment,
    AtlasInput,
    directory_reader,
    extract_atlas_snapshot,
)

__all__ = [
    "DEFAULT_INPUTS",
    "EXTRACTOR_VERSION",
    "AtlasExtraction",
    "AtlasExtractionError",
    "AtlasFragment",
    "AtlasInput",
    "directory_reader",
    "extract_atlas_snapshot",
]
