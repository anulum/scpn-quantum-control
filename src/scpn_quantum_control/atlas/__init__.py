# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Atlas public facade
"""Atlas: an inspectable map of what this repository holds and how it connects.

The facade exposes the snapshot contract: the node and edge vocabulary,
identity classes, edge legality, access projection and the canonical digest
that the browser computes too. Extraction, queries and lenses build on it.
"""

from __future__ import annotations

from .contracts import (
    EDGE_ALIASES_V1,
    EDGE_KINDS,
    NODE_KINDS,
    RELATION_VERSION,
    SNAPSHOT_SCHEMA,
    AtlasContractError,
    AtlasEdge,
    AtlasExclusion,
    AtlasNode,
    AtlasSnapshot,
    AtlasTombstone,
    atlas_snapshot_digest,
    parse_atlas_snapshot,
    project_public,
    snapshot_to_wire,
)

__all__ = [
    "EDGE_ALIASES_V1",
    "EDGE_KINDS",
    "NODE_KINDS",
    "RELATION_VERSION",
    "SNAPSHOT_SCHEMA",
    "AtlasContractError",
    "AtlasEdge",
    "AtlasExclusion",
    "AtlasNode",
    "AtlasSnapshot",
    "AtlasTombstone",
    "atlas_snapshot_digest",
    "parse_atlas_snapshot",
    "project_public",
    "snapshot_to_wire",
]
