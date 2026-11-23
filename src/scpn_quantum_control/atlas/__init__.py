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
that the browser computes too, and the bounded, access-filtered queries over a
validated snapshot, the diff of two snapshots with the potential impact of what
changed, the capability gap report built from explicit mappings, and claim
comparisons judged only under compatible assumptions. Extraction lives in :mod:`scpn_quantum_control.atlas.extractors`.
"""

from __future__ import annotations

from .comparability import ComparableValue, Comparison, compare_values, filter_comparisons
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
from .coverage import CapabilityGapReport, capability_gap_report
from .diff import (
    AtlasDiffError,
    AtlasSnapshotDiff,
    diff_atlas_snapshots,
    resolve_saved_view,
)
from .query import (
    PREBUILT_QUERIES,
    AtlasQuery,
    AtlasQueryError,
    AtlasQueryResult,
    run_atlas_query,
    run_prebuilt_query,
)

__all__ = [
    "EDGE_ALIASES_V1",
    "EDGE_KINDS",
    "NODE_KINDS",
    "PREBUILT_QUERIES",
    "RELATION_VERSION",
    "SNAPSHOT_SCHEMA",
    "AtlasContractError",
    "AtlasDiffError",
    "AtlasEdge",
    "AtlasExclusion",
    "AtlasNode",
    "AtlasQuery",
    "AtlasQueryError",
    "AtlasQueryResult",
    "AtlasSnapshot",
    "AtlasSnapshotDiff",
    "AtlasTombstone",
    "CapabilityGapReport",
    "ComparableValue",
    "Comparison",
    "atlas_snapshot_digest",
    "capability_gap_report",
    "compare_values",
    "diff_atlas_snapshots",
    "filter_comparisons",
    "parse_atlas_snapshot",
    "project_public",
    "resolve_saved_view",
    "run_atlas_query",
    "run_prebuilt_query",
    "snapshot_to_wire",
]
