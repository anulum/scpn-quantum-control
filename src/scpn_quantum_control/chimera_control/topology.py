# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera topology bridge
"""Hierarchy summaries around the existing topology-constraint ledger."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from scpn_quantum_control.topology_control.constraints import (
    ConstraintViolation,
    TopologyConstraintLedger,
)

from .schema import CHIMERA_CONTROL_CLAIM_BOUNDARY, FloatArray, MultiscaleHierarchy


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _matrix(values: object, *, node_count: int) -> FloatArray:
    matrix = np.array(values, dtype=np.float64, copy=True)
    if matrix.shape != (node_count, node_count):
        raise ValueError(f"candidate must have shape ({node_count}, {node_count})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("candidate must contain only finite values")
    return matrix


@dataclass(frozen=True, slots=True)
class HierarchyCouplingSummary:
    """Mean off-diagonal coupling within and between communities at one level.

    ``mean_within`` is ``None`` when every community is a singleton.
    ``mean_between`` is ``None`` for a one-community level because that scale
    has no between-community edge set. Means are descriptive and do not imply
    learned, causal, hardware-realised, or dynamically stable couplings.
    """

    level_name: str
    mean_within: float | None
    mean_between: float | None

    def __post_init__(self) -> None:
        """Validate the level identifier and optional finite means."""
        if not self.level_name.strip():
            raise ValueError("level_name must be non-empty")
        if self.mean_within is not None and not np.isfinite(self.mean_within):
            raise ValueError("mean_within must be finite when present")
        if self.mean_between is not None and not np.isfinite(self.mean_between):
            raise ValueError("mean_between must be finite when present")


@dataclass(frozen=True, slots=True)
class TopologyProjectionReport:
    """Immutable before/after report from ``TopologyConstraintLedger.project``.

    The ledger may report a remaining algebraic-connectivity violation because
    its projection routine does not manufacture connectivity. This record is a
    local candidate audit, not a stability, controllability, PH, DLA, hardware,
    or deployment certificate.
    """

    candidate: FloatArray
    projected: FloatArray
    violations_before: ConstraintViolation
    violations_after: ConstraintViolation
    summaries_before: tuple[HierarchyCouplingSummary, ...]
    summaries_after: tuple[HierarchyCouplingSummary, ...]
    content_digest: str
    claim_boundary: str = CHIMERA_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate and freeze the topology projection report."""
        candidate = np.array(self.candidate, dtype=np.float64, copy=True)
        projected = np.array(self.projected, dtype=np.float64, copy=True)
        if (
            candidate.ndim != 2
            or candidate.shape != projected.shape
            or candidate.size == 0
            or candidate.shape[0] != candidate.shape[1]
        ):
            raise ValueError("candidate and projected must be equal non-empty matrices")
        if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(projected)):
            raise ValueError("topology matrices must contain only finite values")
        if tuple(row.level_name for row in self.summaries_before) != tuple(
            row.level_name for row in self.summaries_after
        ):
            raise ValueError("before and after summaries must use identical levels")
        if not self.summaries_before:
            raise ValueError("topology summaries must be non-empty")
        if not _is_sha256(self.content_digest):
            raise ValueError("content_digest must be a SHA-256 hexadecimal digest")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        candidate.setflags(write=False)
        projected.setflags(write=False)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "projected", projected)


def _summarise(
    matrix: FloatArray,
    hierarchy: MultiscaleHierarchy,
) -> tuple[HierarchyCouplingSummary, ...]:
    rows: list[HierarchyCouplingSummary] = []
    for level in hierarchy.levels:
        membership: dict[int, int] = {}
        for group_index, community in enumerate(level.communities):
            membership.update({node: group_index for node in community})
        within: list[float] = []
        between: list[float] = []
        for left in range(hierarchy.node_count):
            for right in range(left + 1, hierarchy.node_count):
                collection = within if membership[left] == membership[right] else between
                collection.append(float(matrix[left, right]))
        rows.append(
            HierarchyCouplingSummary(
                level_name=level.name,
                mean_within=float(np.mean(within)) if within else None,
                mean_between=float(np.mean(between)) if between else None,
            )
        )
    return tuple(rows)


def project_chimera_coupling(
    candidate: object,
    hierarchy: MultiscaleHierarchy,
    ledger: TopologyConstraintLedger,
) -> TopologyProjectionReport:
    """Project and audit one hierarchy-sized coupling candidate.

    Parameters
    ----------
    candidate
        Finite square matrix with order ``hierarchy.node_count``.
    hierarchy
        Partition hierarchy used only for descriptive within/between summaries.
    ledger
        Existing SCPN topology policy that owns projection and violation
        semantics.

    Returns
    -------
    TopologyProjectionReport
        Read-only original/projected matrices, ledger violation magnitudes,
        multiscale coupling summaries, and a SHA-256 digest.

    """
    original = _matrix(candidate, node_count=hierarchy.node_count)
    projected = ledger.project(original)
    before = ledger.violations(original)
    after = ledger.violations(projected)
    digest = hashlib.sha256()
    digest.update(repr(hierarchy).encode("utf-8"))
    digest.update(repr(ledger).encode("utf-8"))
    digest.update(original.tobytes(order="C"))
    digest.update(projected.tobytes(order="C"))
    return TopologyProjectionReport(
        candidate=original,
        projected=projected,
        violations_before=before,
        violations_after=after,
        summaries_before=_summarise(original, hierarchy),
        summaries_after=_summarise(projected, hierarchy),
        content_digest=digest.hexdigest(),
    )


__all__ = [
    "HierarchyCouplingSummary",
    "TopologyProjectionReport",
    "project_chimera_coupling",
]
