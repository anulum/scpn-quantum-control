# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multiscale chimera observables
"""Hierarchy-aware order parameters composed from oscillatools diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from oscillatools.accel.kuramoto_chimera import community_order_parameters

from .schema import FloatArray, MultiscaleHierarchy


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _immutable(values: object, *, name: str, ndim: int) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LevelOrderParameterSummary:
    """Order-parameter diagnostics for one hierarchy level.

    Attributes
    ----------
    level_name
        Exact hierarchy level identifier.
    community_order_parameters
        Array of shape ``(time, communities)`` with coherence magnitudes.
    mean_by_community
        Time mean of each community column.
    chimera_index
        Mean across-time population variance across communities, matching
        Shanahan's chimera index. It is zero for a single community.
    community_metastability
        Mean across-community population variance through time.

    """

    level_name: str
    community_order_parameters: FloatArray
    mean_by_community: FloatArray
    chimera_index: float
    community_metastability: float

    def __post_init__(self) -> None:
        """Validate and freeze one hierarchy-level diagnostic summary."""
        if not self.level_name.strip():
            raise ValueError("level_name must be non-empty")
        values = _immutable(
            self.community_order_parameters,
            name="community_order_parameters",
            ndim=2,
        )
        means = _immutable(self.mean_by_community, name="mean_by_community", ndim=1)
        if means.shape != (values.shape[1],):
            raise ValueError("mean_by_community shape must match the community columns")
        for name, value in (
            ("chimera_index", self.chimera_index),
            ("community_metastability", self.community_metastability),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value")
        object.__setattr__(self, "level_name", self.level_name.strip())
        object.__setattr__(self, "community_order_parameters", values)
        object.__setattr__(self, "mean_by_community", means)


@dataclass(frozen=True, slots=True)
class MultiscaleOrderParameterReport:
    """Immutable order-parameter report over every hierarchy level.

    ``global_order_parameter`` has one value per input time sample. ``levels``
    preserves the hierarchy's fine-to-coarse order. ``content_digest`` binds
    the numerical trajectory and hierarchy definition.
    """

    hierarchy: MultiscaleHierarchy
    global_order_parameter: FloatArray
    levels: tuple[LevelOrderParameterSummary, ...]
    content_digest: str

    def __post_init__(self) -> None:
        """Validate and freeze the complete multiscale diagnostic report."""
        global_order = _immutable(
            self.global_order_parameter,
            name="global_order_parameter",
            ndim=1,
        )
        if tuple(level.level_name for level in self.levels) != self.hierarchy.level_names:
            raise ValueError("report levels must match hierarchy order exactly")
        if any(
            level.community_order_parameters.shape[0] != global_order.size for level in self.levels
        ):
            raise ValueError("all level reports must match global time length")
        if not _is_sha256(self.content_digest):
            raise ValueError("content_digest must be a SHA-256 hexadecimal digest")
        object.__setattr__(self, "global_order_parameter", global_order)

    def level(self, name: str) -> LevelOrderParameterSummary:
        """Return the report row named ``name`` or raise ``KeyError``."""
        for level in self.levels:
            if level.level_name == name:
                return level
        raise KeyError(f"unknown report level: {name!r}")


def measure_multiscale_order_parameters(
    phases: object,
    hierarchy: MultiscaleHierarchy,
) -> MultiscaleOrderParameterReport:
    """Measure global and community coherence at every hierarchy level.

    Parameters
    ----------
    phases
        Finite phase trajectory of shape ``(time, hierarchy.node_count)`` in
        radians.
    hierarchy
        Complete fine-to-coarse nested partition definition.

    Returns
    -------
    MultiscaleOrderParameterReport
        Read-only global coherence plus per-level Shanahan chimera and
        community-metastability summaries.

    """
    trajectory = _immutable(phases, name="phases", ndim=2)
    if trajectory.shape[1] != hierarchy.node_count:
        raise ValueError("phases node width must match hierarchy.node_count")
    summaries: list[LevelOrderParameterSummary] = []
    for level in hierarchy.levels:
        values = community_order_parameters(trajectory, level.communities)
        summaries.append(
            LevelOrderParameterSummary(
                level_name=level.name,
                community_order_parameters=values,
                mean_by_community=np.mean(values, axis=0),
                chimera_index=float(np.var(values, axis=1).mean()),
                community_metastability=float(np.var(values, axis=0).mean()),
            )
        )
    global_order = np.abs(np.mean(np.exp(1j * trajectory), axis=1)).astype(np.float64)
    digest = hashlib.sha256()
    digest.update(repr(hierarchy).encode("utf-8"))
    digest.update(trajectory.tobytes(order="C"))
    return MultiscaleOrderParameterReport(
        hierarchy=hierarchy,
        global_order_parameter=global_order,
        levels=tuple(summaries),
        content_digest=digest.hexdigest(),
    )


__all__ = [
    "LevelOrderParameterSummary",
    "MultiscaleOrderParameterReport",
    "measure_multiscale_order_parameters",
]
