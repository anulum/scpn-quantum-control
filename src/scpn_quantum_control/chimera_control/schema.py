# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control contracts
"""Immutable hierarchy and target contracts for synthetic chimera control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

CHIMERA_CONTROL_CLAIM_BOUNDARY = (
    "deterministic finite-N synthetic Kuramoto-Sakaguchi analysis and unapplied "
    "control proposals; no thermodynamic-limit, stability, controllability, "
    "biological, EEG, clinical, hardware, or deployment claim"
)


class SyntheticRegime(str, Enum):
    """Named finite synthetic regimes served by the two-population generator.

    ``CHIMERA_TRANSIENT`` uses the Abrams–Mirollo–Strogatz–Wiley strong-
    intra/weak-inter population configuration. ``SYNCHRONISED_CONTROL`` is a
    deterministic contrast with stronger inter-population coupling. The names
    describe generator configurations, not certified dynamical attractors.
    """

    CHIMERA_TRANSIENT = "chimera_transient"
    SYNCHRONISED_CONTROL = "synchronised_control"


@dataclass(frozen=True, slots=True)
class HierarchyLevel:
    """One named partition of oscillator indices.

    Parameters
    ----------
    name
        Stable non-empty identifier used by targets and evidence records.
    communities
        Disjoint non-empty oscillator-index groups. Coverage and index bounds
        are validated by :class:`MultiscaleHierarchy` because they depend on
        the hierarchy's node count.

    """

    name: str
    communities: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Validate, normalise, and freeze the community partition."""
        if not self.name.strip():
            raise ValueError("hierarchy level name must be non-empty")
        if not self.communities:
            raise ValueError("hierarchy level must contain at least one community")
        normalised: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for position, community in enumerate(self.communities):
            if any(
                isinstance(index, bool) or not isinstance(index, (int, np.integer))
                for index in community
            ):
                raise ValueError(f"community {position} indices must be integers")
            members = tuple(int(index) for index in community)
            if not members:
                raise ValueError(f"community {position} must be non-empty")
            if len(members) != len(set(members)):
                raise ValueError(f"community {position} repeats an oscillator index")
            overlap = seen.intersection(members)
            if overlap:
                raise ValueError(f"community {position} overlaps oscillator(s) {sorted(overlap)}")
            seen.update(members)
            normalised.append(tuple(sorted(members)))
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "communities", tuple(normalised))


@dataclass(frozen=True, slots=True)
class MultiscaleHierarchy:
    """A complete fine-to-coarse nested partition hierarchy.

    Parameters
    ----------
    node_count
        Number of oscillators. Every level must partition exactly
        ``range(node_count)``.
    levels
        Levels ordered from fine to coarse. Every community in a finer level
        must be wholly contained in exactly one community at the next level.

    Raises
    ------
    ValueError
        If names repeat, a level omits or adds a node, an index is negative,
        or adjacent levels are not nested partitions.

    """

    node_count: int
    levels: tuple[HierarchyLevel, ...]

    def __post_init__(self) -> None:
        """Validate complete coverage and nesting across hierarchy levels."""
        if (
            isinstance(self.node_count, bool)
            or not isinstance(self.node_count, int)
            or self.node_count < 2
        ):
            raise ValueError("node_count must be an integer greater than one")
        if not self.levels:
            raise ValueError("hierarchy must contain at least one level")
        names = tuple(level.name for level in self.levels)
        if len(names) != len(set(names)):
            raise ValueError("hierarchy level names must be unique")
        expected = set(range(self.node_count))
        for level in self.levels:
            observed = {index for community in level.communities for index in community}
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise ValueError(
                    f"level {level.name!r} must partition every node exactly; "
                    f"missing={missing}, extra={extra}"
                )
        for fine, coarse in zip(self.levels, self.levels[1:], strict=False):
            coarse_sets = tuple(set(community) for community in coarse.communities)
            for community in fine.communities:
                containers = sum(set(community).issubset(group) for group in coarse_sets)
                if containers != 1:
                    raise ValueError(f"level {fine.name!r} is not nested inside {coarse.name!r}")

    def level(self, name: str) -> HierarchyLevel:
        """Return the level called ``name`` or raise ``KeyError``.

        The lookup is exact and case-sensitive so evidence records cannot
        silently bind a target to a different scale.
        """
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(f"unknown hierarchy level: {name!r}")

    @property
    def level_names(self) -> tuple[str, ...]:
        """Return fine-to-coarse level names."""
        return tuple(level.name for level in self.levels)


@dataclass(frozen=True, slots=True)
class HierarchyTarget:
    """Order-parameter targets and weight for one hierarchy level.

    Parameters
    ----------
    level_name
        Exact :class:`HierarchyLevel` name.
    order_parameters
        One desired Kuramoto coherence magnitude in ``[0, 1]`` per community.
    weight
        Non-negative multiplier used by the composed objective.

    """

    level_name: str
    order_parameters: tuple[float, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalise the target row and its weight."""
        if not self.level_name.strip():
            raise ValueError("target level_name must be non-empty")
        targets = tuple(float(value) for value in self.order_parameters)
        if not targets:
            raise ValueError("target order_parameters must be non-empty")
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in targets):
            raise ValueError("target order_parameters must be finite values in [0, 1]")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("target weight must be a finite non-negative value")
        object.__setattr__(self, "level_name", self.level_name.strip())
        object.__setattr__(self, "order_parameters", targets)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class ChimeraControlSpecification:
    """Validated hierarchy and differentiable order-parameter targets.

    Every target name must resolve to exactly one hierarchy level and must
    supply one value per community. A level may be omitted deliberately, but a
    level cannot have two competing target rows.
    """

    hierarchy: MultiscaleHierarchy
    targets: tuple[HierarchyTarget, ...]
    claim_boundary: str = CHIMERA_CONTROL_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate target uniqueness, widths, and the claim boundary."""
        if not self.targets:
            raise ValueError("control specification must contain at least one target")
        names = tuple(target.level_name for target in self.targets)
        if len(names) != len(set(names)):
            raise ValueError("control specification target levels must be unique")
        for target in self.targets:
            level = self.hierarchy.level(target.level_name)
            if len(target.order_parameters) != len(level.communities):
                raise ValueError(
                    f"target {target.level_name!r} requires "
                    f"{len(level.communities)} order parameter(s)"
                )
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")


def two_population_hierarchy(population_size: int) -> MultiscaleHierarchy:
    """Build the canonical two-population and whole-ensemble hierarchy.

    Parameters
    ----------
    population_size
        Oscillators in each population; must be at least two.

    Returns
    -------
    MultiscaleHierarchy
        A ``population`` level with two equally sized communities followed by
        an ``ensemble`` level containing every oscillator.

    """
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < 2
    ):
        raise ValueError("population_size must be an integer greater than one")
    first = tuple(range(population_size))
    second = tuple(range(population_size, 2 * population_size))
    return MultiscaleHierarchy(
        node_count=2 * population_size,
        levels=(
            HierarchyLevel("population", (first, second)),
            HierarchyLevel("ensemble", (first + second,)),
        ),
    )


__all__ = [
    "CHIMERA_CONTROL_CLAIM_BOUNDARY",
    "ChimeraControlSpecification",
    "FloatArray",
    "HierarchyLevel",
    "HierarchyTarget",
    "MultiscaleHierarchy",
    "SyntheticRegime",
    "two_population_hierarchy",
]
