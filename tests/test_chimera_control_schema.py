# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Chimera-control schema tests
"""Production-surface tests for immutable chimera-control contracts."""

from __future__ import annotations

import pytest

from scpn_quantum_control.chimera_control.schema import (
    CHIMERA_CONTROL_CLAIM_BOUNDARY,
    ChimeraControlSpecification,
    HierarchyLevel,
    HierarchyTarget,
    MultiscaleHierarchy,
    two_population_hierarchy,
)


def test_two_population_hierarchy_is_complete_nested_and_lookup_is_exact() -> None:
    """Build the canonical hierarchy and require case-sensitive level lookup."""
    hierarchy = two_population_hierarchy(3)

    assert hierarchy.node_count == 6
    assert hierarchy.level_names == ("population", "ensemble")
    assert hierarchy.level("population").communities == ((0, 1, 2), (3, 4, 5))
    assert hierarchy.level("ensemble").communities == ((0, 1, 2, 3, 4, 5),)
    with pytest.raises(KeyError, match="unknown hierarchy level"):
        hierarchy.level("Population")


@pytest.mark.parametrize("size", [0, 1, True])
def test_two_population_hierarchy_rejects_small_or_boolean_sizes(size: int) -> None:
    """Reject population sizes that cannot represent two real communities."""
    with pytest.raises(ValueError, match="greater than one"):
        two_population_hierarchy(size)


def test_hierarchy_level_normalises_names_and_indices() -> None:
    """Normalise harmless whitespace and community index ordering."""
    level = HierarchyLevel("  fine  ", ((2, 0, 1), (5, 4, 3)))
    assert level.name == "fine"
    assert level.communities == ((0, 1, 2), (3, 4, 5))


@pytest.mark.parametrize(
    ("name", "communities", "message"),
    [
        (" ", ((0,),), "name"),
        ("fine", (), "at least one"),
        ("fine", ((),), "non-empty"),
        ("fine", ((0, 0),), "repeats"),
        ("fine", ((0, 1), (1, 2)), "overlaps"),
    ],
)
def test_hierarchy_level_rejects_malformed_communities(
    name: str,
    communities: tuple[tuple[int, ...], ...],
    message: str,
) -> None:
    """Reject blank, empty, repeated, overlapping, or non-integer communities."""
    with pytest.raises(ValueError, match=message):
        HierarchyLevel(name, communities)

    with pytest.raises(ValueError, match="indices must be integers"):
        HierarchyLevel("fine", ((0, 1.5),))


def test_multiscale_hierarchy_rejects_invalid_node_count_and_empty_levels() -> None:
    """Reject invalid node counts and hierarchies without any scale."""
    fine = HierarchyLevel("fine", ((0,), (1,)))
    with pytest.raises(ValueError, match="greater than one"):
        MultiscaleHierarchy(1, (fine,))
    with pytest.raises(ValueError, match="greater than one"):
        MultiscaleHierarchy(2.5, (fine,))
    with pytest.raises(ValueError, match="at least one level"):
        MultiscaleHierarchy(2, ())


def test_multiscale_hierarchy_rejects_duplicate_names_and_incomplete_partition() -> None:
    """Require unique scale names and a complete partition at every scale."""
    fine = HierarchyLevel("fine", ((0,), (1,)))
    duplicate = HierarchyLevel("fine", ((0, 1),))
    with pytest.raises(ValueError, match="names must be unique"):
        MultiscaleHierarchy(2, (fine, duplicate))
    incomplete = HierarchyLevel("incomplete", ((0,),))
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        MultiscaleHierarchy(2, (incomplete,))


def test_multiscale_hierarchy_rejects_non_nested_partitions() -> None:
    """Reject crossed partitions that violate the fine-to-coarse contract."""
    fine = HierarchyLevel("fine", ((0, 1), (2, 3)))
    crossed = HierarchyLevel("crossed", ((0, 2), (1, 3)))
    with pytest.raises(ValueError, match="not nested"):
        MultiscaleHierarchy(4, (fine, crossed))


def test_hierarchy_target_validates_values_weight_and_name() -> None:
    """Validate target names, bounded order parameters, and finite weights."""
    target = HierarchyTarget(" population ", (1, 0.4), weight=2)
    assert target.level_name == "population"
    assert target.order_parameters == (1.0, 0.4)
    assert target.weight == 2.0
    for values in ((), (-0.1,), (1.1,), (float("nan"),)):
        with pytest.raises(ValueError, match="order_parameters"):
            HierarchyTarget("population", values)
    with pytest.raises(ValueError, match="level_name"):
        HierarchyTarget(" ", (1.0,))
    with pytest.raises(ValueError, match="weight"):
        HierarchyTarget("population", (1.0,), weight=-1.0)


def test_control_specification_validates_target_binding() -> None:
    """Bind each unique target row to an existing hierarchy scale."""
    hierarchy = two_population_hierarchy(2)
    specification = ChimeraControlSpecification(
        hierarchy,
        (
            HierarchyTarget("population", (1.0, 0.4)),
            HierarchyTarget("ensemble", (0.65,), weight=0.5),
        ),
    )
    assert specification.claim_boundary == CHIMERA_CONTROL_CLAIM_BOUNDARY

    with pytest.raises(ValueError, match="at least one target"):
        ChimeraControlSpecification(hierarchy, ())
    duplicate = HierarchyTarget("population", (1.0, 0.4))
    with pytest.raises(ValueError, match="must be unique"):
        ChimeraControlSpecification(hierarchy, (duplicate, duplicate))
    with pytest.raises(KeyError, match="unknown hierarchy level"):
        ChimeraControlSpecification(hierarchy, (HierarchyTarget("missing", (1.0,)),))
    with pytest.raises(ValueError, match="requires 2"):
        ChimeraControlSpecification(hierarchy, (HierarchyTarget("population", (1.0,)),))
    with pytest.raises(ValueError, match="claim_boundary"):
        ChimeraControlSpecification(hierarchy, (duplicate,), claim_boundary=" ")
