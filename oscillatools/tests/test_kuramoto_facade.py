# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the unified Kuramoto toolkit facade
r"""Tests for :mod:`oscillatools`.

The facade is checked for exact parity with :mod:`oscillatools.accel` — it re-exports
every toolkit symbol and each name resolves to the very same object — and the capability map is
checked to partition that surface into disjoint, complete groups. The map's immutability, the
``describe`` summaries and a representative direct import are covered.
"""

from __future__ import annotations

import pytest

import oscillatools as kuramoto
import oscillatools.accel as accel

_HELPERS = {"KURAMOTO_CAPABILITIES", "capabilities", "describe"}


# --------------------------------------------------------------------------- re-export parity


def test_facade_reexports_exactly_the_accel_toolkit() -> None:
    reexported = set(kuramoto.__all__) - _HELPERS
    assert reexported == set(accel.__all__)


def test_every_reexported_name_is_the_same_object() -> None:
    for name in accel.__all__:
        assert getattr(kuramoto, name) is getattr(accel, name)


def test_facade_all_has_no_duplicates() -> None:
    assert len(kuramoto.__all__) == len(set(kuramoto.__all__))


def test_helpers_are_exported() -> None:
    assert set(kuramoto.__all__) >= _HELPERS


# --------------------------------------------------------------------------- capability map


def test_capability_map_partitions_the_toolkit() -> None:
    union: set[str] = set()
    for symbols in kuramoto.KURAMOTO_CAPABILITIES.values():
        group = set(symbols)
        assert union.isdisjoint(group)  # groups are disjoint
        union |= group
    assert union == set(accel.__all__)  # and complete


def test_capability_groups_are_non_empty_tuples() -> None:
    for group, symbols in kuramoto.KURAMOTO_CAPABILITIES.items():
        assert isinstance(group, str)
        assert isinstance(symbols, tuple)
        assert len(symbols) > 0


def test_capabilities_returns_the_immutable_map() -> None:
    mapping = kuramoto.capabilities()
    assert mapping is kuramoto.KURAMOTO_CAPABILITIES
    with pytest.raises(TypeError):
        mapping["forces"] = ()  # type: ignore[index]


def test_expected_groups_are_present() -> None:
    assert set(kuramoto.KURAMOTO_CAPABILITIES) == {
        "forces",
        "integrators",
        "observables",
        "diagnostics",
        "analysis",
        "control_and_design",
        "visualisation",
        "types",
        "dispatch",
        "tier_introspection",
    }


# --------------------------------------------------------------------------- describe


def test_describe_overview_lists_every_group_with_counts() -> None:
    overview = kuramoto.describe()
    for group, symbols in kuramoto.KURAMOTO_CAPABILITIES.items():
        assert f"{group}: {len(symbols)} symbols" in overview


def test_describe_group_lists_its_symbols() -> None:
    listing = kuramoto.describe("diagnostics").splitlines()
    assert set(listing) == set(kuramoto.KURAMOTO_CAPABILITIES["diagnostics"])


def test_describe_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown capability group"):
        kuramoto.describe("nonexistent")


# --------------------------------------------------------------------------- direct import


def test_representative_symbols_import_and_are_callable() -> None:
    from oscillatools import (
        chimera_index,
        kuramoto_dopri_trajectory,
        order_parameter,
        stability_spectrum,
    )

    assert callable(order_parameter)
    assert callable(kuramoto_dopri_trajectory)
    assert callable(chimera_index)
    assert callable(stability_spectrum)
