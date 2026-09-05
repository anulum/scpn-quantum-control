# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Route Catalogue Tests
"""Dedicated owner tests for the no-submit provider route inventory.

Every case drives the public catalogue entry points. The inventory must keep
provider and broker identities distinct, must preserve unknown support as
unknown, and must reach no provider: the submit-path guard below turns any
submission attempt during inventory into a test failure rather than a silent
network call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.hardware import hal as hal_module
from scpn_quantum_control.hardware.aggregators import (
    built_in_aggregator_provider_routes,
)
from scpn_quantum_control.hardware.provider_capability_core import (
    DIRECT_AGGREGATOR,
    ROUTE_CATALOGUE_CONTRACT,
    ROUTE_VERBS,
    ProviderRouteCatalogueEntry,
    RouteVerbSupport,
    build_provider_route_catalogue,
)

_OBSERVED_AT = "2026-09-05"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _entry(
    entries: tuple[ProviderRouteCatalogueEntry, ...], route_id: str
) -> ProviderRouteCatalogueEntry:
    """Return the single inventory row for one declared route identifier."""
    matches = [entry for entry in entries if entry.route_id == route_id]
    assert len(matches) == 1, route_id
    return matches[0]


def test_broker_hosted_iqm_route_stays_distinct_from_direct_iqm() -> None:
    """A brokered route is its own inventory row, never merged with the direct one."""
    entries = build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    iqm = [entry for entry in entries if entry.provider == "iqm"]
    assert len(iqm) >= 2
    assert len({entry.inventory_key for entry in iqm}) == len(iqm)

    direct = _entry(entries, "direct/iqm")
    brokered = _entry(entries, "qbraid/iqm")
    assert direct.provider == brokered.provider == "iqm"
    assert direct.is_direct and direct.broker is None
    assert not brokered.is_direct and brokered.broker == "qbraid"
    assert direct.device != brokered.device
    assert direct.inventory_key != brokered.inventory_key


def test_unknown_support_is_preserved_as_null_everywhere() -> None:
    """A route with no evidence reports unknown, not unsupported and not ready."""
    entries = build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    unevidenced = _entry(entries, "direct/iqm")
    assert unevidenced.unverified
    assert unevidenced.observed_verbs == ()
    for verb in ROUTE_VERBS:
        record = unevidenced.support(verb)
        assert record.declared is None
        assert record.observed is None
        assert record.conformance_owner is None
    payload = json.loads(json.dumps(unevidenced.to_dict()))
    assert payload["contract"] == ROUTE_CATALOGUE_CONTRACT
    assert all(row["observed"] is None and row["declared"] is None for row in payload["verbs"])


def test_partial_evidence_leaves_other_verbs_unknown() -> None:
    """Recording one operation must not imply anything about the others."""
    entries = build_provider_route_catalogue(
        observed_at=_OBSERVED_AT,
        evidence={
            "direct/iqm": (
                RouteVerbSupport(
                    verb="metadata",
                    declared=True,
                    declared_source="hardware/aggregators.py route table",
                    declared_on=_OBSERVED_AT,
                    observed=True,
                    observed_on=_OBSERVED_AT,
                    conformance_owner="tests/test_hardware_hal_iqm_adapters.py",
                ),
            )
        },
    )
    entry = _entry(entries, "direct/iqm")
    assert entry.observed_verbs == ("metadata",)
    assert not entry.unverified
    for verb in ROUTE_VERBS:
        if verb == "metadata":
            continue
        assert entry.support(verb).observed is None
        assert entry.support(verb).declared is None


def test_inventory_performs_no_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the whole inventory must not reach a provider submit path."""

    def _fail_on_submit(*args: object, **kwargs: object) -> None:
        raise AssertionError("route inventory attempted a provider submission")

    monkeypatch.setattr(hal_module.HardwareAbstractionLayer, "submit", _fail_on_submit)
    monkeypatch.setattr(hal_module.HardwareAbstractionLayer, "result", _fail_on_submit)
    monkeypatch.setattr(hal_module.HardwareAbstractionLayer, "cancel", _fail_on_submit)
    entries = build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    assert len(entries) == len(built_in_aggregator_provider_routes())
    assert all(entry.observed_at == _OBSERVED_AT for entry in entries)


def test_every_observed_verb_names_an_existing_conformance_owner() -> None:
    """An advertised demonstration must resolve to a real test file in the tree."""
    entries = build_provider_route_catalogue(
        observed_at=_OBSERVED_AT,
        evidence={
            "direct/iqm": (
                RouteVerbSupport(
                    verb="metadata",
                    declared=True,
                    declared_source="hardware/aggregators.py route table",
                    declared_on=_OBSERVED_AT,
                    observed=True,
                    observed_on=_OBSERVED_AT,
                    conformance_owner="tests/test_hardware_hal_iqm_adapters.py",
                ),
            ),
            "qbraid/iqm": (
                RouteVerbSupport(
                    verb="metadata",
                    declared=True,
                    declared_source="hardware/aggregators.py route table",
                    declared_on=_OBSERVED_AT,
                    observed=True,
                    observed_on=_OBSERVED_AT,
                    conformance_owner="tests/test_hardware_hal_qbraid_adapters.py",
                ),
            ),
        },
    )
    observed = [
        (entry.route_id, entry.support(verb)) for entry in entries for verb in entry.observed_verbs
    ]
    assert observed
    for route_id, record in observed:
        assert record.conformance_owner is not None, route_id
        assert (_REPO_ROOT / record.conformance_owner).is_file(), record.conformance_owner
        assert record.observed_on is not None


def test_every_route_is_inventoried_with_canonical_verbs() -> None:
    """Each declared route yields exactly one row covering every operation once."""
    routes = built_in_aggregator_provider_routes()
    entries = build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    assert len(entries) == len(routes)
    assert len({entry.route_id for entry in entries}) == len(entries)
    for entry in entries:
        assert tuple(record.verb for record in entry.verbs) == ROUTE_VERBS
    for route in routes:
        entry = _entry(entries, route.route_id)
        assert entry.provider == route.provider
        assert entry.device == route.backend_id
        assert entry.modality == route.target_family
        assert entry.submit_requires_approval == route.submit_requires_approval
        if route.aggregator == DIRECT_AGGREGATOR:
            assert entry.broker is None
        else:
            assert entry.broker == route.aggregator


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"verb": "teleport"}, "unknown route verb"),
        ({"verb": "submit", "declared": True}, "declared_source"),
        (
            {"verb": "submit", "declared": True, "declared_source": "route table"},
            "declared_on",
        ),
        (
            {
                "verb": "submit",
                "declared": True,
                "declared_source": "route table",
                "declared_on": "2026-9-5",
            },
            "YYYY-MM-DD",
        ),
        ({"verb": "submit", "observed": True}, "observed_on"),
        (
            {"verb": "submit", "observed": True, "observed_on": _OBSERVED_AT},
            r"conformance_owner must be non-empty text",
        ),
        (
            {
                "verb": "submit",
                "observed": True,
                "observed_on": _OBSERVED_AT,
                "conformance_owner": "",
            },
            r"conformance_owner must be non-empty text",
        ),
        (
            {
                "verb": "submit",
                "observed": True,
                "observed_on": _OBSERVED_AT,
                "conformance_owner": "docs/hardware_guide.md",
            },
            r"must be a tests/test_\* path",
        ),
        (
            {
                "verb": "submit",
                "declared": False,
                "observed": True,
                "observed_on": _OBSERVED_AT,
                "conformance_owner": "tests/test_hardware_hal.py",
            },
            "contradicts",
        ),
        ({"verb": "submit", "declared": 1}, "True, False or None"),
    ],
)
def test_incomplete_or_contradictory_evidence_is_refused(
    kwargs: dict[str, object], message: str
) -> None:
    """Provenance gaps fail closed instead of producing a half-attributed row."""
    with pytest.raises(ValueError, match=message):
        RouteVerbSupport(**kwargs)  # type: ignore[arg-type]  # deliberately malformed


def test_direct_route_may_not_record_the_direct_label_as_a_broker() -> None:
    """The direct aggregator label must be absent, not stored as a broker name."""
    with pytest.raises(ValueError, match="broker as None"):
        ProviderRouteCatalogueEntry(
            route_id="direct/iqm",
            provider="iqm",
            broker=DIRECT_AGGREGATOR,
            device="iqm_cloud",
            modality="iqm",
            observed_at=_OBSERVED_AT,
            verbs=tuple(RouteVerbSupport(verb=verb) for verb in ROUTE_VERBS),
        )


@pytest.mark.parametrize(
    ("verbs", "message"),
    [
        ((), "one record per operation"),
        (tuple(RouteVerbSupport(verb=verb) for verb in reversed(ROUTE_VERBS)), "order"),
        (
            tuple(RouteVerbSupport(verb=verb) for verb in ROUTE_VERBS)
            + (RouteVerbSupport(verb="metadata"),),
            "one record per operation",
        ),
    ],
)
def test_verb_coverage_must_be_exact_and_ordered(
    verbs: tuple[RouteVerbSupport, ...], message: str
) -> None:
    """A row cannot omit, duplicate or reorder the canonical operation set."""
    with pytest.raises(ValueError, match=message):
        ProviderRouteCatalogueEntry(
            route_id="direct/iqm",
            provider="iqm",
            broker=None,
            device="iqm_cloud",
            modality="iqm",
            observed_at=_OBSERVED_AT,
            verbs=verbs,
        )


def test_evidence_for_an_unknown_route_is_refused() -> None:
    """Evidence naming a route outside the inventory fails rather than vanishing."""
    with pytest.raises(ValueError, match="absent from the inventory"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/nonexistent": (RouteVerbSupport(verb="metadata"),)},
        )


def test_repeated_verb_evidence_for_one_route_is_refused() -> None:
    """Two records for the same operation are contradictory input, not a merge."""
    with pytest.raises(ValueError, match="repeats a verb"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={
                "direct/iqm": (
                    RouteVerbSupport(verb="metadata"),
                    RouteVerbSupport(verb="metadata"),
                )
            },
        )


@pytest.mark.parametrize("observed_at", ["", "2026-13-01x", "05-09-2026", "2026/09/05"])
def test_malformed_observation_date_is_refused(observed_at: str) -> None:
    """The inventory timestamp is part of the key and must be well formed."""
    with pytest.raises(ValueError, match="observed_at"):
        build_provider_route_catalogue(observed_at=observed_at)


def test_duplicate_route_identifiers_are_refused() -> None:
    """A repeated route identifier would collapse two rows into one."""
    routes = built_in_aggregator_provider_routes()
    duplicated = (routes[0], routes[0])
    with pytest.raises(ValueError, match="repeat a route_id"):
        build_provider_route_catalogue(observed_at=_OBSERVED_AT, routes=duplicated)


def test_unknown_verb_lookup_raises_key_error() -> None:
    """Looking up an operation outside the canonical set is an error, not None."""
    entries = build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    with pytest.raises(KeyError):
        _entry(entries, "direct/iqm").support("teleport")  # type: ignore[arg-type]
