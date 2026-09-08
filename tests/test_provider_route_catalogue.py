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
import os
from dataclasses import replace
from pathlib import Path

import pytest

from scpn_quantum_control.hardware import hal as hal_module
from scpn_quantum_control.hardware import provider_capability_core, provider_capability_discovery
from scpn_quantum_control.hardware.aggregators import (
    built_in_aggregator_provider_routes,
)
from scpn_quantum_control.hardware.hal import built_in_backend_profiles
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


def test_discovery_facade_exposes_and_executes_the_same_catalogue() -> None:
    """Consumers use the original facade without a second inventory implementation."""
    for name in (
        "DIRECT_AGGREGATOR",
        "ROUTE_VERBS",
        "ROUTE_CATALOGUE_CONTRACT",
        "RouteVerb",
        "RouteVerbSupport",
        "ProviderRouteCatalogueEntry",
        "build_provider_route_catalogue",
    ):
        assert name in provider_capability_discovery.__all__
        assert getattr(provider_capability_discovery, name) is getattr(
            provider_capability_core, name
        )
    entries = provider_capability_discovery.build_provider_route_catalogue(
        observed_at=_OBSERVED_AT
    )
    assert entries == build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    assert entries and all(entry.unverified for entry in entries)


def test_added_metadata_preserves_positional_conformance_arguments() -> None:
    """New keyword-only metadata cannot reinterpret existing root/owner arguments."""
    record = RouteVerbSupport(
        verb="metadata",
        observed=True,
        observed_on=_OBSERVED_AT,
        conformance_owner="tests/test_provider_capability_cloud_adapters.py",
    )
    row = ProviderRouteCatalogueEntry(
        "direct/iqm",
        "iqm",
        None,
        "iqm_cloud",
        "superconducting_gate_model",
        _OBSERVED_AT,
        tuple(
            record if verb == "metadata" else RouteVerbSupport(verb=verb) for verb in ROUTE_VERBS
        ),
        True,
        _REPO_ROOT,
        {("direct/iqm", "metadata"): "tests/test_provider_capability_cloud_adapters.py"},
    )
    assert row.observed_verbs == ("metadata",) and row.sdk_package is None


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
        conformance_root=_REPO_ROOT,
        conformance_owners={
            ("direct/iqm", "metadata"): "tests/test_hardware_hal_iqm_adapters.py",
        },
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


def test_inventory_does_not_read_credential_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured credential references are projected without reading secret values."""

    def reject_environment_access(*args: object, **kwargs: object) -> str:
        raise AssertionError("inventory read the credential environment")

    with monkeypatch.context() as isolated:
        isolated.setattr(type(os.environ), "__getitem__", reject_environment_access)
        isolated.setattr(os, "getenv", reject_environment_access)
        entries = provider_capability_discovery.build_provider_route_catalogue(
            observed_at=_OBSERVED_AT
        )
        payload = json.dumps([entry.to_dict() for entry in entries])
    assert entries and "credential_configuration_refs" in payload


def test_every_observed_verb_names_an_existing_conformance_owner() -> None:
    """An advertised demonstration must resolve to a real test file in the tree."""
    entries = build_provider_route_catalogue(
        observed_at=_OBSERVED_AT,
        conformance_root=_REPO_ROOT,
        conformance_owners={
            ("direct/iqm", "metadata"): "tests/test_hardware_hal_iqm_adapters.py",
            ("qbraid/iqm", "metadata"): "tests/test_hardware_hal_qbraid_adapters.py",
        },
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
        assert entry.target_family == route.target_family
        profile = next(p for p in built_in_backend_profiles() if p.backend_id == route.backend_id)
        assert entry.modality == profile.modality
        assert entry.sdk_package == route.sdk_package == profile.sdk_package
        assert entry.adapter_module == route.adapter_module
        assert entry.credential_configuration_refs
        assert entry.submit_requires_approval == route.submit_requires_approval
        if route.aggregator == DIRECT_AGGREGATOR:
            assert entry.broker is None
        else:
            assert entry.broker == route.aggregator


def test_modality_is_physical_profile_metadata_not_a_provider_label() -> None:
    """Gate, annealing, photonic and dynamic routes preserve HAL semantics."""
    entries = build_provider_route_catalogue(observed_at=_OBSERVED_AT)
    expected = {
        "direct/iqm": "superconducting_gate_model",
        "direct/dwave": "quantum_annealing",
        "direct/quandela": "photonic_gate_model",
        "direct/quera": "neutral_atom_analog",
        "qbraid/iqm": "provider_agnostic_runtime",
    }
    for route_id, modality in expected.items():
        row = _entry(entries, route_id)
        assert row.modality == row.inventory_key[3] == modality
        payload = json.loads(json.dumps(row.to_dict()))
        assert payload["contract"] == "provider_route_catalogue.v2"
        assert payload["modality"] == modality
        assert payload["sdk_package"] == row.sdk_package
        assert payload["credential_configuration_refs"] == list(
            row.credential_configuration_refs or ()
        )
        assert row.unverified


def test_route_aliases_keep_separate_identity_and_observation_bindings() -> None:
    """Shared backend identity cannot merge an alias's observation into its peer."""
    owner = "tests/test_provider_capability_cloud_adapters.py"
    observed_route = "strangeworks/ibm_quantum"
    entries = build_provider_route_catalogue(
        observed_at=_OBSERVED_AT,
        evidence={
            observed_route: (
                RouteVerbSupport(
                    verb="metadata",
                    observed=True,
                    observed_on=_OBSERVED_AT,
                    conformance_owner=owner,
                ),
            )
        },
        conformance_root=_REPO_ROOT,
        conformance_owners={(observed_route, "metadata"): owner},
    )
    direct = _entry(entries, observed_route)
    alias = _entry(entries, "strangeworks/qiskit_runtime")
    assert direct.inventory_key == alias.inventory_key
    assert direct.route_key != alias.route_key
    assert len({row.route_key for row in entries}) == len(entries)
    assert not direct.unverified
    assert alias.unverified and alias.support("metadata").observed is None
    payload = json.loads(json.dumps([direct.to_dict(), alias.to_dict()]))
    assert payload[0]["route_id"] != payload[1]["route_id"]
    assert payload[0]["unverified"] is False and payload[1]["unverified"] is True


def test_custom_backend_requires_explicit_profile_without_fabricated_credentials() -> None:
    """Unknown provider metadata must come from an explicit profile, not its name."""
    base = next(r for r in built_in_aggregator_provider_routes() if r.route_id == "direct/iqm")
    route = replace(
        base, route_id="direct/custom", backend_id="custom", adapter_module="custom.provider"
    )
    with pytest.raises(ValueError, match="missing backend profile"):
        build_provider_route_catalogue(observed_at=_OBSERVED_AT, routes=(route,))
    profile = replace(
        next(p for p in built_in_backend_profiles() if p.backend_id == base.backend_id),
        backend_id="custom",
        modality="declared_custom_modality",
    )
    (row,) = build_provider_route_catalogue(
        observed_at=_OBSERVED_AT, routes=(route,), profiles=(profile,)
    )
    assert row.modality == "declared_custom_modality" and row.unverified
    assert row.credential_configuration_refs is None
    assert row.to_dict()["credential_configuration_refs"] is None
    with pytest.raises(ValueError, match="repeat a backend_id"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT, routes=(route,), profiles=(profile, profile)
        )
    with pytest.raises(ValueError, match="SDK disagrees"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            routes=(replace(route, sdk_package="other-sdk"),),
            profiles=(profile,),
        )


def test_declared_support_requires_a_conformance_owner() -> None:
    """A dated declaration must still identify its direct conformance owner."""
    with pytest.raises(ValueError, match="conformance_owner"):
        RouteVerbSupport(
            verb="submit",
            declared=True,
            declared_source="route table",
            declared_on=_OBSERVED_AT,
        )


def test_declared_source_cannot_be_only_whitespace() -> None:
    """A blank provenance label does not identify a source."""
    with pytest.raises(ValueError, match="declared_source"):
        RouteVerbSupport(
            verb="metadata",
            declared=True,
            declared_source=" \t ",
            declared_on=_OBSERVED_AT,
            conformance_owner="tests/test_provider_route_catalogue.py",
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"verbs": []}, "verbs"),
        ({"verbs": (None,)}, "verbs"),
        ({"submit_requires_approval": 0}, "submit_requires_approval"),
        ({"submit_requires_approval": 1}, "submit_requires_approval"),
        ({"submit_requires_approval": "false"}, "submit_requires_approval"),
        ({"submit_requires_approval": None}, "submit_requires_approval"),
    ],
)
def test_catalogue_rows_refuse_mutable_verbs_and_non_boolean_approval_flags(
    changes: dict[str, object],
    message: str,
) -> None:
    """Frozen rows must not admit mutable evidence or truthy approval labels."""
    row = build_provider_route_catalogue(observed_at=_OBSERVED_AT)[0]
    if isinstance(changes.get("verbs"), list):
        changes = {"verbs": list(row.verbs)}
    with pytest.raises(ValueError, match=message):
        replace(row, **changes)  # type: ignore[arg-type]  # Invalid constructor inputs exercise runtime rejection.


@pytest.mark.parametrize(
    "owner",
    [
        "tests/test_missing_conformance_owner.py",
        "tests/test_provider_route_catalogue.py/../test_hardware_hal_iqm_adapters.py",
        "tests/test_provider_route_catalogue.py/../../docs/hardware_guide.md",
        "tests/test_provider_route_catalogue.py/",
        "tests/test_provider_route_catalogue.txt",
    ],
)
def test_positive_evidence_refuses_unresolvable_owner(owner: str) -> None:
    """Even a registered owner must resolve to a canonical Python test file."""
    with pytest.raises(ValueError, match="conformance_owner"):
        record = RouteVerbSupport(
            verb="metadata",
            observed=True,
            observed_on=_OBSERVED_AT,
            conformance_owner=owner,
        )
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/iqm": (record,)},
            conformance_root=_REPO_ROOT,
            conformance_owners={("direct/iqm", "metadata"): owner},
        )


@pytest.mark.parametrize(
    "route_id, verb, owner",
    [
        ("qbraid/iqm", "metadata", "tests/test_hardware_hal_iqm_adapters.py"),
        ("direct/iqm", "submit", "tests/test_hardware_hal_iqm_adapters.py"),
        ("direct/iqm", "metadata", "tests/test_hardware_hal_qbraid_adapters.py"),
    ],
)
def test_evidence_cannot_borrow_another_route_or_verbs_owner(
    route_id: str,
    verb: str,
    owner: str,
) -> None:
    """A real but misattributed test cannot qualify the submitted observation."""
    record = RouteVerbSupport(
        verb="metadata",
        observed=True,
        observed_on=_OBSERVED_AT,
        conformance_owner="tests/test_hardware_hal_iqm_adapters.py",
    )
    with pytest.raises(ValueError, match="conformance_owner.*registered"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/iqm": (record,)},
            conformance_root=_REPO_ROOT,
            conformance_owners={(route_id, verb): owner},
        )


def test_positive_catalogue_and_direct_entry_require_explicit_authority() -> None:
    """Neither public construction path may qualify an unanchored assertion."""
    record = RouteVerbSupport(
        verb="metadata",
        observed=True,
        observed_on=_OBSERVED_AT,
        conformance_owner="tests/test_hardware_hal_iqm_adapters.py",
    )
    with pytest.raises(ValueError, match="conformance"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/iqm": (record,)},
        )
    with pytest.raises(ValueError, match="conformance"):
        ProviderRouteCatalogueEntry(
            route_id="direct/iqm",
            provider="iqm",
            broker=None,
            device="iqm_cloud",
            modality="iqm",
            observed_at=_OBSERVED_AT,
            verbs=tuple(
                record if verb == "metadata" else RouteVerbSupport(verb=verb)
                for verb in ROUTE_VERBS
            ),
        )


@pytest.mark.parametrize("relative", [False, True])
def test_positive_support_refuses_missing_or_relative_root(
    tmp_path: Path,
    relative: bool,
) -> None:
    """No implicit checkout or missing directory may anchor evidence."""
    owner = "tests/test_hardware_hal_iqm_adapters.py"
    record = RouteVerbSupport(
        verb="metadata",
        observed=True,
        observed_on=_OBSERVED_AT,
        conformance_owner=owner,
    )
    with pytest.raises(ValueError, match="conformance_root"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/iqm": (record,)},
            conformance_root=Path(".") if relative else tmp_path / "missing",
            conformance_owners={("direct/iqm", "metadata"): owner},
        )


@pytest.mark.parametrize("as_directory", [False, True])
def test_owner_reference_refuses_symlink_escape_and_directory(
    tmp_path: Path,
    as_directory: bool,
) -> None:
    """An owner-looking name cannot resolve outside the designated source tree."""
    owner = "tests/test_hardware_hal_iqm_adapters.py"
    candidate = tmp_path / owner
    candidate.parent.mkdir()
    if as_directory:
        candidate.mkdir()
    else:
        candidate.symlink_to(_REPO_ROOT / owner)
    record = RouteVerbSupport(
        verb="metadata",
        observed=True,
        observed_on=_OBSERVED_AT,
        conformance_owner=owner,
    )
    with pytest.raises(ValueError, match="conformance_owner.*regular"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/iqm": (record,)},
            conformance_root=tmp_path,
            conformance_owners={("direct/iqm", "metadata"): owner},
        )


def test_declared_owner_resolution_never_promotes_observed_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source declarations resolve independently of cwd without inventing runs."""
    monkeypatch.chdir(tmp_path)
    owner = "tests/test_hardware_hal_iqm_adapters.py"
    record = RouteVerbSupport(
        verb="metadata",
        declared=True,
        declared_source="hardware/aggregators.py",
        declared_on=_OBSERVED_AT,
        conformance_owner=owner,
    )
    with pytest.raises(ValueError, match="conformance_owner.*registered"):
        build_provider_route_catalogue(
            observed_at=_OBSERVED_AT,
            evidence={"direct/iqm": (record,)},
            conformance_root=_REPO_ROOT,
        )
    entries = build_provider_route_catalogue(
        observed_at=_OBSERVED_AT,
        evidence={"direct/iqm": (record,)},
        conformance_root=_REPO_ROOT,
        conformance_owners={("direct/iqm", "metadata"): owner},
    )
    row = _entry(entries, "direct/iqm")
    assert row.support("metadata").declared is True
    assert row.support("metadata").observed is None
    assert row.unverified and row.observed_verbs == ()
    exported = json.dumps(row.to_dict())
    assert str(_REPO_ROOT) not in exported
    assert "conformance_root" not in exported and "conformance_owners" not in exported


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


@pytest.mark.parametrize(
    "value",
    [
        "2026-99-99",
        "2026-00-01",
        "2026-01-00",
        "2026-04-31",
        "2026-02-29",
        "1900-02-29",
        "0000-01-01",
        "10000-01-01",
        "2026-9-05",
        "2026-09-5",
        "2026-09-05\n",
        " 2026-09-05",
        "２０２６-０９-０５",
        "20260905",
        "2026-W36-6",
        "2026-09-05T00:00:00",
    ],
)
def test_invalid_calendar_dates_are_refused_at_public_boundaries(value: str) -> None:
    """Inventory builders and direct rows reject impossible or noncanonical dates."""
    with pytest.raises(ValueError, match="observed_at"):
        build_provider_route_catalogue(observed_at=value)
    with pytest.raises(ValueError, match="observed_at"):
        ProviderRouteCatalogueEntry(
            route_id="direct/iqm",
            provider="iqm",
            broker=None,
            device="iqm_cloud",
            modality="iqm",
            observed_at=value,
            verbs=tuple(RouteVerbSupport(verb=verb) for verb in ROUTE_VERBS),
        )
    for support in (True, False, None):
        with pytest.raises(ValueError, match="declared_on"):
            RouteVerbSupport(
                verb="metadata",
                declared=support,
                declared_source="hardware/aggregators.py",
                declared_on=value,
            )
        with pytest.raises(ValueError, match="observed_on"):
            RouteVerbSupport(
                verb="metadata",
                observed=support,
                observed_on=value,
                conformance_owner="tests/test_provider_route_catalogue.py",
            )


@pytest.mark.parametrize("value", ["0001-01-01", "2000-02-29", "2024-02-29", "9999-12-31"])
def test_valid_calendar_dates_round_trip_without_promoting_support(value: str) -> None:
    """Valid dates survive JSON export without depending on the workstation clock."""
    evidence = RouteVerbSupport(
        verb="metadata",
        declared_on=value,
        observed_on=value,
    )
    row = _entry(
        build_provider_route_catalogue(
            observed_at=value,
            evidence={"direct/iqm": (evidence,)},
        ),
        "direct/iqm",
    )
    assert row.inventory_key[-1] == value
    assert row.unverified
    exported = json.loads(json.dumps(row.support("metadata").to_dict()))
    assert exported["declared_on"] == exported["observed_on"] == value
    assert exported["declared"] is None and exported["observed"] is None
    positive = RouteVerbSupport(
        verb="metadata",
        declared=True,
        declared_on=value,
        declared_source="hardware/aggregators.py",
        observed=True,
        observed_on=value,
        conformance_owner="tests/test_provider_route_catalogue.py",
    )
    assert positive.to_dict()["declared_on"] == value
    assert positive.to_dict()["observed_on"] == value


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
        _entry(entries, "direct/iqm").support("teleport")  # type: ignore[arg-type]  # invalid verb rejection
