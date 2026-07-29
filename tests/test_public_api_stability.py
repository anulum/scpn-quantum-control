# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for public API stability programme (BL-97)
"""Real-surface tests for ``scpn_quantum_control.public_api_stability``."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest

import scpn_quantum_control.public_api_stability as public_api_stability
from scpn_quantum_control.public_api_stability import (
    PUBLIC_API_STABILITY_CLAIM_BOUNDARY,
    PUBLIC_API_STABILITY_SCHEMA,
    BreakingChangeDecision,
    BreakingChangeKind,
    DeprecationProbe,
    PathClassification,
    PublicApiSymbolRecord,
    assert_public_api_stability_integrity,
    build_public_api_stability_registry,
    classify_api_path,
    deprecated_public,
    get_public_api_symbol,
    iter_public_api_symbols,
    list_public_api_symbol_ids,
    probe_deprecation,
    validate_breaking_change,
    version_compatibility_note,
)


def test_list_is_non_empty_deterministic_and_covers_classes() -> None:
    """Keep catalogue ordering deterministic and every stability class represented."""
    ids = list_public_api_symbol_ids()
    assert len(ids) >= 10
    assert ids == list_public_api_symbol_ids()
    assert len(set(ids)) == len(ids)
    classes = {row.stability_class for row in iter_public_api_symbols()}
    assert "semver_stable" in classes
    assert "experimental_workbench" in classes
    assert "deprecated" in classes
    assert "internal" in classes


def test_get_known_and_unknown_fail_closed() -> None:
    """Resolve declared symbols while refusing blank and unknown identifiers."""
    row = get_public_api_symbol("scpn_quantum_control.stable_core.Problem")
    assert row.stability_class == "semver_stable"
    assert row.visibility == "public"
    assert row.deprecation_state == "active"
    assert row.claim_boundary == PUBLIC_API_STABILITY_CLAIM_BOUNDARY
    with pytest.raises(ValueError, match="non-empty"):
        get_public_api_symbol("  ")
    with pytest.raises(ValueError, match="unknown public API symbol_id"):
        get_public_api_symbol("not.a.real.symbol")


def test_classify_public_vs_internal() -> None:
    """Classify durable, workbench, internal, and undeclared paths honestly."""
    public = classify_api_path("scpn_quantum_control.stable_core.Problem")
    assert public.visibility == "public"
    assert public.guaranteed_stable is True
    assert public.in_catalogue is True

    internal = classify_api_path("scpn_quantum_control._private_helpers")
    assert internal.visibility == "internal"
    assert internal.guaranteed_stable is False
    assert internal.stability_class == "internal"

    workbench = classify_api_path("scpn_quantum_control.scorecard_acceptance_engine")
    assert workbench.visibility == "public"
    assert workbench.guaranteed_stable is False
    assert workbench.stability_class == "experimental_workbench"

    undeclared = classify_api_path("scpn_quantum_control.phase.some_research_helper")
    assert undeclared.guaranteed_stable is False
    assert undeclared.stability_class == "experimental_workbench"
    assert undeclared.in_catalogue is False

    with pytest.raises(ValueError, match="non-empty"):
        classify_api_path("")


def test_undeclared_never_guaranteed_stable() -> None:
    """Bulk workbench paths must not be reported as SemVer-guaranteed."""
    for path in (
        "scpn_quantum_control.phase.something",
        "scpn_quantum_control.analysis.foo",
        "totally.unknown.surface",
    ):
        result = classify_api_path(path)
        assert result.guaranteed_stable is False
        assert result.stability_class != "semver_stable" or result.in_catalogue


def test_build_registry_and_integrity() -> None:
    """Build a complete registry and validate both explicit and default payloads."""
    registry = build_public_api_stability_registry()
    assert registry["schema"] == PUBLIC_API_STABILITY_SCHEMA
    assert registry["blank_entry_count"] == 0
    symbol_count = registry["symbol_count"]
    assert isinstance(symbol_count, int)
    assert symbol_count == len(list_public_api_symbol_ids())
    for key in (
        "semver_stable_count",
        "deprecated_count",
        "internal_count",
        "experimental_workbench_count",
        "public_count",
    ):
        value = registry[key]
        assert isinstance(value, int)
        assert value >= 1 or key == "experimental_workbench_count" and value >= 0
        if key != "experimental_workbench_count":
            assert value >= 1
    validated = assert_public_api_stability_integrity(registry)
    assert validated["symbol_count"] == registry["symbol_count"]
    assert "Narrow durable contract" in str(registry["policy_note"])
    # default integrity path (no payload arg)
    assert assert_public_api_stability_integrity()["blank_entry_count"] == 0


def test_probe_deprecation_active_and_deprecated() -> None:
    """Distinguish active symbols from staged deprecations and reject unknowns."""
    active = probe_deprecation("scpn_quantum_control.stable_core.Problem")
    assert active.is_deprecated is False
    assert active.replacement_target == ""

    deprecated = probe_deprecation("scpn_quantum_control.kuramoto")
    assert deprecated.is_deprecated is True
    assert deprecated.replacement_target == "oscillatools"
    assert "v2" in deprecated.removal_horizon or "major" in deprecated.removal_horizon
    assert "deprecated" in deprecated.warning_message.lower()
    assert "oscillatools" in deprecated.warning_message

    with pytest.raises(ValueError, match="unknown"):
        probe_deprecation("missing.symbol")


def test_validate_breaking_change_fail_closed_without_deprecation() -> None:
    """Refuse unstaged public breaks while permitting governed and internal changes."""
    refuse = validate_breaking_change(
        "scpn_quantum_control.stable_core.Problem",
        change_kind="remove",
    )
    assert refuse.allowed is False
    assert refuse.requires_deprecation is True
    assert "refuse" in refuse.reason

    allow_after = validate_breaking_change(
        "scpn_quantum_control.kuramoto",
        change_kind="remove",
    )
    assert allow_after.allowed is True
    assert allow_after.requires_deprecation is True

    internal = validate_breaking_change(
        "tests.test_public_api_stability",
        change_kind="rename",
    )
    assert internal.allowed is True
    assert internal.requires_deprecation is False

    with pytest.raises(ValueError, match="unknown change_kind"):
        validate_breaking_change(
            "scpn_quantum_control.stable_core.Problem",
            change_kind=cast(BreakingChangeKind, "explode"),
        )


def test_deprecated_public_decorator_emits_warning() -> None:
    """Emit the policy warning and reject blank decorator metadata."""

    @deprecated_public(
        symbol_id="demo.legacy_fn",
        replacement_target="demo.new_fn",
        removal_horizon="v2.0.0",
    )
    def legacy() -> str:
        return "ok"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert legacy() == "ok"
    deprecation_warnings = [
        item for item in caught if issubclass(item.category, DeprecationWarning)
    ]
    assert deprecation_warnings
    assert "demo.legacy_fn" in str(deprecation_warnings[0].message)
    assert "demo.new_fn" in str(deprecation_warnings[0].message)

    with pytest.raises(ValueError, match="replacement_target"):
        deprecated_public(
            symbol_id="x",
            replacement_target="  ",
            removal_horizon="v2",
        )
    with pytest.raises(ValueError, match="symbol_id"):
        deprecated_public(
            symbol_id="",
            replacement_target="y",
            removal_horizon="v2",
        )


def test_version_compatibility_note() -> None:
    """Expose the version, migration, policy, and claim-boundary record."""
    note = version_compatibility_note()
    assert note["schema"] == "public_api_version_compatibility.v1"
    assert "DEPRECATIONS.md" in str(note["deprecation_policy"])
    assert "oscillatools" in str(note["migration_note"])
    assert note["claim_boundary"] == PUBLIC_API_STABILITY_CLAIM_BOUNDARY


def test_iter_filters() -> None:
    """Filter catalogue rows independently by class, state, and visibility."""
    stable = iter_public_api_symbols(stability_class="semver_stable")
    assert stable
    assert all(row.stability_class == "semver_stable" for row in stable)
    deprecated = iter_public_api_symbols(deprecation_state="deprecated")
    assert deprecated
    assert all(row.deprecation_state == "deprecated" for row in deprecated)
    internal = iter_public_api_symbols(visibility="internal")
    assert internal
    assert all(row.visibility == "internal" for row in internal)


def test_record_to_dict_and_dataclasses() -> None:
    """Serialise each public record type without losing decision fields."""
    row = get_public_api_symbol("scpn_quantum_control.accel")
    payload = row.to_dict()
    assert payload["symbol_id"] == "scpn_quantum_control.accel"
    assert payload["stability_class"] == "deprecated"
    assert payload["replacement_target"] == "oscillatools.accel"

    classification = classify_api_path("scpn_quantum_control.accel")
    assert classification.to_dict()["guaranteed_stable"] is False

    probe = probe_deprecation("scpn_quantum_control.accel")
    assert probe.to_dict()["is_deprecated"] is True

    decision = validate_breaking_change("scpn_quantum_control.accel")
    assert decision.to_dict()["allowed"] is True


def test_public_api_symbol_record_validation() -> None:
    """Reject every invalid symbol-record invariant at construction time."""
    with pytest.raises(ValueError, match="symbol_id"):
        PublicApiSymbolRecord(
            symbol_id="",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="s",
        )
    with pytest.raises(ValueError, match="stability_class"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class=cast(Any, "not_a_class"),
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="s",
        )
    with pytest.raises(ValueError, match="deprecation_state"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state=cast(Any, "weird"),
            visibility="public",
            summary="s",
        )
    with pytest.raises(ValueError, match="visibility"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="active",
            visibility=cast(Any, "secret"),
            summary="s",
        )
    with pytest.raises(ValueError, match="owner_surface"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="  ",
            deprecation_state="active",
            visibility="public",
            summary="s",
        )
    with pytest.raises(ValueError, match="summary"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="",
        )
    with pytest.raises(ValueError, match="as_of"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="s",
            as_of="",
        )
    with pytest.raises(ValueError, match="deprecation_state=deprecated"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="deprecated",
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="s",
            replacement_target="y",
            removal_horizon="v2",
        )
    with pytest.raises(ValueError, match="removal_horizon"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="deprecated",
            owner_surface="x",
            deprecation_state="deprecated",
            visibility="public",
            summary="s",
            replacement_target="y",
            removal_horizon="",
        )
    with pytest.raises(ValueError, match="replacement_target"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="deprecated",
            owner_surface="x",
            deprecation_state="deprecated",
            visibility="public",
            summary="s",
            replacement_target="",
            removal_horizon="v2",
        )
    with pytest.raises(ValueError, match="internal"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="internal",
            owner_surface="x",
            deprecation_state="not_applicable",
            visibility="public",
            summary="s",
        )
    with pytest.raises(ValueError, match="not_applicable"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="internal",
            owner_surface="x",
            deprecation_state="active",
            visibility="internal",
            summary="s",
        )
    with pytest.raises(ValueError, match="invalid for this stability_class"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="internal",
            owner_surface="x",
            deprecation_state="deprecated",
            visibility="internal",
            summary="s",
        )
    with pytest.raises(ValueError, match="must use deprecation_state=not_applicable"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="deprecated",
            visibility="internal",
            summary="s",
        )
    with pytest.raises(ValueError, match="replacement_target must be non-empty when present"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="s",
            replacement_target="   ",
        )
    with pytest.raises(ValueError, match="removal_horizon must be non-empty when present"):
        PublicApiSymbolRecord(
            symbol_id="a.b",
            stability_class="semver_stable",
            owner_surface="x",
            deprecation_state="active",
            visibility="public",
            summary="s",
            removal_horizon="   ",
        )


def test_integrity_rejects_blank_and_drift() -> None:
    """Reject malformed rows, duplicate identifiers, counts, and catalogue drift."""
    good = build_public_api_stability_registry()
    assert_public_api_stability_integrity(good)

    bad_blank = dict(good)
    bad_blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_public_api_stability_integrity(bad_blank)

    bad_symbols = dict(good)
    bad_symbols["symbols"] = []
    with pytest.raises(ValueError, match="non-empty symbols"):
        assert_public_api_stability_integrity(bad_symbols)

    not_list = dict(good)
    not_list["symbols"] = "nope"
    with pytest.raises(ValueError, match="non-empty symbols"):
        assert_public_api_stability_integrity(not_list)

    raw_symbols = good["symbols"]
    assert isinstance(raw_symbols, list)
    symbols = [dict(cast(dict[str, object], row)) for row in raw_symbols]

    non_mapping = dict(good)
    non_mapping["symbols"] = [123]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_public_api_stability_integrity(non_mapping)

    blank_id = dict(good)
    blank_row = dict(symbols[0])
    blank_row["symbol_id"] = ""
    blank_id["symbols"] = [blank_row, *symbols[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_public_api_stability_integrity(blank_id)

    bad_class = dict(good)
    bad_class_row = dict(symbols[0])
    bad_class_row["stability_class"] = "invented"
    bad_class["symbols"] = [bad_class_row, *symbols[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_public_api_stability_integrity(bad_class)

    bad_vis = dict(good)
    bad_vis_row = dict(symbols[0])
    bad_vis_row["visibility"] = "secret"
    bad_vis["symbols"] = [bad_vis_row, *symbols[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_public_api_stability_integrity(bad_vis)

    dep_missing = dict(good)
    dep_row = next(row for row in symbols if row["stability_class"] == "deprecated")
    dep_broken = dict(dep_row)
    dep_broken["replacement_target"] = ""
    # keep same set length/ids but break deprecation fields
    rebuilt = []
    for row in symbols:
        if row["symbol_id"] == dep_broken["symbol_id"]:
            rebuilt.append(dep_broken)
        else:
            rebuilt.append(row)
    dep_missing["symbols"] = rebuilt
    with pytest.raises(ValueError, match="missing replacement_target"):
        assert_public_api_stability_integrity(dep_missing)

    semver_internal = dict(good)
    semver_row = next(row for row in symbols if row["stability_class"] == "semver_stable")
    broken_semver = dict(semver_row)
    broken_semver["visibility"] = "internal"
    rebuilt2 = []
    for row in symbols:
        if row["symbol_id"] == broken_semver["symbol_id"]:
            rebuilt2.append(broken_semver)
        else:
            rebuilt2.append(row)
    semver_internal["symbols"] = rebuilt2
    with pytest.raises(ValueError, match="must be visibility=public"):
        assert_public_api_stability_integrity(semver_internal)

    bad_count = dict(good)
    bad_count["symbol_count"] = 0
    with pytest.raises(ValueError, match="symbol_count"):
        assert_public_api_stability_integrity(bad_count)

    duplicate = dict(good)
    duplicate["symbols"] = [symbols[0], symbols[0]]
    duplicate["symbol_count"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        assert_public_api_stability_integrity(duplicate)

    drifted = dict(good)
    first = dict(symbols[0])
    first["symbol_id"] = "invented.stable.symbol"
    first["stability_class"] = "semver_stable"
    first["visibility"] = "public"
    first["deprecation_state"] = "active"
    first["replacement_target"] = ""
    first["removal_horizon"] = ""
    drifted["symbols"] = [first, *symbols[1:]]
    with pytest.raises(ValueError, match="drift|duplicate|missing"):
        assert_public_api_stability_integrity(drifted)


def test_module_all_exports() -> None:
    """Keep the documented query and policy functions publicly exported."""
    assert "classify_api_path" in public_api_stability.__all__
    assert "probe_deprecation" in public_api_stability.__all__
    assert "validate_breaking_change" in public_api_stability.__all__
    assert "deprecated_public" in public_api_stability.__all__


def test_path_classification_and_probe_invariants() -> None:
    """Reject inconsistent path, probe, and breaking-decision records."""
    with pytest.raises(ValueError, match="path_id"):
        PathClassification(
            path_id="",
            visibility="public",
            stability_class="semver_stable",
            guaranteed_stable=True,
            reason="r",
            in_catalogue=True,
        )
    with pytest.raises(ValueError, match="reason"):
        PathClassification(
            path_id="x",
            visibility="public",
            stability_class="semver_stable",
            guaranteed_stable=True,
            reason="",
            in_catalogue=True,
        )
    with pytest.raises(ValueError, match="guaranteed_stable"):
        PathClassification(
            path_id="x",
            visibility="public",
            stability_class="internal",
            guaranteed_stable=True,
            reason="r",
            in_catalogue=False,
        )
    with pytest.raises(ValueError, match="visibility=public"):
        PathClassification(
            path_id="x",
            visibility="internal",
            stability_class="semver_stable",
            guaranteed_stable=True,
            reason="r",
            in_catalogue=True,
        )
    with pytest.raises(ValueError, match="symbol_id"):
        DeprecationProbe(
            symbol_id="",
            is_deprecated=False,
            replacement_target="",
            removal_horizon="",
            warning_message="",
            reason="r",
        )
    with pytest.raises(ValueError, match="reason"):
        DeprecationProbe(
            symbol_id="x",
            is_deprecated=False,
            replacement_target="",
            removal_horizon="",
            warning_message="",
            reason="",
        )
    with pytest.raises(ValueError, match="replacement_target"):
        DeprecationProbe(
            symbol_id="x",
            is_deprecated=True,
            replacement_target="",
            removal_horizon="v2",
            warning_message="w",
            reason="r",
        )
    with pytest.raises(ValueError, match="removal_horizon"):
        DeprecationProbe(
            symbol_id="x",
            is_deprecated=True,
            replacement_target="y",
            removal_horizon="",
            warning_message="w",
            reason="r",
        )
    with pytest.raises(ValueError, match="warning_message"):
        DeprecationProbe(
            symbol_id="x",
            is_deprecated=True,
            replacement_target="y",
            removal_horizon="v2",
            warning_message="",
            reason="r",
        )
    with pytest.raises(ValueError, match="symbol_id"):
        BreakingChangeDecision(
            symbol_id="",
            change_kind="remove",
            allowed=False,
            reason="r",
            requires_deprecation=True,
        )
    with pytest.raises(ValueError, match="reason"):
        BreakingChangeDecision(
            symbol_id="x",
            change_kind="remove",
            allowed=False,
            reason="",
            requires_deprecation=True,
        )
    with pytest.raises(ValueError, match="change_kind"):
        BreakingChangeDecision(
            symbol_id="x",
            change_kind=cast(BreakingChangeKind, "explode"),
            allowed=False,
            reason="r",
            requires_deprecation=True,
        )


def test_internal_path_heuristics_and_decorator_horizon() -> None:
    """Recognise private path forms and require a nonblank removal horizon."""
    assert classify_api_path("tests/unit/foo.py").visibility == "internal"
    assert classify_api_path("docs/internal/plan.md").visibility == "internal"
    assert classify_api_path("pkg.foo._bar.helper").visibility == "internal"
    assert classify_api_path("_root_private").visibility == "internal"
    with pytest.raises(ValueError, match="removal_horizon"):
        deprecated_public(
            symbol_id="x",
            replacement_target="y",
            removal_horizon="  ",
        )


def test_catalogue_map_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise catalogue construction fail-closed paths via monkeypatch."""
    blank = PublicApiSymbolRecord(
        symbol_id="tmp.blank.placeholder",
        stability_class="semver_stable",
        owner_surface="x",
        deprecation_state="active",
        visibility="public",
        summary="s",
    )
    # force blank after construction by rebuilding with invalid catalogue content
    object.__setattr__(blank, "symbol_id", "  ")
    with pytest.raises(RuntimeError, match="blank symbol_id"):
        monkeypatch.setattr(
            public_api_stability,
            "_CANONICAL_SYMBOLS",
            (blank,),
        )
        public_api_stability._catalogue_map()

    good_a = get_public_api_symbol("scpn_quantum_control.stable_core.Problem")
    good_b = PublicApiSymbolRecord(
        symbol_id=good_a.symbol_id,
        stability_class="semver_stable",
        owner_surface="x",
        deprecation_state="active",
        visibility="public",
        summary="duplicate id",
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        monkeypatch.setattr(
            public_api_stability,
            "_CANONICAL_SYMBOLS",
            (good_a, good_b),
        )
        public_api_stability._catalogue_map()

    with pytest.raises(RuntimeError, match="non-empty"):
        monkeypatch.setattr(public_api_stability, "_CANONICAL_SYMBOLS", ())
        public_api_stability._catalogue_map()
