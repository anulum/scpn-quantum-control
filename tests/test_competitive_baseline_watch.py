# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for competitive baseline watch
"""Real-surface tests for ``scpn_quantum_control.competitive_baseline_watch``."""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import pytest

import scpn_quantum_control.competitive_baseline_watch as competitive_baseline_watch
from scpn_quantum_control.competitive_baseline_watch import (
    COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY,
    COMPETITIVE_BASELINE_WATCH_SCHEMA,
    CompetitiveWatchRecord,
    FeedProbeResult,
    RefreshProbeResult,
    assert_competitive_baseline_watch_integrity,
    build_competitive_baseline_watch_registry,
    get_competitive_watch,
    iter_competitive_watch,
    list_competitor_ids,
    probe_feed,
    probe_refresh,
)
from scpn_quantum_control.differentiable_competitive_baselines import (
    REQUIRED_BASELINE_IDS,
    CompetitiveBaselineRow,
    run_competitive_baseline_refresh,
)

# CompetitiveBaselineRow imported for cast target in synthetic helper tests.


def test_list_covers_required_competitors() -> None:
    """Expose every required competitor once and in canonical order."""
    ids = list_competitor_ids()
    assert len(ids) == len(REQUIRED_BASELINE_IDS)
    assert set(ids) == set(REQUIRED_BASELINE_IDS)
    assert ids == REQUIRED_BASELINE_IDS
    assert ids == list_competitor_ids()


def test_get_known_and_unknown_fail_closed() -> None:
    """Return known rows while rejecting blank and unknown identifiers."""
    row = get_competitive_watch("catalyst")
    assert row.competitor_id == "catalyst"
    assert row.claim_boundary == COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY
    assert row.pin_status == "pinned_snapshot"
    assert row.upstream_version
    assert row.scorecard_feed_status == "blocked"
    assert row.blockers
    with pytest.raises(ValueError, match="non-empty"):
        get_competitive_watch("  ")
    with pytest.raises(ValueError, match="unknown competitor_id"):
        get_competitive_watch("not_a_competitor")


def test_inventory_composes_competitive_baselines() -> None:
    """Compose watch rows from the committed competitive-baseline inventory."""
    refresh = run_competitive_baseline_refresh()
    by_id = {row.baseline_id: row for row in refresh.rows}
    for competitor_id in list_competitor_ids():
        watch = get_competitive_watch(competitor_id)
        base = by_id[watch.competitor_id]
        assert watch.upstream_version == base.upstream_version
        assert watch.source_url == base.source_url
        assert watch.display_name == base.display_name


def test_build_registry_and_integrity() -> None:
    """Build and validate the complete JSON-ready watch registry."""
    registry = build_competitive_baseline_watch_registry()
    assert registry["schema"] == COMPETITIVE_BASELINE_WATCH_SCHEMA
    assert registry["blank_entry_count"] == 0
    count = registry["competitor_count"]
    assert isinstance(count, int)
    assert count == len(REQUIRED_BASELINE_IDS)
    pinned = registry["pinned_snapshot_count"]
    assert isinstance(pinned, int)
    assert pinned >= 1
    validated = assert_competitive_baseline_watch_integrity(registry)
    assert validated["competitor_count"] == count
    assert assert_competitive_baseline_watch_integrity()["blank_entry_count"] == 0
    assert "does not scrape" in str(registry["policy_note"])


def test_probe_refresh_never_invent_green() -> None:
    """Keep refresh probes non-promotional while verification blockers remain."""
    for competitor_id in list_competitor_ids():
        probe = probe_refresh(competitor_id)
        assert probe.allowed_green_current is False
        assert probe.blockers
        assert probe.competitor_id == competitor_id
        assert probe.upstream_version == get_competitive_watch(competitor_id).upstream_version
    with pytest.raises(ValueError, match="unknown"):
        probe_refresh("missing")


def test_probe_feed_bl56_blocked() -> None:
    """Keep scorecard feeds blocked without accepted evidence packages."""
    feed = probe_feed("pennylane", feed_target="scorecard_acceptance")
    assert feed.allowed is False
    assert feed.status == "blocked"
    assert feed.blockers
    assert any("scorecard_category:" in pointer for pointer in feed.pointers)
    assert "refuse invent-green" in feed.reason


def test_probe_feed_bl52_pending_pointers() -> None:
    """Expose bounded route pointers without mutating the route matrix."""
    feed = probe_feed("catalyst", feed_target="governed_route_matrix")
    assert feed.allowed is False
    assert feed.status == "pending"
    assert feed.pointers
    assert any("competitor_id:catalyst" in pointer for pointer in feed.pointers)
    assert feed.blockers
    with pytest.raises(ValueError, match="unknown feed_target"):
        probe_feed("catalyst", feed_target=cast(Any, "unknown_feed"))


def test_iter_filters() -> None:
    """Filter the canonical watch deterministically by pin and refresh state."""
    pinned = iter_competitive_watch(pin_status="pinned_snapshot")
    assert pinned
    assert all(row.pin_status == "pinned_snapshot" for row in pinned)
    pending = iter_competitive_watch(refresh_state="pending_verification")
    assert pending
    assert all(row.refresh_state == "pending_verification" for row in pending)


def test_record_to_dict_and_probes() -> None:
    """Serialise records and probe results into JSON-ready mappings."""
    row = get_competitive_watch("jax")
    payload = row.to_dict()
    assert payload["competitor_id"] == "jax"
    assert payload["pin_status"] == "pinned_snapshot"
    refresh = probe_refresh("jax")
    assert refresh.to_dict()["allowed_green_current"] is False
    feed = probe_feed("jax", feed_target="governed_route_matrix")
    assert feed.to_dict()["allowed"] is False


def test_module_exports() -> None:
    """Keep the documented public watch operations exported."""
    assert "probe_refresh" in competitive_baseline_watch.__all__
    assert "probe_feed" in competitive_baseline_watch.__all__
    assert "build_competitive_baseline_watch_registry" in competitive_baseline_watch.__all__


def test_watch_record_validation() -> None:
    """Reject malformed watch records and invent-green combinations."""
    base_kwargs: dict[str, Any] = {
        "competitor_id": "jax",
        "display_name": "JAX",
        "pin_status": "pinned_snapshot",
        "upstream_version": "0.1",
        "source_url": "https://example.com",
        "source_kind": "official_docs",
        "checked_on": "2026-06-27",
        "refresh_due_on": "2026-08-11",
        "refresh_state": "pending_verification",
        "scorecard_categories": ("jax_native_transforms",),
        "route_matrix_feed_status": "pending",
        "scorecard_feed_status": "blocked",
        "blockers": ("re-pin pending",),
    }
    ok = CompetitiveWatchRecord(**base_kwargs)
    assert ok.competitor_id == "jax"

    with pytest.raises(ValueError, match="unknown competitor"):
        CompetitiveWatchRecord(**{**base_kwargs, "competitor_id": cast(Any, "nope")})
    with pytest.raises(ValueError, match="pin_status"):
        CompetitiveWatchRecord(**{**base_kwargs, "pin_status": cast(Any, "weird")})
    with pytest.raises(ValueError, match="refresh_state"):
        CompetitiveWatchRecord(**{**base_kwargs, "refresh_state": cast(Any, "weird")})
    with pytest.raises(ValueError, match="display_name"):
        CompetitiveWatchRecord(**{**base_kwargs, "display_name": ""})
    with pytest.raises(ValueError, match="route_matrix_feed_status"):
        CompetitiveWatchRecord(**{**base_kwargs, "route_matrix_feed_status": cast(Any, "nope")})
    with pytest.raises(ValueError, match="scorecard_feed_status"):
        CompetitiveWatchRecord(**{**base_kwargs, "scorecard_feed_status": cast(Any, "nope")})
    with pytest.raises(ValueError, match="source_url"):
        CompetitiveWatchRecord(**{**base_kwargs, "source_url": ""})
    with pytest.raises(ValueError, match="source_kind"):
        CompetitiveWatchRecord(**{**base_kwargs, "source_kind": ""})
    with pytest.raises(ValueError, match="checked_on"):
        CompetitiveWatchRecord(**{**base_kwargs, "checked_on": ""})
    with pytest.raises(ValueError, match="refresh_due_on"):
        CompetitiveWatchRecord(**{**base_kwargs, "refresh_due_on": ""})
    with pytest.raises(ValueError, match="as_of"):
        CompetitiveWatchRecord(**{**base_kwargs, "as_of": ""})
    with pytest.raises(ValueError, match="scorecard_categories"):
        CompetitiveWatchRecord(**{**base_kwargs, "scorecard_categories": ("",)})
    with pytest.raises(ValueError, match="blockers must be non-empty strings"):
        CompetitiveWatchRecord(**{**base_kwargs, "blockers": ("ok", "  ")})
    with pytest.raises(ValueError, match="pinned_snapshot requires"):
        CompetitiveWatchRecord(**{**base_kwargs, "upstream_version": ""})
    with pytest.raises(ValueError, match="unpinned rows must not"):
        CompetitiveWatchRecord(
            **{
                **base_kwargs,
                "pin_status": "unpinned",
                "upstream_version": "1.0",
                "refresh_state": "blocked",
                "blockers": ("unpinned",),
            }
        )
    with pytest.raises(ValueError, match="cannot be refresh_state=fresh"):
        CompetitiveWatchRecord(
            **{
                **base_kwargs,
                "pin_status": "unpinned",
                "upstream_version": "",
                "refresh_state": "fresh",
                "blockers": (),
            }
        )
    with pytest.raises(ValueError, match="scorecard_feed_status must be blocked"):
        CompetitiveWatchRecord(**{**base_kwargs, "scorecard_feed_status": "ready_pointer"})
    with pytest.raises(ValueError, match="requires at least one blocker"):
        CompetitiveWatchRecord(**{**base_kwargs, "blockers": ()})


def test_refresh_and_feed_probe_invariants() -> None:
    """Reject invalid refresh and feed probe result combinations."""
    with pytest.raises(ValueError, match="competitor_id"):
        RefreshProbeResult(
            competitor_id="",
            refresh_state="due",
            pin_status="unpinned",
            upstream_version="",
            allowed_green_current=False,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="reason"):
        RefreshProbeResult(
            competitor_id="jax",
            refresh_state="due",
            pin_status="unpinned",
            upstream_version="",
            allowed_green_current=False,
            reason="",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="blockers"):
        RefreshProbeResult(
            competitor_id="jax",
            refresh_state="due",
            pin_status="pinned_snapshot",
            upstream_version="1",
            allowed_green_current=False,
            reason="r",
            blockers=(),
        )
    with pytest.raises(ValueError, match="cannot list blockers"):
        RefreshProbeResult(
            competitor_id="jax",
            refresh_state="fresh",
            pin_status="pinned_snapshot",
            upstream_version="1",
            allowed_green_current=True,
            reason="r",
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="blockers entries must be non-empty"):
        RefreshProbeResult(
            competitor_id="jax",
            refresh_state="due",
            pin_status="pinned_snapshot",
            upstream_version="1",
            allowed_green_current=False,
            reason="r",
            blockers=("ok", " "),
        )
    with pytest.raises(ValueError, match="competitor_id"):
        FeedProbeResult(
            competitor_id="",
            feed_target="governed_route_matrix",
            status="blocked",
            allowed=False,
            reason="r",
            pointers=(),
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="feed_target"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target=cast(Any, "x"),
            status="blocked",
            allowed=False,
            reason="r",
            pointers=(),
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="status"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="governed_route_matrix",
            status=cast(Any, "nope"),
            allowed=False,
            reason="r",
            pointers=(),
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="reason"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="governed_route_matrix",
            status="blocked",
            allowed=False,
            reason="",
            pointers=(),
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="allowed feed cannot list blockers"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="governed_route_matrix",
            status="ready_pointer",
            allowed=True,
            reason="r",
            pointers=("p",),
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="blocked feed requires blockers"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="governed_route_matrix",
            status="blocked",
            allowed=False,
            reason="r",
            pointers=(),
            blockers=(),
        )
    with pytest.raises(ValueError, match="blockers entries must be non-empty"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="governed_route_matrix",
            status="blocked",
            allowed=False,
            reason="r",
            pointers=(),
            blockers=(" ",),
        )
    with pytest.raises(ValueError, match="pointers entries must be non-empty"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="governed_route_matrix",
            status="blocked",
            allowed=False,
            reason="r",
            pointers=(" ",),
            blockers=("b",),
        )
    with pytest.raises(ValueError, match="must not allow invent-green"):
        FeedProbeResult(
            competitor_id="jax",
            feed_target="scorecard_acceptance",
            status="ready_pointer",
            allowed=True,
            reason="r",
            pointers=("p",),
            blockers=(),
        )


def test_integrity_rejects_blank_and_invent_green() -> None:
    """Reject incomplete, duplicate, or promotional registry payloads."""
    good = build_competitive_baseline_watch_registry()
    assert_competitive_baseline_watch_integrity(good)

    bad_blank = dict(good)
    bad_blank["blank_entry_count"] = 1
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_competitive_baseline_watch_integrity(bad_blank)

    empty = dict(good)
    empty["competitors"] = []
    with pytest.raises(ValueError, match="non-empty competitors"):
        assert_competitive_baseline_watch_integrity(empty)

    not_map = dict(good)
    not_map["competitors"] = [123]
    with pytest.raises(ValueError, match="must be a mapping"):
        assert_competitive_baseline_watch_integrity(not_map)

    raw = good["competitors"]
    assert isinstance(raw, list)
    symbols = [dict(cast(dict[str, object], row)) for row in raw]

    blank_id = dict(good)
    blank_row = dict(symbols[0])
    blank_row["competitor_id"] = ""
    blank_id["competitors"] = [blank_row, *symbols[1:]]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_competitive_baseline_watch_integrity(blank_id)

    invent = dict(good)
    invent_row = dict(symbols[0])
    invent_row["scorecard_feed_status"] = "ready_pointer"
    invent["competitors"] = [
        invent_row if row["competitor_id"] == invent_row["competitor_id"] else row
        for row in symbols
    ]
    with pytest.raises(ValueError, match="invent-green scorecard"):
        assert_competitive_baseline_watch_integrity(invent)

    unpinned_fresh = dict(good)
    bad_pin = dict(symbols[0])
    bad_pin["pin_status"] = "unpinned"
    bad_pin["upstream_version"] = ""
    bad_pin["refresh_state"] = "fresh"
    bad_pin["blockers"] = []
    unpinned_fresh["competitors"] = [
        bad_pin if row["competitor_id"] == bad_pin["competitor_id"] else row for row in symbols
    ]
    with pytest.raises(ValueError, match="invent-green|unpinned yet fresh"):
        assert_competitive_baseline_watch_integrity(unpinned_fresh)

    pinned_no_ver = dict(good)
    no_ver = dict(symbols[0])
    no_ver["pin_status"] = "pinned_snapshot"
    no_ver["upstream_version"] = ""
    pinned_no_ver["competitors"] = [
        no_ver if row["competitor_id"] == no_ver["competitor_id"] else row for row in symbols
    ]
    with pytest.raises(ValueError, match="without version"):
        assert_competitive_baseline_watch_integrity(pinned_no_ver)

    unpinned_with_ver = dict(good)
    with_ver = dict(symbols[0])
    with_ver["pin_status"] = "unpinned"
    with_ver["upstream_version"] = "1.2.3"
    with_ver["refresh_state"] = "blocked"
    with_ver["blockers"] = ["x"]
    unpinned_with_ver["competitors"] = [
        with_ver if row["competitor_id"] == with_ver["competitor_id"] else row for row in symbols
    ]
    with pytest.raises(ValueError, match="unpinned with non-empty"):
        assert_competitive_baseline_watch_integrity(unpinned_with_ver)

    no_blockers = dict(good)
    due_row = dict(symbols[0])
    due_row["refresh_state"] = "due"
    due_row["blockers"] = []
    no_blockers["competitors"] = [
        due_row if row["competitor_id"] == due_row["competitor_id"] else row for row in symbols
    ]
    with pytest.raises(ValueError, match="without blockers"):
        assert_competitive_baseline_watch_integrity(no_blockers)

    bad_count = dict(good)
    bad_count["competitor_count"] = 0
    with pytest.raises(ValueError, match="competitor_count"):
        assert_competitive_baseline_watch_integrity(bad_count)

    unknown = dict(good)
    unk = dict(symbols[0])
    unk["competitor_id"] = "totally_unknown"
    unknown["competitors"] = [unk, *symbols[1:]]
    with pytest.raises(ValueError, match="unknown competitor_id"):
        assert_competitive_baseline_watch_integrity(unknown)

    duplicate = dict(good)
    duplicate["competitors"] = [symbols[0], symbols[0]]
    duplicate["competitor_count"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        assert_competitive_baseline_watch_integrity(duplicate)

    missing = dict(good)
    missing["competitors"] = symbols[1:]
    missing["competitor_count"] = len(symbols) - 1
    with pytest.raises(ValueError, match="missing competitors"):
        assert_competitive_baseline_watch_integrity(missing)


def test_classify_unpinned_and_due_via_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise unpinned/due classification paths with synthetic baseline rows."""
    real = run_competitive_baseline_refresh().rows[0]
    from scpn_quantum_control import competitive_baseline_watch as mod

    assert mod._pin_status("") == "unpinned"
    assert mod._pin_status("  ") == "unpinned"
    assert mod._pin_status("1.0") == "pinned_snapshot"

    # due path: as_of far in the future past refresh window
    far = date(2030, 1, 1)
    state = mod._classify_refresh_state(real, as_of=far)
    assert state == "due"
    watch_due = mod._from_baseline_row(real, as_of=far)
    assert watch_due.refresh_state == "due"
    assert watch_due.blockers

    # blank version classification without constructing invalid CompetitiveBaselineRow
    class _FakeRow:
        baseline_id = real.baseline_id
        display_name = real.display_name
        upstream_version = ""
        source_url = real.source_url
        source_kind = real.source_kind
        checked_on = real.checked_on
        refresh_due_on = real.refresh_due_on
        scorecard_categories = real.scorecard_categories

        def is_fresh(self, *, as_of: date) -> bool:
            return False

    fake = cast(CompetitiveBaselineRow, _FakeRow())
    assert mod._classify_refresh_state(fake, as_of=date(2026, 7, 23)) == "blocked"
    blocked = mod._from_baseline_row(fake, as_of=date(2026, 7, 23))
    assert blocked.pin_status == "unpinned"
    assert blocked.refresh_state == "blocked"
    assert (
        "blank" in blocked.blockers[0].lower() or "unpinned" in " ".join(blocked.blockers).lower()
    )

    # catalogue coverage failure
    with pytest.raises(RuntimeError, match="must cover"):
        monkeypatch.setattr(
            mod,
            "run_competitive_baseline_refresh",
            lambda: type("R", (), {"rows": (real,)})(),
        )
        # only one row but REQUIRED has many — rebuild
        mod._build_canonical_watch()


def test_integrity_bad_pin_and_refresh_class() -> None:
    """Reject unsupported pin and refresh vocabularies in registry rows."""
    good = build_competitive_baseline_watch_registry()
    raw = good["competitors"]
    assert isinstance(raw, list)
    symbols = [dict(cast(dict[str, object], row)) for row in raw]
    bad_pin = dict(good)
    row = dict(symbols[0])
    row["pin_status"] = "nope"
    bad_pin["competitors"] = [
        row if r["competitor_id"] == row["competitor_id"] else r for r in symbols
    ]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_competitive_baseline_watch_integrity(bad_pin)

    bad_refresh = dict(good)
    row2 = dict(symbols[0])
    row2["refresh_state"] = "nope"
    bad_refresh["competitors"] = [
        row2 if r["competitor_id"] == row2["competitor_id"] else r for r in symbols
    ]
    with pytest.raises(ValueError, match="blank or invalid"):
        assert_competitive_baseline_watch_integrity(bad_refresh)


def test_from_baseline_row_fresh_refresh_fallthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned row with non-due/non-pending refresh state falls through without extra blockers."""
    from scpn_quantum_control.differentiable_competitive_baselines import (
        run_competitive_baseline_refresh,
    )

    refresh = run_competitive_baseline_refresh()
    row = next(r for r in refresh.rows if r.upstream_version and r.upstream_version.strip())
    monkeypatch.setattr(
        competitive_baseline_watch,
        "_classify_refresh_state",
        lambda _row, *, as_of: "fresh",
    )
    record = competitive_baseline_watch._from_baseline_row(row, as_of=row.checked_on)
    assert record.pin_status == "pinned_snapshot"
    assert record.refresh_state == "fresh"
    # Default watch blockers remain; no due/pending-specific insert.
    assert all("past freshness window" not in b for b in record.blockers)
    assert all("continuous re-pin pending" not in b for b in record.blockers)
