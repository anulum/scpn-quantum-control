# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — continuous competitive baseline watch
"""Fail-closed continuous competitive-baseline watch surface.

Productises a versioned catalogue of competitor baseline rows with pin /
version / source / refresh fields, plus structured feed probes toward
route-matrix (route matrix) and scorecard (scorecard engine).

Composes :mod:`differentiable_competitive_baselines` snapshot inventory.
Does **not** scrape vendors, invent Verified-At-Source pins, or promote
scorecard categories to invent-green ``at_baseline`` / ``exceeds_baseline``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

from .differentiable_competitive_baselines import (
    REQUIRED_BASELINE_IDS,
    CompetitiveBaselineId,
    CompetitiveBaselineRow,
    run_competitive_baseline_refresh,
)

PinStatus = Literal["pinned_snapshot", "unpinned", "blocked"]
"""Pin vocabulary: snapshot pins are declared evidence, not live-verified green."""

RefreshState = Literal["fresh", "due", "blocked", "pending_verification"]
"""Refresh state for continuous watch (never invent live scrape wins)."""

FeedTarget = Literal["governed_route_matrix", "scorecard_acceptance"]
"""Downstream feed targets for structured watch outputs."""

FeedStatus = Literal["ready_pointer", "blocked", "pending"]
"""Feed readiness toward route and scorecard governance targets."""

COMPETITIVE_BASELINE_WATCH_SCHEMA: Final[str] = "competitive_baseline_watch.v1"
"""JSON schema identifier for serialised watch payloads."""

COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY: Final[str] = (
    "competitive baseline watch only; pinned_snapshot records declared "
    "comparison coverage from the committed refresh inventory; refresh/feed "
    "probes never invent live scrape wins or promote scorecard categories to "
    "at_baseline/exceeds_baseline without accepted scorecard evidence packages"
)
"""Shared claim boundary for watch rows and probe results."""

# Deterministic inventory clock for watch classification (not a live network clock).
_WATCH_AS_OF: Final[date] = date(2026, 8, 25)

# Explicit blockers when continuous re-verification / CI harness is not landed.
_DEFAULT_WATCH_BLOCKERS: Final[tuple[str, ...]] = (
    "continuous re-verification schedule not automated in CI",
    "Verified-At-Source re-pin not executed this watch cycle",
)

_ROUTE_MATRIX_FEED_BLOCKERS: Final[tuple[str, ...]] = (
    "route-matrix feed is pointer-only until a boundary update package is accepted",
    "no invent-green competitor parity from watch alone",
)

_SCORECARD_FEED_BLOCKERS: Final[tuple[str, ...]] = (
    "scorecard category remains behind_baseline until claim-ledger and external evidence",
    "watch feed must not invent at_baseline/exceeds_baseline promotion",
)


@dataclass(frozen=True, slots=True)
class CompetitiveWatchRecord:
    """One competitor under continuous baseline watch.

    Attributes
    ----------
    competitor_id
        Canonical baseline identity (same set as competitive baselines).
    display_name
        Human-readable competitor label.
    pin_status
        Whether a non-empty snapshot version is declared (not invent-green).
    upstream_version
        Declared version label from the refresh inventory (may be descriptive).
    source_url
        Official source URL for the baseline.
    source_kind
        Official docs / repository kind.
    checked_on
        Snapshot check date (ISO string).
    refresh_due_on
        Refresh due date (ISO string).
    refresh_state
        Continuous-watch refresh classification at the inventory as-of date.
    scorecard_categories
        Scorecard category identifiers this competitor feeds.
    route_matrix_feed_status
        Structured feed readiness toward the route matrix.
    scorecard_feed_status
        Structured feed readiness toward the scorecard engine.
    blockers
        Non-empty when watch cannot claim green current / ready feeds.
    as_of
        Inventory date label used for refresh classification.
    claim_boundary
        Non-promotional claim boundary.

    """

    competitor_id: CompetitiveBaselineId
    display_name: str
    pin_status: PinStatus
    upstream_version: str
    source_url: str
    source_kind: str
    checked_on: str
    refresh_due_on: str
    refresh_state: RefreshState
    scorecard_categories: tuple[str, ...]
    route_matrix_feed_status: FeedStatus
    scorecard_feed_status: FeedStatus
    blockers: tuple[str, ...]
    as_of: str = "2026-08-25"
    claim_boundary: str = COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate watch-record invariants."""
        if self.competitor_id not in REQUIRED_BASELINE_IDS:
            raise ValueError(f"unknown competitor_id: {self.competitor_id!r}")
        if self.pin_status not in {"pinned_snapshot", "unpinned", "blocked"}:
            raise ValueError(f"unknown pin_status: {self.pin_status!r}")
        if self.refresh_state not in {"fresh", "due", "blocked", "pending_verification"}:
            raise ValueError(f"unknown refresh_state: {self.refresh_state!r}")
        if self.route_matrix_feed_status not in {"ready_pointer", "blocked", "pending"}:
            raise ValueError(
                f"unknown route_matrix_feed_status: {self.route_matrix_feed_status!r}"
            )
        if self.scorecard_feed_status not in {"ready_pointer", "blocked", "pending"}:
            raise ValueError(f"unknown scorecard_feed_status: {self.scorecard_feed_status!r}")
        if not self.display_name or not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if not self.source_url or not self.source_url.strip():
            raise ValueError("source_url must be non-empty")
        if not self.source_kind or not self.source_kind.strip():
            raise ValueError("source_kind must be non-empty")
        if not self.checked_on or not self.checked_on.strip():
            raise ValueError("checked_on must be non-empty")
        if not self.refresh_due_on or not self.refresh_due_on.strip():
            raise ValueError("refresh_due_on must be non-empty")
        if not self.as_of or not self.as_of.strip():
            raise ValueError("as_of must be non-empty")
        if any(not item or not str(item).strip() for item in self.scorecard_categories):
            raise ValueError("scorecard_categories entries must be non-empty")
        if any(not item or not str(item).strip() for item in self.blockers):
            raise ValueError("blockers must be non-empty strings when present")
        if self.pin_status == "unpinned" and self.upstream_version.strip():
            raise ValueError("unpinned rows must not carry a non-empty upstream_version")
        if self.pin_status == "pinned_snapshot" and not self.upstream_version.strip():
            raise ValueError("pinned_snapshot requires non-empty upstream_version")
        if self.pin_status == "unpinned" and self.refresh_state == "fresh":
            raise ValueError("unpinned rows cannot be refresh_state=fresh (refuse invent-green)")
        if self.refresh_state in {"due", "blocked", "pending_verification"} and not self.blockers:
            raise ValueError("non-green refresh_state requires at least one blocker")
        if self.scorecard_feed_status != "blocked":
            # Product honesty: never claim a ready feed without scorecard evidence.
            raise ValueError("scorecard_feed_status must be blocked until evidence packages exist")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this watch record."""
        return {
            "competitor_id": self.competitor_id,
            "display_name": self.display_name,
            "pin_status": self.pin_status,
            "upstream_version": self.upstream_version,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            "checked_on": self.checked_on,
            "refresh_due_on": self.refresh_due_on,
            "refresh_state": self.refresh_state,
            "scorecard_categories": list(self.scorecard_categories),
            "route_matrix_feed_status": self.route_matrix_feed_status,
            "scorecard_feed_status": self.scorecard_feed_status,
            "blockers": list(self.blockers),
            "as_of": self.as_of,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class RefreshProbeResult:
    """Structured result of a continuous-watch refresh probe.

    Attributes
    ----------
    competitor_id
        Queried competitor.
    refresh_state
        Classified refresh state.
    pin_status
        Pin status at probe time.
    upstream_version
        Declared version (empty when unpinned).
    allowed_green_current
        Always false unless all green criteria met (product refuses invent-green).
    reason
        Human-readable decision reason.
    blockers
        Open blockers when not green.

    """

    competitor_id: str
    refresh_state: RefreshState
    pin_status: PinStatus
    upstream_version: str
    allowed_green_current: bool
    reason: str
    blockers: tuple[str, ...]
    claim_boundary: str = COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate refresh probe invariants."""
        if not self.competitor_id or not self.competitor_id.strip():
            raise ValueError("competitor_id must be non-empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed_green_current and self.blockers:
            raise ValueError("allowed_green_current cannot list blockers")
        if not self.allowed_green_current and not self.blockers:
            raise ValueError("non-green refresh probe requires blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe."""
        return {
            "competitor_id": self.competitor_id,
            "refresh_state": self.refresh_state,
            "pin_status": self.pin_status,
            "upstream_version": self.upstream_version,
            "allowed_green_current": self.allowed_green_current,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class FeedProbeResult:
    """Structured feed probe toward route-matrix or scorecard governance.

    Attributes
    ----------
    competitor_id
        Source competitor row.
    feed_target
        Downstream target engine.
    status
        Feed readiness.
    allowed
        Whether an invent-green promotion/update is allowed (false when blocked).
    reason
        Human-readable decision reason.
    pointers
        Structured evidence / route / category pointers (may be empty when blocked).
    blockers
        Open blockers when not allowed.

    """

    competitor_id: str
    feed_target: FeedTarget
    status: FeedStatus
    allowed: bool
    reason: str
    pointers: tuple[str, ...]
    blockers: tuple[str, ...]
    claim_boundary: str = COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate feed probe invariants."""
        if not self.competitor_id or not self.competitor_id.strip():
            raise ValueError("competitor_id must be non-empty")
        if self.feed_target not in {"governed_route_matrix", "scorecard_acceptance"}:
            raise ValueError(f"unknown feed_target: {self.feed_target!r}")
        if self.status not in {"ready_pointer", "blocked", "pending"}:
            raise ValueError(f"unknown status: {self.status!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.blockers:
            raise ValueError("allowed feed cannot list blockers")
        if not self.allowed and not self.blockers:
            raise ValueError("blocked feed requires blockers")
        if any(not item or not item.strip() for item in self.blockers):
            raise ValueError("blockers entries must be non-empty")
        if any(not item or not item.strip() for item in self.pointers):
            raise ValueError("pointers entries must be non-empty when present")
        if self.feed_target == "scorecard_acceptance" and self.allowed:
            raise ValueError(
                "scorecard feed must not allow invent-green promotion from watch alone"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this feed probe."""
        return {
            "competitor_id": self.competitor_id,
            "feed_target": self.feed_target,
            "status": self.status,
            "allowed": self.allowed,
            "reason": self.reason,
            "pointers": list(self.pointers),
            "blockers": list(self.blockers),
            "claim_boundary": self.claim_boundary,
        }


def _classify_refresh_state(row: CompetitiveBaselineRow, *, as_of: date) -> RefreshState:
    """Classify refresh state from inventory freshness without network I/O."""
    if not row.upstream_version or not row.upstream_version.strip():
        return "blocked"
    if row.is_fresh(as_of=as_of):
        # Snapshot is within age window, but continuous watch still needs re-verification ops.
        return "pending_verification"
    return "due"


def _pin_status(version: str) -> PinStatus:
    """Map version string to pin status (blank → unpinned, refuse invent-green)."""
    if not version or not version.strip():
        return "unpinned"
    return "pinned_snapshot"


def _from_baseline_row(row: CompetitiveBaselineRow, *, as_of: date) -> CompetitiveWatchRecord:
    """Build one watch record from a competitive-baseline inventory row."""
    pin = _pin_status(row.upstream_version)
    refresh_state = _classify_refresh_state(row, as_of=as_of)
    blockers: list[str] = list(_DEFAULT_WATCH_BLOCKERS)
    if pin == "unpinned":
        blockers.insert(0, "upstream_version blank; refuse invent-green current baseline")
        refresh_state = "blocked"
    elif refresh_state == "due":
        blockers.insert(0, "snapshot past freshness window; refresh due")
    elif refresh_state == "pending_verification":
        blockers.insert(0, "snapshot within age window but continuous re-pin pending")

    return CompetitiveWatchRecord(
        competitor_id=row.baseline_id,
        display_name=row.display_name,
        pin_status=pin,
        upstream_version=row.upstream_version.strip(),
        source_url=row.source_url,
        source_kind=row.source_kind,
        checked_on=row.checked_on.isoformat(),
        refresh_due_on=row.refresh_due_on.isoformat(),
        refresh_state=refresh_state,
        scorecard_categories=tuple(str(item) for item in row.scorecard_categories),
        route_matrix_feed_status="pending",
        scorecard_feed_status="blocked",
        blockers=tuple(blockers),
        as_of=as_of.isoformat(),
    )


def _build_canonical_watch(
    *,
    as_of: date = _WATCH_AS_OF,
) -> tuple[CompetitiveWatchRecord, ...]:
    """Compose watch catalogue from the committed competitive-baseline refresh."""
    refresh = run_competitive_baseline_refresh()
    records = tuple(_from_baseline_row(row, as_of=as_of) for row in refresh.rows)
    seen = tuple(record.competitor_id for record in records)
    if tuple(sorted(seen)) != tuple(sorted(REQUIRED_BASELINE_IDS)):
        raise RuntimeError(
            f"competitive baseline watch must cover all required competitors (got={seen!r})"
        )
    # Preserve REQUIRED_BASELINE_IDS order for deterministic inventory.
    by_id = {record.competitor_id: record for record in records}
    return tuple(by_id[identifier] for identifier in REQUIRED_BASELINE_IDS)


_CANONICAL_WATCH: Final[tuple[CompetitiveWatchRecord, ...]] = _build_canonical_watch()
_WATCH_BY_ID: Final[Mapping[str, CompetitiveWatchRecord]] = {
    record.competitor_id: record for record in _CANONICAL_WATCH
}


def list_competitor_ids() -> tuple[str, ...]:
    """Return all watched competitor identifiers in canonical order.

    Returns
    -------
    tuple[str, ...]
        Ordered competitor identifiers.

    """
    return tuple(record.competitor_id for record in _CANONICAL_WATCH)


def get_competitive_watch(competitor_id: str) -> CompetitiveWatchRecord:
    """Return one watch row or raise for blank/unknown identifiers.

    Parameters
    ----------
    competitor_id
        Competitor / baseline key.

    Returns
    -------
    CompetitiveWatchRecord
        Matching catalogue row.

    Raises
    ------
    ValueError
        If ``competitor_id`` is blank or unknown (fail closed).

    """
    if not competitor_id or not str(competitor_id).strip():
        raise ValueError("competitor_id must be a non-empty string")
    key = str(competitor_id).strip()
    try:
        return _WATCH_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown competitor_id {key!r}; refuse invent-green baseline "
            f"(known_count={len(_WATCH_BY_ID)})"
        ) from exc


def iter_competitive_watch(
    *,
    pin_status: PinStatus | None = None,
    refresh_state: RefreshState | None = None,
) -> tuple[CompetitiveWatchRecord, ...]:
    """Return filtered watch rows in stable order.

    Parameters
    ----------
    pin_status
        Optional pin filter.
    refresh_state
        Optional refresh-state filter.

    Returns
    -------
    tuple[CompetitiveWatchRecord, ...]
        Matching rows.

    """
    rows: Iterable[CompetitiveWatchRecord] = _CANONICAL_WATCH
    if pin_status is not None:
        rows = (row for row in rows if row.pin_status == pin_status)
    if refresh_state is not None:
        rows = (row for row in rows if row.refresh_state == refresh_state)
    return tuple(rows)


def probe_refresh(competitor_id: str) -> RefreshProbeResult:
    """Probe continuous-watch refresh state without inventing live scrape wins.

    Parameters
    ----------
    competitor_id
        Competitor key.

    Returns
    -------
    RefreshProbeResult
        Structured status; ``allowed_green_current`` is false when blockers remain.

    Raises
    ------
    ValueError
        If ``competitor_id`` is blank or unknown.

    """
    record = get_competitive_watch(competitor_id)
    # Product policy: continuous watch never claims green-current without CI re-pin.
    allowed = False
    reason = (
        f"refresh_state={record.refresh_state}; pin_status={record.pin_status}; "
        "refuse invent-green current baseline without continuous re-verification"
    )
    return RefreshProbeResult(
        competitor_id=record.competitor_id,
        refresh_state=record.refresh_state,
        pin_status=record.pin_status,
        upstream_version=record.upstream_version,
        allowed_green_current=allowed,
        reason=reason,
        blockers=record.blockers,
    )


def probe_feed(
    competitor_id: str,
    *,
    feed_target: FeedTarget,
) -> FeedProbeResult:
    """Probe structured feed readiness toward route or scorecard governance.

    Feed is always fail-closed for invent-green scorecard promotion. The route
    matrix may return category/route pointers while still blocking automatic
    mutation.

    Parameters
    ----------
    competitor_id
        Competitor key.
    feed_target
        ``governed_route_matrix`` or ``scorecard_acceptance``.

    Returns
    -------
    FeedProbeResult
        Structured feed decision.

    Raises
    ------
    ValueError
        If identifiers / targets are invalid.

    """
    if feed_target not in {"governed_route_matrix", "scorecard_acceptance"}:
        raise ValueError(f"unknown feed_target: {feed_target!r}")
    record = get_competitive_watch(competitor_id)

    if feed_target == "scorecard_acceptance":
        pointers = tuple(
            f"scorecard_category:{category}" for category in record.scorecard_categories
        )
        return FeedProbeResult(
            competitor_id=record.competitor_id,
            feed_target=feed_target,
            status="blocked",
            allowed=False,
            reason=(
                "refuse invent-green scorecard promotion from watch alone; "
                "require claim_ledger_promoted_row + external_baseline_comparison"
            ),
            pointers=pointers,
            blockers=_SCORECARD_FEED_BLOCKERS,
        )

    # governed_route_matrix
    pointers = (
        f"competitor_id:{record.competitor_id}",
        f"source_url:{record.source_url}",
        f"pin_status:{record.pin_status}",
        "related_surface:governed_route_matrix.competitor_boundary",
    )
    return FeedProbeResult(
        competitor_id=record.competitor_id,
        feed_target=feed_target,
        status="pending",
        allowed=False,
        reason=(
            "route-matrix feed emits structured pointers only; automatic matrix "
            "mutation requires an accepted boundary-update package"
        ),
        pointers=pointers,
        blockers=_ROUTE_MATRIX_FEED_BLOCKERS,
    )


def build_competitive_baseline_watch_registry() -> dict[str, object]:
    """Build the full serialisable competitive baseline watch registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every required competitor (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_WATCH]
    pinned = sum(1 for row in _CANONICAL_WATCH if row.pin_status == "pinned_snapshot")
    unpinned = sum(1 for row in _CANONICAL_WATCH if row.pin_status == "unpinned")
    pending = sum(1 for row in _CANONICAL_WATCH if row.refresh_state == "pending_verification")
    due = sum(1 for row in _CANONICAL_WATCH if row.refresh_state == "due")
    blocked = sum(1 for row in _CANONICAL_WATCH if row.refresh_state == "blocked")
    return {
        "schema": COMPETITIVE_BASELINE_WATCH_SCHEMA,
        "claim_boundary": COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY,
        "competitor_count": len(rows),
        "pinned_snapshot_count": pinned,
        "unpinned_count": unpinned,
        "pending_verification_count": pending,
        "due_count": due,
        "blocked_refresh_count": blocked,
        "blank_entry_count": 0,
        "as_of": _WATCH_AS_OF.isoformat(),
        "competitors": rows,
        "policy_note": (
            "Composes differentiable_competitive_baselines snapshot inventory; "
            "does not scrape vendors or invent Verified-At-Source pins; "
            "scorecard feeds stay blocked without accepted evidence packages."
        ),
    }


def assert_competitive_baseline_watch_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry covers all required competitors without blanks.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_competitive_baseline_watch_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If coverage, blanks, or invent-green feed claims appear.

    """
    registry = (
        dict(payload) if payload is not None else build_competitive_baseline_watch_registry()
    )
    competitors = registry.get("competitors")
    if not isinstance(competitors, list) or not competitors:
        raise ValueError(
            "competitive baseline watch registry must contain a non-empty competitors list"
        )
    seen: set[str] = set()
    blank = 0
    for index, row in enumerate(competitors):
        if not isinstance(row, Mapping):
            raise ValueError(f"competitor row {index} must be a mapping")
        competitor_id = row.get("competitor_id")
        pin_status = row.get("pin_status")
        refresh_state = row.get("refresh_state")
        scorecard_feed = row.get("scorecard_feed_status")
        if not competitor_id or not str(competitor_id).strip():
            blank += 1
            continue
        cid = str(competitor_id).strip()
        if cid not in REQUIRED_BASELINE_IDS:
            raise ValueError(f"unknown competitor_id in registry: {cid!r}")
        if cid in seen:
            raise ValueError(f"duplicate competitor_id in registry: {cid!r}")
        seen.add(cid)
        if pin_status not in {"pinned_snapshot", "unpinned", "blocked"}:
            blank += 1
            continue
        if refresh_state not in {"fresh", "due", "blocked", "pending_verification"}:
            blank += 1
            continue
        if pin_status == "unpinned" and refresh_state == "fresh":
            raise ValueError(f"competitor {cid!r} invent-green: unpinned yet fresh")
        version = row.get("upstream_version", "")
        if pin_status == "pinned_snapshot" and (not version or not str(version).strip()):
            raise ValueError(f"competitor {cid!r} pinned_snapshot without version")
        if pin_status == "unpinned" and version and str(version).strip():
            raise ValueError(f"competitor {cid!r} unpinned with non-empty version")
        if scorecard_feed != "blocked":
            raise ValueError(
                f"competitor {cid!r} invent-green scorecard feed status {scorecard_feed!r}"
            )
        blockers = row.get("blockers")
        if refresh_state != "fresh" and (not isinstance(blockers, list) or not blockers):
            raise ValueError(f"competitor {cid!r} non-green refresh without blockers")
    if blank:
        raise ValueError(
            f"competitive baseline watch registry has {blank} blank or invalid entries"
        )
    missing = set(REQUIRED_BASELINE_IDS) - seen
    if missing:
        raise ValueError(f"competitive baseline watch registry missing competitors: {missing}")
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    competitor_count = registry.get("competitor_count", -1)
    if not isinstance(competitor_count, int) or competitor_count != len(competitors):
        raise ValueError("competitor_count does not match competitors list length")
    return registry


__all__ = [
    "COMPETITIVE_BASELINE_WATCH_CLAIM_BOUNDARY",
    "COMPETITIVE_BASELINE_WATCH_SCHEMA",
    "CompetitiveWatchRecord",
    "FeedProbeResult",
    "FeedStatus",
    "FeedTarget",
    "PinStatus",
    "RefreshProbeResult",
    "RefreshState",
    "assert_competitive_baseline_watch_integrity",
    "build_competitive_baseline_watch_registry",
    "get_competitive_watch",
    "iter_competitive_watch",
    "list_competitor_ids",
    "probe_feed",
    "probe_refresh",
]
