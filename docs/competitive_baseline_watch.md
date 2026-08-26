# Continuous competitive baseline watch

Fail-closed **ops watch** over differentiable competitive baselines. Composes the
committed refresh inventory (`differentiable_competitive_baselines`) into a
queryable catalogue with pin / version / refresh honesty and structured feed
probes toward the governed route matrix and scorecard acceptance engine.

Module: `scpn_quantum_control.competitive_baseline_watch`

## Rules

| Field | Honesty rule |
|---|---|
| `pin_status=pinned_snapshot` | Non-empty declared version from inventory (not live scrape) |
| `pin_status=unpinned` | Blank version — **cannot** be green/current |
| `refresh_state` | `pending_verification` / `due` / `blocked` with blockers |
| `allowed_green_current` | Always false until continuous re-pin automation lands |
| Scorecard feed | Always **blocked** (no invent-green promotion from watch alone) |
| Route-matrix feed | Pointer-only / pending; no automatic matrix mutation |

Claim boundary:

> competitive baseline watch only; pinned_snapshot records declared comparison
> coverage from the committed refresh inventory; refresh/feed probes never invent
> live scrape wins or promote scorecard categories to at_baseline/exceeds_baseline
> without accepted scorecard evidence packages

## Public API

```python
from scpn_quantum_control.competitive_baseline_watch import (
    assert_competitive_baseline_watch_integrity,
    build_competitive_baseline_watch_registry,
    list_competitor_ids,
    probe_feed,
    probe_refresh,
)

reg = assert_competitive_baseline_watch_integrity(
    build_competitive_baseline_watch_registry()
)
assert reg["blank_entry_count"] == 0
assert len(list_competitor_ids()) == 9

probe = probe_refresh("catalyst")
assert probe.allowed_green_current is False
assert probe.blockers

feed = probe_feed("catalyst", feed_target="scorecard_acceptance")
assert feed.allowed is False
assert feed.status == "blocked"
```

## Vocabulary and immutable records

The module exposes four closed string vocabularies:

| Type | Values | Meaning |
|---|---|---|
| `PinStatus` | `pinned_snapshot`, `unpinned`, `blocked` | Whether the committed inventory declares a usable version label |
| `RefreshState` | `fresh`, `due`, `blocked`, `pending_verification` | Classification against the deterministic inventory date |
| `FeedTarget` | `governed_route_matrix`, `scorecard_acceptance` | Downstream surface receiving structured pointers |
| `FeedStatus` | `ready_pointer`, `blocked`, `pending` | Readiness of the pointer payload, not permission to promote a claim |

`CompetitiveWatchRecord` is the canonical immutable catalogue row. It carries
the competitor identity, declared upstream version and source, snapshot dates,
scorecard-category pointers, both feed states, blockers, the deterministic
`as_of` date, and the shared claim boundary. Construction validates every
closed vocabulary and non-empty field. In particular:

- an unpinned row cannot be `fresh` or carry an upstream version;
- a pinned snapshot must carry a non-empty upstream version;
- every non-green refresh state must explain itself with blockers; and
- Scorecard promotion remains blocked until a separate evidence package exists.

`RefreshProbeResult` and `FeedProbeResult` are likewise frozen, slot-backed
records. Their `to_dict()` methods return JSON-ready mappings: tuple fields are
materialised as lists, while the original records remain immutable.

## Catalogue access

### `list_competitor_ids()`

Returns all required competitor identifiers as a tuple in canonical inventory
order. The function performs no discovery and does not consult the network.

### `get_competitive_watch(competitor_id)`

Returns the immutable row for a non-blank known identifier. Blank or unknown
identifiers raise `ValueError`; the API never synthesises a row for an
unrecognised competitor.

```python
row = get_competitive_watch("jax")
assert row.competitor_id == "jax"
assert row.scorecard_feed_status == "blocked"
```

### `iter_competitive_watch(*, pin_status=None, refresh_state=None)`

Returns a stable tuple filtered by either closed vocabulary. Supplying both
filters applies their intersection. With neither filter it returns the full
canonical catalogue.

```python
pinned_pending = tuple(
    row
    for row in iter_competitive_watch(
        pin_status="pinned_snapshot",
        refresh_state="pending_verification",
    )
)
```

## Probe policy

### `probe_refresh(competitor_id)`

Builds a `RefreshProbeResult` from the committed row. It does not scrape,
re-pin, or update any source. `allowed_green_current` remains false while the
continuous verification blockers are present, even when the snapshot lies
inside its declared freshness window. Blank or unknown identifiers raise
`ValueError` through the catalogue lookup.

### `probe_feed(competitor_id, *, feed_target)`

Builds structured pointers for one of the two explicit targets:

- `governed_route_matrix` returns competitor/category pointers with a pending,
  non-mutating decision; and
- `scorecard_acceptance` returns scorecard-category pointers with a blocked decision.

The result is advisory data. `allowed=False` means consumers must not mutate a
route matrix, scorecard, claim ledger, or public comparison. Blank/unknown
competitors and unsupported targets raise `ValueError`.

## Registry construction and integrity

`build_competitive_baseline_watch_registry()` returns the complete serialisable
payload. The registry includes its schema, claim boundary, policy note,
deterministic `as_of` date, competitor rows, count fields, and refresh/feed
summaries.

Pass that mapping to `assert_competitive_baseline_watch_integrity(payload)` at
storage or transport boundaries. Omitting `payload` validates a newly built
canonical registry. The validator returns a normalised dictionary on success
and raises `ValueError` for:

- an empty or non-mapping competitor collection;
- blank fields, unsupported vocabularies, duplicate or unknown identifiers;
- missing required competitors or inconsistent count fields;
- unpinned/fresh or pinned/blank-version contradictions;
- non-green rows without blockers; or
- any invented scorecard-ready state.

Integrity validation is deliberately stricter than JSON shape validation: it
protects the evidence and non-promotion semantics carried by the payload.

## Failure handling

Callers should treat `ValueError` as invalid input or invalid registry custody,
not as permission to fall back to a guessed competitor. `RuntimeError` is
reserved for an internally incomplete composed inventory. Neither exception
triggers refresh, network access, or downstream mutation.

## Operational non-effects

Importing or calling this module does **not**:

- access vendor endpoints, repositories, credentials, or provider hardware;
- claim that a declared snapshot was verified live;
- execute Catalyst, JAX, PennyLane, Qiskit, or another competitor runtime;
- modify governed routes, scorecards, claim ledgers, changelogs, or releases;
- establish performance, feature parity, scientific superiority, or market
  advantage.

Those actions require their own governed evidence and acceptance boundaries.

## Bounded product status

Shipped: competitor set and cadence fields · snapshot schema fields · composition
from `competitive_baselines` · fail-closed feed probes with pointers and blockers.

Open: Catalyst harness CI job · automated continuous re-pin · accepted
route-boundary update packages · human changelog automation.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
