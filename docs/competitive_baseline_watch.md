# Continuous competitive baseline watch (BL-61 / W1)

Fail-closed **ops watch** over differentiable competitive baselines. Composes the
committed refresh inventory (`differentiable_competitive_baselines`) into a
queryable catalogue with pin / version / refresh honesty and structured feed
probes toward BL-52 (route matrix) and BL-56 (scorecard engine).

Module: `scpn_quantum_control.competitive_baseline_watch`

## Rules

| Field | Honesty rule |
|---|---|
| `pin_status=pinned_snapshot` | Non-empty declared version from inventory (not live scrape) |
| `pin_status=unpinned` | Blank version — **cannot** be green/current |
| `refresh_state` | `pending_verification` / `due` / `blocked` with blockers |
| `allowed_green_current` | Always false until continuous re-pin automation lands |
| BL-56 feed | Always **blocked** (no invent-green promotion from watch alone) |
| BL-52 feed | Pointer-only / pending; no automatic matrix mutation |

Claim boundary:

> competitive baseline watch only; pinned_snapshot records declared comparison
> coverage from the committed refresh inventory; refresh/feed probes never invent
> live scrape wins or promote scorecard categories to at_baseline/exceeds_baseline
> without BL-56 evidence packages

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

feed = probe_feed("catalyst", feed_target="bl56_scorecard")
assert feed.allowed is False
assert feed.status == "blocked"
```

## Bounded product status

Shipped: S61.0 competitor set + cadence fields · S61.1 snapshot schema fields ·
S61.2 compose from competitive_baselines · S61.4/S61.5 feed **probes** (fail-closed
pointers/blockers).

Open: S61.3 Catalyst harness CI job · automated continuous re-pin · accepted
BL-52 boundary-update packages · S61.6 human changelog automation.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
