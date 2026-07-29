# Research-lane registry API

Module: `scpn_quantum_control.analysis.research_lane_registry`

This API provides the typed BL-84 catalogue and its exact source-inventory
gate. It is deterministic, credential-free, and read-only except for the
separate evidence runner writing its requested output files.

## Constants

| Name | Contract |
|---|---|
| `RESEARCH_LANE_REGISTRY_SCHEMA` | Versioned serialization identifier, currently `scpn.research-lane-registry.v1` |
| `RESEARCH_LANE_REGISTRY_BOUNDARY` | Global non-promotion statement applied to every row and report |

## Enums

### `ResearchLaneMaturity`

Human-reviewed software maturity: `RESEARCH`, `PROTOTYPE`, or
`PRODUCT_CANDIDATE`. The last value does not itself productise a module.

### `ResearchLaneDiffHook`

Relationship to differentiable work:

- `NONE`: no registered differentiable relationship;
- `DIAGNOSTIC`: usable only as a diagnostic input;
- `CANDIDATE`: a possible hook requiring separate evidence;
- `BOUNDED_COMPOSITION`: consumed by an already bounded, separately governed route;
- `DEFERRED`: blocked behind an owner gate.

### `ResearchLaneClaimStatus`

Strongest claim class carried by the lane itself: `RESEARCH_ONLY`,
`DIAGNOSTIC_ONLY`, `EVIDENCE_BOUNDED`, or `REFUSE_ONLY`. Bounded evidence is
still limited to the cited artefact and global boundary.

## Records

### `ResearchLaneRecord`

```python
ResearchLaneRecord(
    module: str,
    summary: str,
    maturity: ResearchLaneMaturity,
    diff_hook: ResearchLaneDiffHook,
    claim_status: ResearchLaneClaimStatus,
    promotion_targets: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
)
```

The frozen, slotted record validates its namespace, non-empty summary,
duplicate-free target/evidence tuples, evidence requirements, and promotion
route requirements. `family` returns `analysis` or `gauge`.
`registry_grants_productisation`, `registry_grants_control`, and
`registry_grants_publication_claim` always return `False`. `as_dict()` emits
enum values, lists, and all three explicit denials as JSON-ready primitives.

Construction raises `ValueError` for malformed or internally inconsistent
rows. The governance module cannot register itself.

### `ResearchLaneInventoryReport`

```python
ResearchLaneInventoryReport(
    registered_modules: tuple[str, ...],
    discovered_modules: tuple[str, ...],
    missing_modules: tuple[str, ...],
    orphaned_records: tuple[str, ...],
)
```

`passed` is true only when both drift tuples are empty. `as_dict()` includes
sets as sorted lists plus registered/discovered counts.

### `ResearchLaneRegistryReport`

Carries the schema, global boundary, immutable rows, inventory report, and a
SHA-256 content digest. `as_dict()` additionally renders maturity, diff-hook,
and claim-status counts. The digest covers the payload before the digest field
is added, avoiding self-reference.

## Lookup and listing

### `list_research_lanes()`

Returns the immutable tuple in canonical module order. It takes no arguments
and performs no discovery.

### `get_research_lane(module)`

Returns the exact reviewed row for a fully qualified module path. Relative
names are deliberately not expanded. Raises `KeyError` when the module is not
registered.

## Discovery and validation

### `discover_research_lane_modules(package_root=None)`

Scans top-level `*.py` files in `analysis/` and `gauge/`. `package_root` must be
the directory containing those two packages; when omitted, the installed
package root is used. Returns sorted fully qualified names. Raises
`FileNotFoundError` if either package directory is missing.

### `validate_research_lane_inventory(discovered_modules=None)`

Returns an inventory report. Supplying an iterable bypasses filesystem
discovery and is useful for CI fixtures or packaged consumers. Duplicate input
names are normalized. This function reports drift without raising.

### `assert_research_lane_inventory(discovered_modules=None)`

Returns the passing report or raises `RuntimeError` containing both the
unregistered discovered modules and orphaned rows. Use this form for CI.

## Evidence output

### `build_research_lane_registry_report()`

Runs the strict inventory assertion, builds count summaries, canonicalizes the
payload as sorted compact JSON, and computes its SHA-256 digest. Raises
`RuntimeError` on inventory drift.

### `render_research_lane_registry_markdown(report=None)`

Renders the full matrix as deterministic Markdown ending in one newline. With
no argument it first builds and validates the report. Passing a prebuilt report
avoids repeated discovery.

## Full autodoc

::: scpn_quantum_control.analysis.research_lane_registry
    options:
      show_root_heading: false
      show_source: false
      members_order: source
