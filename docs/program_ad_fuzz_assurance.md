# Rust Program AD fuzz assurance

Versioned **fuzz-assurance product** over ambient
`scpn_quantum_engine/fuzz` cargo-fuzz bins: target catalogue, time-boxed
CI-optional policy, and dry-run probe helpers. Does **not** execute cargo-fuzz
or invent-green continuous multi-hour coverage.

Module: `scpn_quantum_control.program_ad_fuzz_assurance`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `program_ad_fuzz_assurance.v2` |
| Default target | `program_ad_ir` |
| Default time box | 300 s |
| Max time box | 3600 s (hard product bound) |
| Continuous fuzz default | **False** |
| Invent-green continuous coverage | **Forbidden** |
| Blank/unknown target | Fail closed |
| cargo-fuzz execution | Not performed by this product module |

Claim boundary:

> Rust Program AD fuzz assurance product only; catalogues ambient
> scpn_quantum_engine/fuzz targets and time-boxed CI-optional policy; does not
> execute cargo-fuzz or invent-green continuous multi-hour coverage; multi-day
> corpus retention, automated crash-to-regression conversion, and
> parity-certificate fuzz-case ingestion remain unimplemented

## Public API

```python
from scpn_quantum_control.program_ad_fuzz_assurance import (
    assert_fuzz_assurance_integrity,
    build_fuzz_assurance_registry,
    dry_run_fuzz_target,
    fuzz_assurance_policy,
    list_fuzz_target_ids,
)

assert "program_ad_ir" in list_fuzz_target_ids()
policy = fuzz_assurance_policy()
assert policy.continuous_fuzz_default is False
assert policy.invent_green_forbidden is True

reg = assert_fuzz_assurance_integrity(build_fuzz_assurance_registry())
d = dry_run_fuzz_target("program_ad_ir")
assert d.allowed is True
assert d.time_box_seconds == 300

refused = dry_run_fuzz_target("program_ad_ir", request_continuous=True)
assert refused.allowed is False
```

## Targets

| Target | Cargo bin path |
|---|---|
| `program_ad_ir` | `scpn_quantum_engine/fuzz/fuzz_targets/program_ad_ir.rs` |
| `studio_kuramoto_input` | `…/studio_kuramoto_input.rs` |
| `ml_dsa_ntt` | `…/ml_dsa_ntt.rs` |
| `knm_validators` | `…/knm_validators.rs` |

`list_fuzz_target_ids()` returns these identifiers in deterministic catalogue
order. `get_fuzz_target(target_id)` performs exact lookup after trimming outer
whitespace; blank and unknown identifiers raise `ValueError` rather than
inventing an assurance result. `iter_fuzz_targets(posture=..., kind=...)`
applies optional posture and target-kind filters and returns immutable
`FuzzTarget` records. A query with no matches returns an empty tuple.

Each target record contains the identifier, title, summary, kind, ambient Rust
path, owning package, execution posture, parity-certificate pointer, API
stability class, inventory date, and shared claim boundary.
`FuzzTarget.to_dict()` exposes the same fields as a JSON-ready mapping.

## Policy and time bounds

`fuzz_assurance_policy()` returns the immutable `FuzzPolicy`:

- `default_time_box_seconds=300`;
- `max_time_box_seconds=3600`;
- `continuous_fuzz_default=False`;
- `ci_optional=True`; and
- `invent_green_forbidden=True`.

`validate_time_box_seconds(value)` accepts a positive integer no greater than
the policy maximum. Booleans, non-integers, zero/negative values, and values
above 3600 raise `ValueError`. Acceptance validates the requested bound only;
it does not start a process or reserve CI capacity.

## Dry-run decisions

`dry_run_fuzz_target(...)` validates the target and returns a
`FuzzProbeDecision`. With ordinary bounded input, the outcome is
`allowed_dry_run`, the selected time box is recorded, blockers are empty, and
the reason explicitly states that cargo-fuzz was not executed.

`request_continuous=True` and `request_invent_green_coverage=True` are refused
before time-box validation. If both are requested, the decision contains both
deduplicated blockers. Refusal uses `time_box_seconds=0`; it is an auditable
policy result, not a failed or cancelled fuzz process.

The decision record exposes target id, `allowed_dry_run` or `refused` outcome,
allow flag, reason, blocker tuple, selected time box, and claim boundary.

## Corpus and crash boundaries

`corpus_governance_policy()` identifies the ambient corpus and artifact paths,
sets `retention_ops_implemented=False`, and records
`open_capability="multi_day_corpus_retention"`.
`crash_pipeline_policy()` sets `automated_pipeline_implemented=False` and
records
`open_capability="automated_crash_to_regression_conversion"`. These mappings
are honest capability declarations; they do not create directories, retain
artifacts, triage crashes, or generate regression tests.

## Registry and integrity

`map_fuzz_public_surfaces()` returns deterministic rows for this Python product
and the ambient Rust fuzz package, including roles, stability class, target
identifiers, and claim boundary. `build_fuzz_assurance_registry()` combines
those surfaces with the schema, counts, default target, policy, corpus/crash
boundaries, canonical target rows, and policy note.

`assert_fuzz_assurance_integrity(payload=None)` validates either an explicit
payload or a freshly built registry. It rejects missing/empty target lists,
non-mapping rows, blank or duplicate identifiers, invalid postures, missing
Rust paths, absent default target, drift from the canonical target set,
non-zero blank counts, inconsistent target counts, missing/non-mapping policy,
a continuous default, or permission to invent green coverage. It returns a
shallow registry mapping after validation.

## Exported constants and types

`PROGRAM_AD_FUZZ_ASSURANCE_SCHEMA` identifies product registries and
`PROGRAM_AD_FUZZ_CLAIM_BOUNDARY` is carried through targets, policy decisions,
surfaces, and registries. `DEFAULT_TIME_BOX_SECONDS` and
`MAX_TIME_BOX_SECONDS` expose the bounded defaults. The literal vocabularies
are `TargetKind`, `TargetPosture`, and `ProbeOutcome`; the immutable records are
`FuzzTarget`, `FuzzPolicy`, and `FuzzProbeDecision`.

## Operational boundaries

Importing, listing, validating, dry-running, or building this registry does not
invoke cargo-fuzz, execute Rust/PyO3 code, run a long-lived CI job, mutate a
corpus, ingest a crash, create a regression, feed a parity certificate, contact
a provider, or promote a coverage/security claim. Those actions require their
own governed implementation and evidence.

## Bounded product status

Shipped: target catalogue · time-boxed CI-optional policy · dry-run probe ·
corpus/crash capability-boundary policies · docs.

Open: multi-day corpus retention · automated crash-to-regression conversion ·
parity-certificate fuzz-case ingestion · live multi-hour CI cargo-fuzz job
wiring.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
