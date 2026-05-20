# Paper 0 Source-Validation Artefacts

This directory is the Paper 0 source-validation artefact package. It contains
the recent tuned extraction, canonical review, validation-spec promotion,
fixture outputs, and coverage reconciliation material for Paper 0. Keep Paper 0
generated JSON, JSONL, Markdown reports, and fixture outputs here, inside the
Paper 0 publication package.

This package is intentionally limited to Paper 0 material from the tuned
full-fidelity extraction and subsequent source-validation promotion pipeline.
Operational handover/runbook notes live outside this package under ignored
internal notes.

## File Families

- `paper0_block_inventory_*`, `paper0_headings_*`, `paper0_table_inventory_*`,
  `paper0_media_inventory_*`, `paper0_pandoc_ast.json`, `paper0_full_pandoc.md`:
  source extraction and manuscript structure.
- `paper0_equations_*`, `paper0_claim_candidates_*`,
  `paper0_mechanism_hits_*`, `paper0_exhaustive_register_*`,
  `paper0_domain_review_queue_*`: exhaustive record capture and review queues.
- `paper0_canonical_review_*`: canonical source ledger and canonical review
  reports used by promotion builders.
- `paper0_*_validation_specs_*.json` and
  `paper0_*_validation_specs_report_*.md`: promoted source-bounded validation
  specs.
- `paper0_*_fixture_result_*.json` and `paper0_*_fixture_report_*.md`:
  executable fixture outputs for promoted slices.
- `paper0_validation_coverage_reconciliation_*`: current reconciliation of
  promoted spans against the full Paper 0 ledger.
- `paper0_upde_*` and `paper0_topology_*`: UPDE/topology-specific aggregate
  artefacts generated during earlier promotion phases.
- `PAPER0_EQUATIONS_CAPTURED.md`, `paper0_foundational_extraction_index_*`,
  `paper0_validation_program_*`: human-facing index and planning notes for the
  same Paper 0 validation programme; these belong here, not in `docs/internal/`.
## Current Layout Rule

The Python loaders and promotion scripts currently resolve artefacts from this
directory root. Do not move these generated Paper 0 files into deeper
subdirectories unless the corresponding script defaults, spec loaders, tests,
and session handover notes are updated in the same change.

## Latest Verified Promotion State

The authoritative live state is the latest generated reconciliation report:

- `paper0_validation_coverage_reconciliation_2026-05-15.json`
- `paper0_validation_coverage_reconciliation_2026-05-15.md`

Do not rely on this README as a live counter after additional slices. At the
time of the runbook update, the current mid-slice worktree is promoting
`P0R00905-P0R00986`; after that slice is committed, the next boundary is
`P0R00987`.
