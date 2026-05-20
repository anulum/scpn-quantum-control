# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 processing methodology

# Paper 0 Processing Methodology

This document records the repeatable method used for GOTM-SCPN Paper 0:
The Foundational Framework. It is the baseline that future Book II papers must
follow before they can be described as processed in this repository.

The method is source-bounded. It preserves source statements, claim candidates,
fixtures, and promotion gates. It does not convert a manuscript statement into
external scientific validation, measured-system evidence, simulator evidence, or
hardware evidence unless a later evidence-class gate explicitly does that work.

## Completion Standard

A paper is processed only when all of these artefacts exist:

| Required artefact | Paper 0 implementation |
|---|---|
| Source inventory | Local extraction package under `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/`. |
| Full text dump | `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source/paper0_full_dump_pp001-679.txt`. |
| Extraction index | `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/extraction/paper0_extraction_INDEX.md`. |
| Partitioned extraction parts | `paper0_extraction_part01` through `paper0_extraction_part05`. |
| Consolidated issue ledger | `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/extraction/paper0_extraction_FLAGS.md`. |
| Independent assessment | `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/synthesis/paper0_assessment.md`. |
| Stable source identifiers | Public register span `P0R00001` through `P0R06211`. |
| Generated validation package | `src/scpn_quantum_control/paper0/`. |
| Spec loaders and fixtures | `scpn_quantum_control.paper0.spec_loader` and generated validation modules. |
| Tests and gates | Generated tests plus Paper 0 lane and replay gates. |
| Public status surface | `docs/paper0/paper0_validation_register.md`. |
| Downstream pathway | `docs/paper0/paper0_experimental_pathway.md`. |
| Claim boundary | Source-accounting and fixture-preservation only. |

Future papers must not be marked complete from extraction notes alone. They need
the same source register, generated artefacts, tests, promotion gates, and public
claim-boundary page.

## Required Folder Contract

Every Book II foundational-paper package must use the same evidence-class
layout:

| Folder | Use |
|---|---|
| `source/` | Canonical source files, source metadata, and full source dumps. |
| `extraction/` | Extracted equations, definitions, mechanisms, flags, and extraction indexes. |
| `source_validation_artifacts/` | Stable source ledgers, validation specs, fixtures, promotion gates, and reconciliation reports. |
| `synthesis/` | Scientific synthesis and formalisation derived from extraction. |
| `validation_protocols/` | Preregistered or proposed protocols for validating claims. |
| `experiments/` | Prepared experiment packages, manifests, run plans, and submission bundles. |
| `results/` | Measured-system, simulator, QPU, or provider results tied to protocols. |
| `revisions/` | Corrections proposed back into the foundational paper or book text. |
| `legacy_pre_tuned_extraction/` | Historical extraction attempts that predate the tuned methodology. |

This separation is mandatory. Source claims, mathematical synthesis,
preregistration, execution packages, evidence, and theory corrections must not
be mixed in a flat directory. The separation is also the public claim-boundary
guard: a metaphysical or source statement remains a source statement until a
validation protocol and evidence package move it into a stronger evidence
class.

## Stage 1: Source Inventory

1. Record the canonical title, revision, page span, source file identity, and
   intended role in the Book II publication sequence.
2. Produce a zero-filtered text dump from the source PDF or manuscript export.
   Preserve page boundaries where the source permits it.
3. Store the dump beside the paper extraction package, not in the repository
   root.
4. Create an extraction index before detailed work starts. The index must define
   page ranges, batch ranges, target files, and completion status.
5. Declare whether the source is canonical, a revision candidate, or a legacy
   attempt.

Paper 0 used a single complete dump and an index that split the source into
large extraction parts by page range and batch range.

## Stage 2: Full-Fidelity Extraction

The extraction target is not a short summary. It preserves:

- definitions and glossary-level terms;
- equations and mathematical grammar;
- mechanisms and process claims;
- empirical hooks and proposed observables;
- falsification conditions;
- dependencies on earlier or later papers;
- internal contradictions, deprecations, or revision flags;
- missing constants, unspecified units, and unsupported parameter choices.

Each extraction part must link back to the index, name its source page range,
and retain enough local context that later processing does not require re-reading
the whole manuscript for every slice.

For Paper 0 the extraction was split into five public extraction parts plus a
consolidated flags file. Later papers should use the same partitioning idea, but
the exact number of parts should follow the source size.

## Stage 3: Flag and Assessment Ledger

After extraction, create two review surfaces:

1. A consolidated flags file for unresolved source issues.
2. A paper assessment that states what the source provides, what is unclear, and
   which downstream validation lanes are plausible.

Flags are not failures. They preserve scientific caution and prevent later
promotion from silently absorbing unsupported assumptions. Required flag classes
include notation conflict, deprecated formulation, missing units, missing
parameter derivation, unsupported empirical leap, ambiguous source hierarchy,
and external-evidence requirement.

## Stage 4: Stable Source Ledger

Convert extracted source claims into stable record identifiers. Paper 0 uses
`P0R` identifiers and publicly reports the completed span:

```text
P0R00001 through P0R06211
```

For future papers use the corresponding paper number in the identifier prefix,
for example `P1R00001` for Paper 1 and `P2R00001` for Paper 2. Identifiers must
be stable once promoted. If a later revision changes a source statement, append
or supersede records instead of renumbering the existing ledger.

Each source record needs at minimum:

- record identifier;
- source page or source-local span;
- source heading or section;
- extracted statement class;
- claim boundary;
- downstream lane candidate, if any;
- blocker or falsifier, if the statement can be promoted later.

## Stage 5: Validation Spec Generation

Generate validation specifications from coherent source slices. Each spec must
preserve:

- source start and source end identifiers;
- source record count;
- component label;
- extracted claim candidates;
- fixture fields;
- explicit non-promotion boundary;
- required evidence before any stronger claim class is allowed.

Generated files must carry the repository header and must be deterministic.
Hand-maintained corrections belong in the source ledger or generator input, not
as silent drift in generated outputs.

## Stage 6: Fixture and Loader Generation

For Paper 0, generated modules live under:

```text
src/scpn_quantum_control/paper0/
```

Future papers should use matching package namespaces, for example:

```text
src/scpn_quantum_control/paper1/
src/scpn_quantum_control/paper2/
```

Each generated fixture module should validate source-accounting structure, not
pretend to validate the physical claim. Loader functions must resolve paths
relative to the repository package layout and fail closed when required artefacts
are missing, stale, or malformed.

## Stage 7: Tests and Reproducibility Gates

The minimum test surface for a processed paper is:

- spec shape tests;
- source-span accounting tests;
- loader path tests;
- fixture result tests;
- public register drift tests;
- promotion-boundary tests;
- negative tests for missing or weakened claim-boundary fields.

Paper 0 additionally has downstream lane and K_nm replay gates. Those gates are
not required for every future paper at ingestion time, but every paper needs an
equivalent promotion-gate design before any statement moves beyond
source-accounting.

## Stage 8: Public Register and Claim Boundary

Publish a concise register page under `docs/` only after the generated package
and tests exist. The register must state both what completion means and what it
does not mean.

The required wording pattern is:

- the paper is ingested into a source-validation register;
- the register preserves source spans, specs, fixtures, loaders, and tests;
- the register is not external validation;
- stronger evidence classes require separate measured-system, simulator, or
  hardware artefacts.

Paper 0's public register is `docs/paper0/paper0_validation_register.md`.

## Stage 9: Downstream Pathway

Only after ingestion is complete should the paper feed downstream experimental
planning. A downstream pathway must separate:

- source accounting;
- executable specs;
- no-hardware validation;
- measured-system candidates;
- simulator evidence;
- hardware or QPU execution.

No hardware submission is allowed from source ingestion alone.

## Evidence Trail

Paper 0 processing is recoverable from three evidence classes:

1. Public extraction and assessment artefacts under
   `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/`.
2. Public generated package and documentation under `src/scpn_quantum_control/paper0/`
   and `docs/paper0/paper0_*`.
3. The Paper 0 source-validation artefact package under
   `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/`,
   containing the tuned extraction captures, canonical review ledgers,
   validation specs, fixture outputs, promotion gates, and reconciliation
   reports.
4. Append-only internal session and handover logs under `.coordination/`.

Public documentation intentionally does not copy internal logs. It references
the existence of the audit trail and the internal runbook while keeping the
reproducible public method in this document and the executable artefacts in
tracked source, data, and test files.

## Current Status of Papers 1 and 2

The current Paper 1 and Paper 2 folders should be treated as pre-method or
legacy material until rerun through this methodology.

| Paper | Current observed status | Required next action |
|---|---|---|
| Paper 1: Layer 1 - Quantum Biological | Legacy and partial extraction material, including archived earlier attempts. | Rebuild from source inventory through stable `P1R` ledger, generated specs, fixtures, tests, and public register. |
| Paper 2: Layer 2 - Neurochemical-Neurological | Raw dump and assessment only. | Start from source inventory and run the complete tuned extraction process. |

Neither Paper 1 nor Paper 2 should be represented as processed until the same
completion standard used for Paper 0 is satisfied.

## Replication Checklist for the Next Paper

Use this checklist before starting Paper 1:

1. Create the canonical paper directory with the final Book II paper number and
   title.
2. Place raw source material and full text dump inside that directory.
3. Create the extraction index with page and batch partitions.
4. Extract definitions, equations, processes, observables, falsifiers, and flags
   without compressing away technical content.
5. Write the consolidated flags file.
6. Write the assessment.
7. Promote extracted claims to stable source identifiers.
8. Generate validation specs.
9. Generate fixture modules and loaders.
10. Add focused tests for specs, loaders, fixtures, and claim boundaries.
11. Add a public validation-register page.
12. Add or update a downstream pathway only after ingestion is complete.
13. Run the focused verification gates and record the result in a fresh
    append-only session log.

This checklist is the minimum bar for calling the next Book II paper processed.
