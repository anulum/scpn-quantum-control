<!--
SPDX-FileCopyrightText: 2026 Anulum
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Paper 0 — source-accounting register

## What this folder is

This package is the **systematic, machine-checked extraction of GOTM-SCPN
Paper 0: *The Foundational Framework*** — the foundational manuscript that is the
**canonical source of the SCPN theories and their mathematics and physics**. Each
module here corresponds to a bounded span of that manuscript: it carries the
generated configuration/result types for the span, the extracted source metadata,
and a validator that confirms the ingested fixtures preserve that metadata and
its claim boundaries.

In other words, this is where the book becomes code-addressable: ~470 modules,
one per Paper 0 section/derivation, generated from the extraction package under
`paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/`
(extraction parts 01–05, with a consolidated issue ledger). It is the research
trajectory and source-accounting register for the theory — the upstream from
which the workbench's `K_nm`, XY mapping, UPDE, and related constructs descend.

## What this folder is NOT

It is **a register, not a scientific-evidence promotion layer.** Passing
validators confirm *source ingestion, spec consistency, fixture shape, component
labelling, and claim-boundary preservation* — they do **not** establish external
experimental validation or hardware evidence for any Paper 0 claim. A claim
becomes promotable only through the hardware ledger and preregistered-replay
gates documented in [Methodology](../../../docs/methodology.md), never from an
extraction note alone.

## Why it is excluded from the quality gates

Because it is generated source-accounting for the manuscript rather than the
maintained product surface, Paper 0 is **excluded by design** from:

- the wheel and sdist (`[tool.hatch.build.targets.*] exclude`),
- coverage (`[tool.coverage.run] omit`),
- linting (`[tool.ruff] extend-exclude`), and
- type-checking (`[tool.mypy] exclude`).

This keeps the project's quality metrics measuring the hand-maintained codebase,
not the generated extraction corpus. Paper 0 has its own reconciliation test
(`tests/test_reconcile_paper0_validation_coverage.py`) and its public
methodology lives in [`docs/paper0/`](../../../docs/paper0/).

## Where to go next

- Processing methodology and claim classes: `docs/paper0/paper0_processing_methodology.md`.
- Validation register and lane registry: `docs/paper0/paper0_validation_register.md`, `docs/paper0/paper0_lane_registry.md`.
- Preregistered downstream replay: `docs/paper0/paper0_knm_preregistered_replay.md`.

Later Book II papers will receive their own packages once rerun with the tuned
Paper 0 extraction method.
