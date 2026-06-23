# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Methodology and evidence governance

# Methodology and Evidence Governance

This is the single entry point for the project's evidence-governance material.
You do not need to read it to run the examples — start with
[Onboarding](onboarding.md) and the [Example Gallery](examples_gallery.md). Read
this page when you need to understand *what a result in this repository is
allowed to claim* and *how a claim is promoted from simulation to hardware
evidence*. The deep registers, ledgers, and contracts are linked below rather
than inlined, so each stays the authoritative source for its own scope.

## Why this layer exists

A useful result here is not just a plot or a notebook output. It is a result
with its inputs, code path, dependency context, claim class, and promotion rule
recorded. The governance layer separates four altitudes of confidence so that a
fast simulation result is never read as a hardware claim:

- **simulation science** — fast iteration on the statevector simulator;
- **method verification** — reproducible checks, classical baselines, and
  preregistered replay;
- **hardware evidence** — claims backed by raw-count artefacts on real devices;
- **commercial readiness** — stable facades, release gates, and deployment
  boundaries.

Higher-cost routes carry a higher evidence burden. The rest of this page is a
map to where each rule and record lives.

## Claim classes and boundaries

- [Results & Claims](PAPER_CLAIMS.md) — the claim matrix and what each class
  permits.
- [Claim Boundary: SCPN/FIM](campaigns/scpn_fim_claim_boundary_2026-05-05.md) —
  a worked claim boundary for one experiment family.
- [Classical Irreproducibility](classical_irreproducibility.md) — why classical
  references are part of the evidence, not a footnote.
- [Licensing FAQ](licensing_faq.md) and
  [Core Package Boundary](core_package_boundary.md) — what is in scope for the
  stable surface versus the research workbench.

## Hardware evidence

- [Hardware Status Ledger](hardware_status_ledger.md) — the promotion state of
  every hardware claim.
- [Hardware Result Packs](hardware_result_packs.md) and the
  [Release Checklist](hardware_result_pack_release_checklist.md) — the
  raw-count-backed artefacts and how they are promoted.
- [IBM Guide](hardware_guide.md) — running on real devices.

## Runtime contracts

- [QPU Data Artifact](qpu_data_artifact.md) — the typed, hash-bound input
  contract.
- [Pipeline Runtime Contract](pipeline_runtime_contract.md) and
  [QPU Compute Unit](qpu_compute_unit.md).
- [QPU Provider Readiness](qpu_provider_readiness.md) — fail-closed provider
  boundaries.

## Validation protocols and readiness

- [Protocol: SCPN/FIM Validation](campaigns/scpn_fim_validation_protocol_2026-05-05.md).
- [Methods Benchmark Dashboard](methods_benchmark_dashboard.md) and the
  [Classical Baselines](classical_baselines.md) they compare against.
- Campaign readiness indices:
  [S1 Feedback](campaigns/s1_feedback_readiness_index_2026-05-06.md),
  [S1 Live Submission Preflight](campaigns/s1_live_submission_preflight_2026-05-06.md),
  [Hybrid Feedback Loop S1](campaigns/hybrid_feedback_loop_s1_2026-05-06.md),
  [S2 Scaling](campaigns/s2_scaling_readiness_index_2026-05-06.md), and
  [S3 Design](campaigns/s3_design_readiness_index_2026-05-06.md).
