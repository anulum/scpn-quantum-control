<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — IBM Quantum Credits Follow-Up Dossier -->

# IBM Quantum Credits Follow-Up Dossier

Date: 2026-05-06

This dossier closes the roadmap item for revisiting IBM Quantum Credits. It
does not send an email, submit a new application, or authorise QPU spend.

## Request Boundary

The follow-up should request a 5-10 hour QPU allocation. The scope must not be
shrunk to the first 10-15 minute validation block, because the scientific value
comes from completing the full controlled sequence:

1. multi-device DLA parity replication;
2. systematic state/layout randomisation;
3. full readout-confusion calibration where justified;
4. GUESS / symmetry-decay validation with explicit controlled-noise circuits;
5. limited scaling follow-up only after live depth and cost gates pass.

Each submitted job remains small and gated. The allocation size is for the full
programme, not for uncontrolled QPU use.

## Current Evidence Base

Use these committed package gates as support:

- `docs/dla_parity_submission_checklist_2026-05-06.md`
- `docs/rust_vqe_methods_submission_checklist_2026-05-06.md`
- `docs/joss_software_submission_checklist_2026-05-06.md`
- `docs/scpn_fim_submission_checklist_2026-05-06.md`
- `docs/combined_submission_checklist_2026-05-06.md`
- `docs/publication_phase2_package_2026-05-05.md`
- `docs/phase2_no_qpu_crosschecks_2026-05-05.md`

The follow-up should emphasise that the Open Plan work produced committed raw
counts, job IDs, SHA256 hashes, analysis scripts, and conservative claim
boundaries. It should not imply that all papers are already formally accepted
or submitted unless that has happened.

## Drafts

Current drafts:

- `paper/drafts/2026-05-05_kovos_ibm_credits_revisit_en.md`
- `paper/drafts/2026-05-05_ibm_quantum_allocation_request_en.md`
- `paper/drafts/2026-05-05_schneider_followup_de.md`

The Kovos draft already asks for a 5-10 hour programme. The allocation request
draft has been aligned to the same scope.

## Affiliation Boundary

Safe wording:

> We are in discussion with Prof. Dr. Johannes Schneider at the University of
> Liechtenstein regarding scientific review, possible co-authorship if he
> approves the manuscript after review, and potential institutional support for
> follow-up QPU allocation.

Avoid:

> University affiliation is already guaranteed.

Do not overstate institutional backing until written confirmation exists.

## QPU Spend Gate

Before any IBM follow-up run:

- confirm backend availability in the IBM dashboard;
- run live transpilation depth checks;
- record estimated QPU time for each block;
- confirm the remaining account budget;
- create a preregistered manifest;
- get explicit approval for the submission.

The credits request itself should ask for 5-10 hours, but the experiment runner
must still spend only the minimum necessary QPU time per validated block.
