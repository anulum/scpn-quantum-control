<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Combined Submission Checklist -->

# Combined Paper Submission Checklist

Date: 2026-05-06

This checklist is the portfolio-level pre-upload gate for the current
SCPN-QUANTUM-CONTROL paper set. It ties together the individual paper
checklists, final PDF/source rebuild requirements, arXiv/JOSS metadata, and
minimal AI-disclosure policy. It does not claim that submission has happened.

## Paper Set

| Package | Source | PDF | Checklist |
|---------|--------|-----|-----------|
| DLA parity hardware preprint | `paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity.tex` | `paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity.pdf` | `docs/publication/dla_parity_submission_checklist_2026-05-06.md` |
| Rust/VQE methods paper | `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.tex` | `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.pdf` | `docs/publication/rust_vqe_methods_submission_checklist_2026-05-06.md` |
| JOSS-style software note | `paper/submissions_joss/submission_joss_001_software_framework_note/paper.md`; preview `paper/submissions_joss/submission_joss_001_software_framework_note/paper_preview.tex` | `paper/submissions_joss/submission_joss_001_software_framework_note/paper_preview.pdf` | `docs/publication/joss_software_submission_checklist_2026-05-06.md` |
| SCPN/FIM Hamiltonian paper | `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian.tex` | `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian.pdf` | `docs/campaigns/scpn_fim_submission_checklist_2026-05-06.md` |

## Published Zenodo Records

The public publication sources embed the current Zenodo DOIs. Keep this index
aligned with the source files before any regenerated PDF or metadata upload.

| Package | DOI |
|---------|-----|
| DLA parity hardware preprint | [`10.5281/zenodo.20382000`](https://doi.org/10.5281/zenodo.20382000) |
| Reduced-Pauli entanglement checks | [`10.5281/zenodo.20382075`](https://doi.org/10.5281/zenodo.20382075) |
| Monitored feedback versus open-loop control | [`10.5281/zenodo.20382099`](https://doi.org/10.5281/zenodo.20382099) |
| SCPN/FIM Hamiltonian paper | [`10.5281/zenodo.20382125`](https://doi.org/10.5281/zenodo.20382125) |
| Rust/VQE methods paper | [`10.5281/zenodo.20382138`](https://doi.org/10.5281/zenodo.20382138) |
| IBM Fez synchronisation paper | [`10.5281/zenodo.20382203`](https://doi.org/10.5281/zenodo.20382203) |
| JOSS-style software note | [`10.5281/zenodo.20382333`](https://doi.org/10.5281/zenodo.20382333) |

## Claim Boundaries That Must Survive Upload

- DLA parity: claim parity-sector/excitation-number correlated leakage
  asymmetry, not DLA-parity-only protection.
- Rust/VQE methods: claim selected hot-path acceleration and artefact-first
  workflow, not whole-workflow or universal VQE speedup.
- JOSS-style note: claim specialised reusable software infrastructure, not a
  general quantum simulator or quantum-advantage engine.
- SCPN/FIM: claim exact small-system structure and negative IBM hardware
  falsification, not hardware coherence protection.

## Final PDF Build Gate

Run each paper from its own source directory before upload:

```bash
cd paper/submissions/submission_002_phase1_dla_parity
pdflatex -interaction=nonstopmode -halt-on-error phase1_dla_parity.tex

cd ../rust_vqe_methods
pdflatex -interaction=nonstopmode -halt-on-error rust_vqe_methods.tex

cd ../scpn_fim_hamiltonian
pdflatex -interaction=nonstopmode -halt-on-error scpn_fim_hamiltonian.tex

cd ../joss/software_framework_note
pdflatex -interaction=nonstopmode -halt-on-error paper_preview.tex
```

If bibliography output changes, rerun the appropriate bibliography tool and
repeat `pdflatex` until references stabilise. Do not upload PDFs with unresolved
citations, missing bibliography entries, line-broken URLs, or malformed IBM job
IDs.

## Reproducibility Gate

Preferred no-QPU gates:

```bash
scpn-bench reproduce-methods
scpn-bench fim-all
```

Optional wider portfolio gate:

```bash
scpn-bench all --include-readout
```

These commands must analyse committed artefacts only. They must not submit IBM
jobs or contact provider-cloud execution services.

## arXiv Metadata Draft

Recommended primary categories:

| Paper | Primary category | Secondary categories |
|-------|------------------|----------------------|
| DLA parity hardware preprint | `quant-ph` | `physics.comp-ph` |
| Rust/VQE methods paper | `quant-ph` | `cs.SE`, `physics.comp-ph` |
| SCPN/FIM Hamiltonian paper | `quant-ph` | `physics.comp-ph` |

Suggested repository line for all arXiv metadata:

```text
Code and data: https://github.com/anulum/scpn-quantum-control
```

JOSS/pyOpenSci should use the JOSS checklist metadata gate rather than arXiv
categories.

## URL and Identifier Gate

Before upload, verify:

- GitHub URL: `https://github.com/anulum/scpn-quantum-control`.
- Documentation URL: `https://anulum.github.io/scpn-quantum-control`.
- ORCID: `0009-0009-3560-0851`.
- Contact: `protoscience@anulum.li`.
- DLA job IDs match `docs/publication/dla_parity_submission_checklist_2026-05-06.md`.
- FIM job IDs match `docs/campaigns/scpn_fim_submission_checklist_2026-05-06.md`.
- Zenodo DOI references resolve to the intended software or data records.
- No internal `.coordination/`, private dataset, backup, credential, or token
  path appears in public manuscript text.

## Minimal AI Disclosure Policy

Use only if required by the target venue. Keep it short:

> AI-assisted tools were used for drafting and editing support. Numerical
> claims, artefact provenance, scientific framing, authorship, and final
> responsibility were verified and retained by the author.

Do not list tool vendors or model names in public tracked files. Do not allow
the disclosure to imply that generated text was accepted as evidence.

## Upload Readiness Checklist

Before public submission:

- All individual paper checklists are complete.
- Final PDFs are rebuilt from committed source.
- All references and URLs resolve.
- All table values are traceable to committed JSON/CSV artefacts.
- No unsupported quantum-advantage, protection, backend-general, or
  DLA-parity-only claim appears.
- AI disclosure is absent where not required, or minimal where required.
- No new QPU spend is implied or hidden.
- Worktree is clean and the final submission commit is pushed only after
  explicit approval.

## QPU Boundary

The combined submission package requires no additional QPU time. IBM Quantum
Credits follow-up, multi-device replication, adaptive FIM, analogue FIM, and
new readout-calibration campaigns remain separate tasks requiring manifests,
QPU-time estimates, and explicit approval.
