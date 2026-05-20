# Venue format readiness: Rust/VQE methods paper

Date: 2026-05-05

## Recommended submission split

- JOSS: submit `paper/joss/software_framework_note/paper.md` with `paper/joss/software_framework_note/paper.bib`.
- SoftwareX / arXiv / institutional review: use `paper/rust_vqe_methods/rust_vqe_methods.tex` and `paper/rust_vqe_methods/rust_vqe_methods.pdf` as the longer methods preprint.

## JOSS readiness

JOSS expects a short Markdown paper, not a full LaTeX methods article. The local JOSS package now follows the current JOSS review expectations:

- YAML metadata with title, tags, author, ORCID, affiliation, date, and bibliography.
- `Summary`.
- `Statement of need`.
- `State of the field`.
- `Software design`.
- `Research impact statement`.
- `AI usage disclosure`.
- References including software archive DOI and companion hardware/data work.

Before submission, still check:

- Repository has a clear OSI-approved licence file.
- Installation instructions are current.
- Example usage is current.
- API/user documentation is reachable from README.
- Tests and CI status are documented.
- Latest release/version matches `CITATION.cff` and the JOSS paper.
- Community/support/contribution instructions are visible.

## SoftwareX readiness

The current `paper/rust_vqe_methods/rust_vqe_methods.tex` is a readable preprint, not yet an Elsevier production template. For SoftwareX, convert the same content into the current Elsevier/SoftwareX template and preserve:

- Metadata and author affiliations.
- Software motivation and impact.
- Architecture and implementation.
- Benchmark methodology and generated artefact provenance.
- Reproducibility/data availability.
- Limitations.
- References.

## Formatting decision

Use JOSS first if the goal is a software publication focused on reuse, documentation, tests, and research impact. Use SoftwareX if the goal is a longer methodological article with benchmark tables and implementation detail.
