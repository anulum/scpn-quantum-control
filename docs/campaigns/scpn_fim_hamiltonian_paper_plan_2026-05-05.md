# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — SCPN/FIM Hamiltonian Paper Plan

# SCPN/FIM Self-Referential Hamiltonian Paper Plan

Date: 2026-05-05

## Working title

`Fisher-information-inspired collective feedback in Kuramoto-XY quantum Hamiltonians`

Alternative:

`Magnetisation-sector structure induced by a self-referential FIM term in Kuramoto-XY quantum simulation`

## Purpose

This paper is the theory and computational-mechanism track behind the SCPN
quantum-control programme. It studies whether a collective
Fisher-information-inspired feedback term changes synchronisation, spectral
statistics, magnetisation-sector structure, localisation-like diagnostics, and
hardware-facing decoherence predictions in Kuramoto-XY quantum simulations.

The paper must remain separate from the DLA-parity hardware paper and the
Rust/VQE methods paper:

- The DLA-parity paper is the real-hardware phenomenology and validation anchor.
- The Rust/VQE methods paper is the reproducible software and benchmark anchor.
- This paper is the Hamiltonian-mechanism and falsifiable-prediction anchor.

## Core Hamiltonian

The model family is:

```text
H(lambda) = H_XY + H_FIM(lambda)
H_FIM(lambda) = -lambda * M^2 / n
M = sum_i Z_i
```

where `H_XY` is the Kuramoto-derived heterogeneous XY Hamiltonian and `M` is
the total magnetisation operator. The FIM term is interpreted conservatively as
a collective magnetisation-feedback term inspired by information geometry.

## Safe claim boundary at project start

Safe now:

- The Hamiltonian family is mathematically definable.
- The term creates an explicit sector-dependent energy contribution through
  `M^2`.
- The existing package has the infrastructure needed to generate Hamiltonians,
  VQE ansaetze, exact small-n references, benchmark artefacts, and IBM hardware
  packages.
- The previous DLA-parity study demonstrates that hardware claims must be
  controlled for depth, popcount, layout, readout, and calibration context.

Not safe yet:

- Do not claim a demonstrated universal protection principle.
- Do not claim quantum advantage.
- Do not claim strict many-body localisation unless the diagnostics support the
  term and the limitations are stated.
- Do not claim hardware robustness until equal-depth IBM runs exist.
- Do not use old internal or application numbers as paper claims unless they are
  regenerated into committed JSON/CSV artefacts.

## Scientific questions

1. Does `H_FIM` produce a predictable magnetisation-sector energy structure?
2. Does increasing `lambda` change level-spacing statistics relative to plain
   heterogeneous XY?
3. Does the FIM term reduce entanglement growth or scrambling in a way that is
   visible in small-n exact simulation?
4. Does the term predict sector-resolved survival or leakage differences under
   calibrated noise models?
5. Which predicted effects are large enough to justify a minimal IBM pilot?
6. If IBM hardware disagrees with simulation, which confound explains the
   failure: excitation count, circuit depth, layout, readout, calibration drift,
   or the absence of a real FIM mechanism?

## Required offline artefacts before IBM runs

Every numerical paper claim must come from committed artefacts.

Minimum artefact set:

- `data/scpn_fim_hamiltonian/fim_spectrum_summary_2026-05-05.json` —
  generated initial n=4,6,8 exact spectrum artefact.
- `data/scpn_fim_hamiltonian/fim_spectrum_summary_2026-05-05.csv` —
  generated compact spectrum table.
- `data/scpn_fim_hamiltonian/fim_sector_spectrum_summary_2026-05-05.csv` —
  generated magnetisation-sector spectrum table.
- `data/scpn_fim_hamiltonian/fim_level_spacing_summary_2026-05-05.json` —
  generated full-spectrum and sector adjacent-gap-ratio artefact.
- `data/scpn_fim_hamiltonian/fim_level_spacing_summary_2026-05-05.csv` —
  generated adjacent-gap-ratio table.
- `data/scpn_fim_hamiltonian/fim_entanglement_summary_2026-05-05.json` —
  generated low-energy eigenstate bipartition-entropy artefact.
- `data/scpn_fim_hamiltonian/fim_entanglement_rows_2026-05-05.csv` —
  generated per-eigenstate entropy table.
- `data/scpn_fim_hamiltonian/fim_entanglement_aggregate_2026-05-05.csv` —
  generated entropy aggregate table.
- `data/scpn_fim_hamiltonian/fim_sector_survival_prediction_2026-05-05.json`
  — generated commutator and sector-conservation artefact.
- `data/scpn_fim_hamiltonian/fim_sector_survival_summary_2026-05-05.csv`
  — generated commutator and maximum off-sector-coupling table.
- `data/scpn_fim_hamiltonian/fim_sector_survival_rows_2026-05-05.csv`
  — generated per-sector energy-barrier table.
- `data/scpn_fim_hamiltonian/fim_vqe_ground_state_summary_2026-05-05.json`
  — generated small-n FIM VQE ground-state comparison artefact.
- `data/scpn_fim_hamiltonian/fim_vqe_ground_state_rows_2026-05-05.csv`
  — generated per-seed VQE table.
- `data/scpn_fim_hamiltonian/fim_vqe_ground_state_aggregate_2026-05-05.csv`
  — generated aggregate VQE table.
- `data/scpn_fim_hamiltonian/fim_ibm_candidate_protocol_2026-05-05.json`
  — generated non-submitting IBM pilot candidate protocol.
- `data/scpn_fim_hamiltonian/fim_ibm_candidate_protocol_2026-05-05.csv`
  — generated candidate circuit table.

Initial artefact hashes:

| Artefact | SHA256 |
| --- | --- |
| `fim_spectrum_summary_2026-05-05.json` | `451442c677b73419a5826fd22d3426f498a5e1186545987067ee2f3e240cef5e` |
| `fim_spectrum_summary_2026-05-05.csv` | `aadcd6426c66967f3f92441c2114def324c68199958dff5d6a23acf586efcc9f` |
| `fim_sector_spectrum_summary_2026-05-05.csv` | `9e957d1d16e16cf3f41bec0644cdc508586ff716e52f61810d723c91dcdddc53` |
| `fim_level_spacing_summary_2026-05-05.json` | `dfcc2882e561100c6c07afc2518e5b35c863628dc015c22f2d12e75ad931f959` |
| `fim_level_spacing_summary_2026-05-05.csv` | `4874687619524561b819e7397d5e6c599e35e9ea569015149c0f076a6dec1cf8` |
| `fim_entanglement_summary_2026-05-05.json` | `ea9a8b81ecad09b9e354e50e61053f816263e44686660e5a3fc6fcb147c2692a` |
| `fim_entanglement_rows_2026-05-05.csv` | `980e54502b61f409d2d631cc2cdc136ed841e16273702f247864a8a0772b9e5b` |
| `fim_entanglement_aggregate_2026-05-05.csv` | `3f0c1951e7491b0ec0cde01cd7abe0914cd598e7c91d2e57bf5c745187d109f5` |
| `fim_sector_survival_prediction_2026-05-05.json` | `addfa842cd81e1f38f3582baabf7cc3fdffc250ceeae66f922f61c6abde7fd72` |
| `fim_sector_survival_summary_2026-05-05.csv` | `a27931c7add47c08c3a3545a5e79952f73ce63e10dddc495bbddbab23623095e` |
| `fim_sector_survival_rows_2026-05-05.csv` | `d2f243c11a73b98851c436528328ac5345b665506bd27cf792209b904a4f65be` |
| `fim_vqe_ground_state_summary_2026-05-05.json` | `8d25ed4ba4593778b5f96b88ed1c571ebb04e03bcf779bdc0911a160e6792ecf` |
| `fim_vqe_ground_state_rows_2026-05-05.csv` | `c8ab197a7c7f5a60783ae76dc6f8d7c9eeb5abc75ae987ced721d9be81bdf759` |
| `fim_vqe_ground_state_aggregate_2026-05-05.csv` | `cf6305f59c7b207b41eda11654e9a15df176f65cd97404140e935657f0bb2d51` |
| `fim_ibm_candidate_protocol_2026-05-05.json` | `9b76136c7bc090f9738fcad58eab7f5b1b8bb5f26ede7ebfc32c114234407839` |
| `fim_ibm_candidate_protocol_2026-05-05.csv` | `a577bfbd082d4528ed6471dfa95ac186b7619fd1822be99a08cf5b160eda4ac4` |

Minimum scripts:

- `scripts/analyse_fim_spectrum.py` — implemented initial exact spectrum and
  magnetisation-sector summaries.
- `scripts/analyse_fim_level_spacing.py` — implemented initial adjacent-gap
  ratio summaries for full spectra and magnetisation sectors.
- `scripts/analyse_fim_entanglement.py` — implemented low-energy eigenstate
  bipartition-entropy summaries.
- `scripts/analyse_fim_sector_survival.py` — implemented commutator,
  ideal-sector-conservation, and per-sector energy-barrier summaries.
- `scripts/benchmark_fim_vqe_ground_state.py` — implemented small-n VQE
  ground-state scoring against exact dense diagonalisation.
- `scripts/prepare_fim_ibm_pilot.py` — implemented a non-submitting IBM pilot
  candidate protocol with QPU gates and falsification rule.

## Proposed paper structure

1. Introduction and motivation.
2. Kuramoto-XY mapping and collective FIM-inspired feedback.
3. Magnetisation-sector decomposition induced by `M^2`.
4. Exact small-n spectral and localisation-like diagnostics.
5. VQE and ansatz implications.
6. Noise-model survival predictions and IBM pilot design.
7. Discussion: what the term supports, what it does not support, and what the
   hardware can falsify.
8. Data and code availability.

Initial manuscript draft:

- `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian.tex`
- `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian_refs.bib`

## Publication stance

The paper should be framed as a theoretical/computational physics note with
hardware-facing predictions. It should not be framed as a hardware-confirmed
result until the IBM pilot exists.

The current manuscript-safe claim boundary is recorded in
`docs/campaigns/scpn_fim_claim_boundary_2026-05-05.md`.

The first IBM-readiness step is also complete: local, non-submitting circuit
preparation generated `fim_ibm_circuit_preparation_2026-05-05.json` and `.csv`.
This does not replace live backend transpilation.

Recommended venues after artefacts exist:

- arXiv preprint first.
- Quantum Science and Technology, Physical Review A, or New Journal of Physics
  depending on final scope and whether IBM data are included.
- If the result remains purely computational, SoftwareX/JOSS is not the right
  target; the methods paper already covers software.

## Decision gate for IBM runs

IBM runs are justified only after offline artefacts identify the smallest
falsification experiment with a predicted effect large enough to survive
hardware noise.

Minimum IBM pilot:

- n=4 only.
- `lambda = 0` plus two non-zero values selected from offline artefacts.
- Equal-depth circuits.
- Matched qubit layout where possible.
- Explicit magnetisation/popcount controls.
- Same-day readout calibration or parity-readout correction.
- Fixed shot budget and pre-registered stop rule.

Continuation to n=6 or n=8 requires the n=4 pilot to show a stable sign or a
clear falsification that motivates the next experiment.

## Immediate next step

Inspect the generated artefacts and draft the manuscript claim table. The
current generated artefacts establish the exact small-n spectrum,
magnetisation-sector energy structure, adjacent-gap diagnostics, low-energy
bipartition entropy, ideal sector-conservation checks, VQE ground-state
baseline, and a non-submitting IBM pilot candidate protocol.

Important scientific boundary: in the ideal model, `H_XY`, `H_FIM`, and
`H_XY + H_FIM` conserve total magnetisation. The FIM term changes sector
energies and low-energy structure, but it does not create ideal unitary leakage
between magnetisation sectors. Any IBM leakage or survival asymmetry must
therefore be tested as a noise, state-preparation, transpilation, layout, or
readout phenomenon.
