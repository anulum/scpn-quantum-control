<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — SCPN/FIM Submission Checklist -->

# SCPN/FIM Hamiltonian Paper Submission Checklist

Date: 2026-05-06

This checklist freezes the submission boundary for the SCPN/FIM Hamiltonian
paper. It indexes committed artefacts only and preserves the negative hardware
falsification result as a scientific boundary, not as a failed result to hide.

## Submission Scope

Supported:

- Definition and exact small-system characterisation of
  `H_FIM(lambda) = -lambda M^2 / n`.
- Magnetisation-sector energy shifts and ideal-sector conservation checks.
- Exact spectra, adjacent-gap diagnostics, low-energy entanglement diagnostics,
  sector-survival predictions, and n=4 FIM VQE scoring.
- IBM Heron r2 pilot and repeated follow-up on `ibm_kingston`.
- Repeated follow-up falsification: in the tested digital Trotter circuit
  family, `lambda = 4` increased leakage and reduced exact-state retention
  relative to `lambda = 0`.
- Full 16-state readout-matrix mitigation only for the repeated dataset where
  the complete calibration basis exists.

Not supported:

- FIM improves hardware coherence in this implementation.
- Backend-general FIM behaviour.
- Hardware many-body localisation.
- Broad quantum advantage.
- Strict many-body localisation from small-n adjacent-gap diagnostics.
- A claim that the negative result rules out all native, analogue, pulse-level,
  or adaptive FIM implementations.

## Required Claim Wording

Use the conservative claim:

> The collective FIM term is mathematically well defined and reshapes the exact
> small-system Kuramoto-XY spectrum, but the tested IBM Heron r2 digital
> Trotter implementation falsifies the simple hardware-protection hypothesis:
> `lambda = 4` increased leakage relative to `lambda = 0`.

Avoid the stronger claim:

> The FIM term protects quantum coherence on hardware.

The abstract, conclusion, and hardware section must make the negative
falsification visible. It is part of the contribution.

## Committed Artefact Index

| Item | Path |
|------|------|
| Claim boundary | `docs/scpn_fim_claim_boundary_2026-05-05.md` |
| Repeated follow-up analysis note | `docs/scpn_fim_repeated_followup_analysis_2026-05-05.md` |
| Repeated follow-up protocol | `docs/scpn_fim_repeated_followup_protocol_2026-05-05.md` |
| Validation protocol | `docs/scpn_fim_validation_protocol_2026-05-05.md` |
| Paper source | `paper/scpn_fim_hamiltonian.tex` |
| Paper PDF | `paper/scpn_fim_hamiltonian.pdf` |
| Exact spectra summary | `data/scpn_fim_hamiltonian/fim_spectrum_summary_2026-05-05.json` |
| Level-spacing summary | `data/scpn_fim_hamiltonian/fim_level_spacing_summary_2026-05-05.json` |
| Entanglement summary | `data/scpn_fim_hamiltonian/fim_entanglement_summary_2026-05-05.json` |
| Sector-survival prediction | `data/scpn_fim_hamiltonian/fim_sector_survival_prediction_2026-05-05.json` |
| FIM VQE summary | `data/scpn_fim_hamiltonian/fim_vqe_ground_state_summary_2026-05-05.json` |
| Pilot raw counts | `data/scpn_fim_hamiltonian/fim_ibm_pilot_raw_counts_2026-05-05_ibm-run-4c0bd60c3fc2c532.json` |
| Repeated raw counts | `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_raw_counts_2026-05-05_ibm-run-cf4835290f607387.json` |
| Repeated analysis | `data/scpn_fim_hamiltonian/fim_ibm_repeated_followup_analysis_2026-05-05_ibm-run-cf4835290f607387.json` |
| Full-basis readout matrix mitigation | `data/scpn_fim_hamiltonian/fim_readout_matrix_mitigation_summary_2026-05-05_ibm-run-cf4835290f607387.json` |

## Generator Scripts

| Artefact group | Generator |
|----------------|-----------|
| Exact spectra | `scripts/analyse_fim_spectrum.py` |
| Level spacing | `scripts/analyse_fim_level_spacing.py` |
| Entanglement | `scripts/analyse_fim_entanglement.py` |
| Sector survival | `scripts/analyse_fim_sector_survival.py` |
| FIM VQE | `scripts/benchmark_fim_vqe_ground_state.py` |
| Pilot analysis | `scripts/analyse_fim_ibm_pilot.py` |
| Repeated follow-up analysis | `scripts/analyse_fim_ibm_repeated_followup.py` |
| Full-basis readout mitigation | `scripts/analyse_fim_readout_matrix_mitigation.py` |

## IBM Job IDs That Must Match the Paper

| Block | Backend | Job ID |
|-------|---------|--------|
| FIM pilot | `ibm_kingston` | `ibm-run-4c0bd60c3fc2c532` |
| FIM repeated follow-up | `ibm_kingston` | `ibm-run-cf4835290f607387` |

Do not allow line-break artefacts, punctuation changes, or invented job IDs.

## Reproducibility Gate

Preferred offline gate:

```bash
scpn-bench fim-all
```

Optional full portfolio gate:

```bash
scpn-bench all --include-readout
```

These commands analyse committed artefacts only. They must not submit IBM jobs.

## Final Manual Pre-Upload Gate

Before arXiv or journal upload:

- Rebuild `paper/scpn_fim_hamiltonian.pdf` from
  `paper/scpn_fim_hamiltonian.tex`.
- Verify every table value against the committed JSON/CSV artefact cited in
  the paper.
- Verify the paper states `H_FIM(lambda) = -lambda M^2 / n` and the
  `M^2 = n I + 2 sum_{i<j} Z_i Z_j` hardware-burden implication clearly.
- Verify the repeated IBM result is described as a negative/falsifying result.
- Verify the full-basis readout mitigation claim is restricted to the repeated
  dataset with the complete 16-state calibration basis.
- Verify no hardware-protection, backend-general, or quantum-advantage claim
  appears.
- Verify the GitHub URL is `https://github.com/anulum/scpn-quantum-control`.
- Verify AI disclosure, if required by the venue, is minimal and venue
  compliant.

## QPU Boundary

No additional QPU time is required for this submission package. Native analogue
FIM, pulse-level FIM, adaptive-lambda feedback, and multi-backend FIM
replication remain follow-up experiments requiring separate manifests,
calibration gates, QPU-time estimates, and explicit approval.
