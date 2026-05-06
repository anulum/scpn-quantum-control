# DLA Parity Preprint Submission Checklist

Date: 2026-05-06

This checklist freezes the conservative pre-submission boundary for the DLA
parity hardware paper. It indexes committed artefacts only and does not add new
analysis results or authorise QPU use.

## Submission Scope

Supported:

- Phase 1 `n=4` raw-count campaign under `data/phase1_dla_parity/`.
- Phase 2 A+G `n=4` replication under `data/phase2_dla_parity/`.
- Phase 2 B-C `n=6,8` mixed scaling continuation under
  `data/phase2_scaling_bc/`.
- Phase 2 popcount-control follow-up under `data/phase2_popcount_control/`.
- Exact-state parity-readout correction under `data/phase2_readout_mitigation/`.
- Offline GUESS-readiness depth fit under `data/phase2_guess_calibration/`.
- Phase 3 second-backend fixed-layout non-replication under
  `data/phase3_multidevice_dla/`.
- Phase 3 full-basis readout calibration under `data/readout_full_basis/`.
- Phase 3 state/layout randomisation under `data/phase3_state_layout_dla/`.

Not supported:

- DLA-parity-only causality.
- Broad quantum advantage.
- Monotone scaling.
- Backend-stable parity protection.
- Layout-independent parity protection.
- Full confusion-matrix readout mitigation for datasets without complete basis
  calibration.
- Demonstrated GUESS zero-noise extrapolation.
- Any `n=10,12` hardware claim.

## Required Claim Wording

Use the conservative claim:

> IBM Heron r2 hardware shows a replicated small-system leakage asymmetry in
> the heterogeneous Kuramoto-XY experiment, correlated with parity sector,
> excitation number, state choice, and layout/readout context.

Avoid the stronger claim:

> The odd DLA parity sector is intrinsically protected.

The popcount-control result must remain visible in the abstract, conclusion, or
claim-boundary paragraph. It is not a minor control; it is the reason the causal
language is downgraded.

The Phase 3 state/layout result must also remain visible in the abstract,
conclusion, or claim-boundary paragraph. It is the reason layout-independent and
same-sector-state-independent causal language is rejected.

## Committed Artefact Index

| Item | Path |
|------|------|
| Phase 2 package manifest | `docs/publication_phase2_package_2026-05-05.md` |
| No-QPU cross-checks | `docs/phase2_no_qpu_crosschecks_2026-05-05.md` |
| Hardware continuation protocol | `docs/next_validation_protocols_2026-05-05.md` |
| Phase 1 analysis | `scripts/analyse_phase1_dla_parity.py` |
| Phase 2 replication analysis | `scripts/analyse_phase2_dla_parity.py` |
| Phase 2 scaling analysis | `scripts/analyse_phase2_scaling_bc.py` |
| Popcount-control analysis | `scripts/analyse_phase2_popcount_control.py` |
| Readout-correction analysis | `scripts/analyse_phase2_readout_mitigation.py` |
| GUESS-readiness analysis | `scripts/analyse_phase2_guess_calibration.py` |
| Phase 3 state/layout analysis | `scripts/analyse_phase3_state_layout_dla.py` |
| Phase 3 state/layout manifest | `docs/phase3_state_layout_dla_manifest_2026-05-07.md` |
| Paper source | `paper/phase1_dla_parity.tex` |
| Paper PDF | `paper/phase1_dla_parity.pdf` |

## Job IDs That Must Match the Paper

| Block | Backend | Job ID |
|-------|---------|--------|
| Phase 2 A main | `ibm_kingston` | `d7stu94t738s73ch5keg` |
| Phase 2 G readout | `ibm_kingston` | `d7su3tkt738s73ch5ql0` |
| Phase 2 B-C scaling | `ibm_kingston` | `d7sudr2udops7397ae30` |
| Popcount main | `ibm_kingston` | `d7svcnkt738s73ch7agg` |
| Popcount readout | `ibm_kingston` | `d7svhsaudops7397bp30` |
| Phase 3 multi-device main | `ibm_marrakesh` | `d7tnd3vljm6s73bcsh6g` |
| Phase 3 multi-device readout | `ibm_marrakesh` | `d7tnehqudops73988btg` |
| Phase 3 full-basis readout | `ibm_marrakesh` | `d7tnljvljm6s73bcsql0` |
| Phase 3 state/layout main | `ibm_marrakesh` | `d7ts9avljm6s73bd2ej0` |
| Phase 3 state/layout readout | `ibm_marrakesh` | `d7tsdnfljm6s73bd2j70` |

Do not allow line-break artefacts, plus signs, or punctuation changes inside
these job IDs.

## Final Manual Pre-Upload Gate

Before arXiv or journal upload:

- Rebuild `paper/phase1_dla_parity.pdf` from `paper/phase1_dla_parity.tex`.
- Verify every path in the Data and Code Availability section matches committed
  files.
- Verify the GitHub URL is `https://github.com/anulum/scpn-quantum-control`.
- Verify the Zenodo DOI, if listed, resolves to the intended artefact package.
- Verify the paper does not claim full readout-matrix inversion for datasets
  lacking complete calibration basis circuits.
- Verify the AI disclosure, if required by the venue, is minimal and does not
  obscure human responsibility for claims and authorship.
- Verify no unpublished QPU job, cancelled run, or fake IBM result is cited.

## QPU Boundary

No additional QPU time is required for this submission package. The completed
Phase 3 second-backend, full-basis-readout, and state/layout controls are part
of the current claim boundary. Further multi-backend statistics, additional
layout randomisation, full missing readout calibration for older datasets, and
GUESS zero-noise extrapolation remain follow-up experiments requiring separate
offline manifests, QPU-time estimates, and explicit approval.
