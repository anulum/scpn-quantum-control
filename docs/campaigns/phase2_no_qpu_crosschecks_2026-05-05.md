# Phase 2 no-QPU cross-checks

Date: 2026-05-05

These checks spend no additional IBM QPU time. They use only promoted Phase 2
raw counts and derived summaries.

## Readout mitigation boundary

Script: `scripts/analyse_phase2_readout_mitigation.py`

Output:
`data/phase2_readout_mitigation/phase2_readout_mitigation_summary_2026-05-05.json`

The existing raw counts include selected readout-only calibration states, not a
complete computational-basis calibration set. Therefore the repository cannot
honestly claim a full `2^n x 2^n` confusion-matrix inversion without new QPU
calibration circuits.

Implemented correction:

- exact-state parity-confusion inversion,
- only for initial states with matching readout-only calibration rows,
- no full-count redistribution across all basis states.

Main result after correction:

- Phase 2 A+G `n=4` replication: Fisher `chi2=73.894393`,
  `p=4.160825e-08`, significant depths `5/10`.
- Popcount original contrast: Fisher `p=9.267543e-18`.
- Popcount within-even swap: Fisher `p=4.466548e-20`.
- Popcount within-odd swap: Fisher `p=3.590153e-15`.
- Popcount excitation-inversion arm: Fisher `p=5.929924e-18`.

Conclusion: readout-only parity correction weakens the Phase 2 A+G combined
statistic but does not remove the promoted sign/statistical conclusion. The
popcount-control downgrading of the causal claim also survives correction.

## Offline GUESS-readiness calibration

Script: `scripts/analyse_phase2_guess_calibration.py`

Output:
`data/phase2_guess_calibration/phase2_guess_calibration_summary_2026-05-05.json`

This is not GUESS zero-noise extrapolation because the promoted Phase 2 datasets
do not include explicit folded-noise scale factors. The analysis fits
log parity-survival versus circuit depth as a cumulative-noise proxy.

Fit quality:

- Phase 2 A+G even: `alpha=0.034190`, `R2=0.9989`.
- Phase 2 A+G odd: `alpha=0.034662`, `R2=0.9997`.
- Phase 2 A+G sector mean: `alpha=0.034427`, `R2=0.9995`.
- Popcount state fits: `R2=0.9582` to `0.9921`.

Conclusion: parity leakage is smooth enough versus depth to justify preparing a
future GUESS-style folded-noise protocol. It is not yet a demonstrated error
mitigation result.

## Hardware follow-up preparation

Prepared protocol:
`docs/campaigns/next_validation_protocols_2026-05-05.md`

Recommended order:

1. Keep the current paper as a no-new-QPU preprint after incorporating the
   readout and GUESS-readiness boundaries.
2. Use the state/layout-randomization protocol only if reviewers or collaborators
   ask for mechanism separation beyond the popcount control.
3. Use the minimal multi-device protocol only if backend-transfer evidence is
   required before journal submission.
