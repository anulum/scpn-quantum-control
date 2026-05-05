# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 2 publication package

# Phase 2 Publication Package -- 2026-05-05

This manifest defines the release-ready Phase 2 publication package for the
DLA parity hardware continuation. It is an index of committed artefacts only;
it does not introduce new analysis results.

## Validated scope

| Scope | Status |
|---|---|
| Phase 1 `n=4` DLA parity campaign | Promoted raw-count dataset. |
| Phase 2 A+G `n=4` replication | Promoted raw-count replication with same-day readout baseline. |
| Phase 2 B-C `n=6,8` scaling | Promoted mixed early-scaling evidence. |
| Phase 2 popcount control | Promoted excitation-count confound control. |
| Phase 2 D-E larger scaling | Not submitted; no hardware claim. |
| Phase 2 F/GUESS mitigation | Not submitted; no mitigation claim. |
| Multi-device replication | Not yet run; no device-independence claim. |
| Broad quantum advantage | Not claimed. |

## Hardware jobs and QPU use

| Block | Backend | Jobs | Circuits | Shots | IBM-reported QPU seconds |
|---|---|---|---:|---:|---:|
| A main | `ibm_kingston` | `d7stu94t738s73ch5keg` | 600 | 4096 | 660 |
| G readout | `ibm_kingston` | `d7su3tkt738s73ch5ql0` | 12 | 8192 | 27 |
| B-C scaling | `ibm_kingston` | `d7sudr2udops7397ae30` | 280 | 4096 | 305 |
| Popcount main | `ibm_kingston` | `d7svcnkt738s73ch7agg` | 360 | 4096 | not yet recorded |
| Popcount readout | `ibm_kingston` | `d7svhsaudops7397bp30` | 5 | 8192 | not yet recorded |

Promoted A+G plus B-C QPU use: 992 quantum seconds, or 16.53 minutes.
The popcount-control jobs completed successfully; IBM-reported QPU seconds are
pending dashboard reconciliation and are not included in that total.

## Raw data and integrity

| Artefact | SHA256 |
|---|---|
| `data/phase2_dla_parity/phase2_reduced_ag_2026-05-05T121357Z.json` | `7c5f2a32d5a113d916d84d26d27a69336846364d5ee23ba4621b059125e0f5d5` |
| `data/phase2_scaling_bc/phase2_scaling_bc_2026-05-05T124722Z.json` | `f9718c3789329dbaa96a1667f8a581e3d1774632b961a1760c044138ccab6550` |
| `figures/phase2/phase2_n4_replication_asymmetry.png` | `12d12e491a1499309dbcf1bb27cb994bbfa262f9f3ccac25647aa1250add99bd` |
| `figures/phase2/phase2_bc_scaling_mixed_asymmetry.png` | `1c8b93c34f346d3aea865bcec470dca31a879870af3df3ce12aa4ad4f3d015e0` |
| `data/phase2_popcount_control/phase2_popcount_control_2026-05-05T135318Z.json` | `f43cbd7e466a3267847b44a750aeba7801cbc52ef10e9808573ef7ed01ec3cf0` |

## Reproduction commands

Phase 2 A+G:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python \
  scripts/analyse_phase2_dla_parity.py --verify-integrity
```

Phase 2 B-C:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python \
  scripts/analyse_phase2_scaling_bc.py \
  data/phase2_scaling_bc/phase2_scaling_bc_2026-05-05T124722Z.json \
  --sha256 f9718c3789329dbaa96a1667f8a581e3d1774632b961a1760c044138ccab6550
```

Figures:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python \
  scripts/plot_phase2_dla_parity.py
```

Phase 2 popcount control:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python \
  scripts/analyse_phase2_popcount_control.py --verify-integrity
```

Paper build:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error phase1_dla_parity.tex
```

## Compact result table

| Block | Result | Interpretation |
|---|---|---|
| Phase 2 A+G | Fisher chi2 `140.671952`, p `3.773718e-20`, 6/10 depths significant. | Strong same-device `n=4` raw-count replication. |
| Phase 2 B-C `n=6` | Fisher chi2 `46.531552`, p `1.883218e-07`, 2/4 depths significant. | Mixed sign; significant negative depths at 8 and 20. |
| Phase 2 B-C `n=8` | Fisher chi2 `29.420107`, p `2.675193e-04`, 3/4 depths significant. | Positive middle-depth sign at 4, 8, and 14. |
| Phase 2 popcount control | Excitation-inversion Fisher chi2 `139.164854`, p `8.829457e-24`; within-sector swaps also significant. | Excitation count and state choice materially contribute; no DLA-parity-only causal claim. |

## Claim boundary

The package supports a narrow hardware claim:

The `n=4` parity-sector/excitation-number correlated leakage asymmetry is
reproduced from independently submitted Phase 2 raw counts on `ibm_kingston`,
with same-day readout control. The first `n=6,8` scaling continuation is mixed:
`n=8` preserves a positive middle-depth sign, while `n=6` has negative
significant depths. The popcount-control follow-up shows that excitation count
and state choice materially contribute to the observed leakage pattern.

The package does not support:

- broad quantum advantage;
- monotone scaling validation;
- `n=10,12` hardware validation;
- GUESS mitigation validation;
- multi-device replication;
- mitigated observable-level performance claims;
- a DLA-parity-only causal interpretation.
