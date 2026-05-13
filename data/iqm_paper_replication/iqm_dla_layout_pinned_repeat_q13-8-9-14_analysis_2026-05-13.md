# IQM Layout-Pinned DLA Minimal Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_dla_layout_pinned_repeat_q13-8-9-14_2026-05-13_executed.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Requested physical layout: `[13, 8, 9, 14]`
- Total circuits: `8`
- Total shots: `2048`
- Combined Fisher p-value: `0.138807`
- Sign matches vs IBM Phase 2: `1 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.414062 | 0.457031 | -0.094017 | 0.372776 | +0.086512 | no |
| 6 | 0.425781 | 0.527344 | -0.192593 | 0.0268594 | +0.043469 | no |
| 10 | 0.457031 | 0.441406 | +0.035398 | 0.789854 | +0.044680 | yes |

## Same-Layout Readout Baselines

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.035156 | 0.957031 |
| `0001` | odd | 0.074219 | 0.925781 |

## Claim Boundary

Layout-pinned IQM repeat is controlled diagnostic hardware evidence. It is not sufficient for manuscript claim upgrade until repeated statistics, calibration treatment, and cross-layout consistency are established.
