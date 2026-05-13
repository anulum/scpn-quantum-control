# IQM Layout-Pinned DLA Minimal Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_dla_layout_pinned_repeat_q17-18-19-15_2026-05-13_executed.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Requested physical layout: `[17, 18, 19, 15]`
- Total circuits: `8`
- Total shots: `2048`
- Combined Fisher p-value: `0.0460421`
- Sign matches vs IBM Phase 2: `2 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.445312 | 0.425781 | +0.045872 | 0.721486 | +0.086512 | yes |
| 6 | 0.433594 | 0.554688 | -0.218310 | 0.0079439 | +0.043469 | no |
| 10 | 0.562500 | 0.511719 | +0.099237 | 0.287503 | +0.044680 | yes |

## Same-Layout Readout Baselines

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.105469 | 0.894531 |
| `0001` | odd | 0.062500 | 0.937500 |

## Claim Boundary

Layout-pinned IQM repeat is controlled diagnostic hardware evidence. It is not sufficient for manuscript claim upgrade until repeated statistics, calibration treatment, and cross-layout consistency are established.
