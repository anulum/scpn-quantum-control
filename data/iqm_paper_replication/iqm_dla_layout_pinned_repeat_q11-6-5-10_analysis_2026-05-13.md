# IQM Layout-Pinned DLA Minimal Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_dla_layout_pinned_repeat_q11-6-5-10_2026-05-13_executed.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Requested physical layout: `[11, 6, 5, 10]`
- Total circuits: `8`
- Total shots: `2048`
- Combined Fisher p-value: `0.208287`
- Sign matches vs IBM Phase 2: `1 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.535156 | 0.539062 | -0.007246 | 1 | +0.086512 | no |
| 6 | 0.546875 | 0.472656 | +0.157025 | 0.111455 | +0.043469 | yes |
| 10 | 0.445312 | 0.515625 | -0.136364 | 0.132567 | +0.044680 | no |

## Same-Layout Readout Baselines

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.167969 | 0.828125 |
| `0001` | odd | 0.109375 | 0.886719 |

## Claim Boundary

Layout-pinned IQM repeat is controlled diagnostic hardware evidence. It is not sufficient for manuscript claim upgrade until repeated statistics, calibration treatment, and cross-layout consistency are established.
