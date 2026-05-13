# IQM Layout-Pinned DLA Minimal Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_dla_layout_pinned_repeat_q0-1-4-3_2026-05-13_executed.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Requested physical layout: `[0, 1, 4, 3]`
- Total circuits: `8`
- Total shots: `2048`
- Combined Fisher p-value: `0.572403`
- Sign matches vs IBM Phase 2: `2 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.386719 | 0.367188 | +0.053191 | 0.715331 | +0.086512 | yes |
| 6 | 0.496094 | 0.468750 | +0.058333 | 0.5957 | +0.043469 | yes |
| 10 | 0.441406 | 0.500000 | -0.117188 | 0.215096 | +0.044680 | no |

## Same-Layout Readout Baselines

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.066406 | 0.933594 |
| `0001` | odd | 0.046875 | 0.953125 |

## Claim Boundary

Layout-pinned IQM repeat is controlled diagnostic hardware evidence. It is not sufficient for manuscript claim upgrade until repeated statistics, calibration treatment, and cross-layout consistency are established.
