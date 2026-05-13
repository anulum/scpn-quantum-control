# IQM Partial Layout Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_partial_layout_repeat_calibration_repeat_20260513T1735Z_q2-7-12-13_executed.json`
- Layout: `[2, 7, 12, 13]`
- Label: `calibration_repeat_20260513T1735Z`
- Total circuits: `5`
- Total shots: `1280`
- Combined Fisher p-value over matched depths: `0.608758`
- Sign matches vs IBM Phase 2: `2 / 2`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.332031 | 0.308594 | +0.075949 | 0.635869 | +0.086512 | yes |
| 6 | 0.378906 | 0.339844 | +0.114943 | 0.407163 | +0.043469 | yes |

## Readout Control

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.117188 | 0.875000 |

## Claim Boundary

Partial calibration-time repeat over d4/d6 plus a readout control. It is useful for drift diagnostics only and is not a complete layout block or manuscript claim upgrade.
