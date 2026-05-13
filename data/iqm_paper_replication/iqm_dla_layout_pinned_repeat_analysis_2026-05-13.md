# IQM Layout-Pinned DLA Minimal Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_dla_layout_pinned_repeat_2026-05-13_executed.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Requested physical layout: `[9, 4, 3, 8]`
- Total circuits: `8`
- Total shots: `2048`
- Combined Fisher p-value: `0.242579`
- Sign matches vs IBM Phase 2: `1 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.445312 | 0.371094 | +0.200000 | 0.105451 | +0.086512 | yes |
| 6 | 0.449219 | 0.484375 | -0.072581 | 0.478566 | +0.043469 | no |
| 10 | 0.425781 | 0.468750 | -0.091667 | 0.374107 | +0.044680 | no |

## Same-Layout Readout Baselines

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.042969 | 0.957031 |
| `0001` | odd | 0.023438 | 0.976562 |

## Interpretation

The pinned repeat removes the first run automatic-layout confound by requesting the same four physical qubits for all paired circuits. It still does not reproduce a stable IBM-like positive asymmetry across depths: depth 4 matches the IBM sign, while depths 6 and 10 do not. The readout baselines on the same requested layout have low parity leakage relative to the DLA circuits, so readout alone does not explain the high DLA leakage.

## Claim Boundary

Layout-pinned repeated minimal IQM run controls the first run automatic-layout confound, but remains low-statistics and mixed-sign; do not upgrade manuscript claims without additional repeated statistics and mitigation analysis.
