# IQM Layout-Pinned DLA Minimal Repeat Analysis

- Source: `data/iqm_paper_replication/iqm_dla_layout_pinned_repeat_q2-7-12-13_2026-05-13_executed.json`
- IBM reference: `data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json`
- Requested physical layout: `[2, 7, 12, 13]`
- Total circuits: `8`
- Total shots: `2048`
- Combined Fisher p-value: `0.372907`
- Sign matches vs IBM Phase 2: `3 / 3`

| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.308594 | 0.292969 | +0.053333 | 0.772561 | +0.086512 | yes |
| 6 | 0.355469 | 0.316406 | +0.123457 | 0.399756 | +0.043469 | yes |
| 10 | 0.453125 | 0.382812 | +0.183673 | 0.127608 | +0.044680 | yes |

## Same-Layout Readout Baselines

| Initial | Sector | Parity leakage | Initial-state retention |
|---|---|---:|---:|
| `0011` | even | 0.085938 | 0.914062 |
| `0001` | odd | 0.035156 | 0.964844 |

## Claim Boundary

Layout-pinned IQM repeat is controlled diagnostic hardware evidence. It is not sufficient for manuscript claim upgrade until repeated statistics, calibration treatment, and cross-layout consistency are established.
