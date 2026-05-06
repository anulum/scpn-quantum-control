# Phase 3 State/Layout DLA Manifest

Date: 2026-05-07
Backend: `ibm_marrakesh`
Jobs: `d7ts9avljm6s73bd2ej0`, `d7tsdnfljm6s73bd2j70`
Raw-count SHA256: `03068ddaa9794f1ac19614e700887a84dd013cd5af107f49b39c3cff9e5674ac`

## Scope

This analysis uses the committed Phase 3 state/layout raw-count artefact only. It separates state, excitation-count, and physical-layout effects for the `n=4` DLA parity programme.

## Readiness
- Circuits: `495`
- Max transpiled depth: `294`
- Max total gates: `762`

## Comparison Summary
- `excitation_inversion_E0_minus_O3`: mean difference `-0.000524`, signs +/−/0 = `5/7/0`, Fisher p `1.274304e-26`
- `original_E0_minus_O0`: mean difference `0.006312`, signs +/−/0 = `6/6/0`, Fisher p `9.772757e-57`
- `within_even_E0_minus_E1`: mean difference `-0.005872`, signs +/−/0 = `2/10/0`, Fisher p `2.522756e-59`
- `within_odd_O0_minus_O1`: mean difference `0.009191`, signs +/−/0 = `7/5/0`, Fisher p `1.398605e-27`

## Decision Flags
- `layout_spread_exceeds_mean_original_contrast`: `True`
- `original_contrast_mixed_sign`: `True`
- `within_sector_controls_significant`: `True`

## Artefacts
- `data/phase3_state_layout_dla/phase3_state_layout_summary_2026-05-07.json` SHA256 `170c5883574b4ea95adb173187120e94c5d755f072b289301d2a4c2c03c748fa`
- `data/phase3_state_layout_dla/phase3_state_layout_row_metrics_2026-05-07.csv` SHA256 `033556dad102de0703993350d1a101bdf3503e10d5773c4ba8789a9512e9f7df`
- `data/phase3_state_layout_dla/phase3_state_layout_layout_metrics_2026-05-07.csv` SHA256 `6705de9976698d534437f4c82742c10013eb64e055cd2ba9d0dae0ddbaf60a31`
- `data/phase3_state_layout_dla/phase3_state_layout_readout_metrics_2026-05-07.csv` SHA256 `fd16fb4eaa802a2ccb0b302936ef3574570734c66028184108d575bbc7784450`

## Claim Boundary

Supported:
- mechanism-separation evidence for this backend and calibration window
- state-level, depth-level, and layout-level parity-leakage summaries
- whether layout/state spread is comparable to the original contrast

Blocked:
- DLA-parity-only causality
- backend-universal protection
- monotone scaling
- quantum advantage
- full 16-state confusion-matrix mitigation from the five-state readout block

## Reproduction

```bash
python scripts/analyse_phase3_state_layout_dla.py
```
