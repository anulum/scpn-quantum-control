# S3 Design Surrogate Rehearsal

Protocol ID: `s3_ml_augmented_pulse_ansatz_design_2026-05-06`

Submission state: no hardware submission; deterministic no-QPU surrogate rehearsal.

## Dataset
- rows: 84
- sizes: [3, 4, 5]

## Surrogate
- model: closed_form_ridge_linear_surrogate
- train rows: 67
- holdout rows: 17
- holdout metrics: {"mae": 0.009179707010294096, "r2": 0.9999999903608034, "rmse": 0.011105732775517062}

## Claim Boundary
This is a deterministic no-QPU surrogate rehearsal over proxy scores. It does not demonstrate pulse-level hardware improvement, VQE improvement, or quantum advantage.
