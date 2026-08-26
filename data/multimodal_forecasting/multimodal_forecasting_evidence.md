# Multimodal Forecasting Evidence

Schema: `scpn.multimodal_forecasting.v1`
Content digest: `2d770f7d5a23b27d06717ba45fa925981c993d1b0feea20578e6b60ab52d199d`

## Custody and held-out point forecast

- Dataset digest: `8c6ee2912ad293f3deddc21f684b7241d400d4083a1bf789cf8364a7c1e1d196`; train / calibration / test samples: `64` / `24` / `32`.
- Model digest: `a19e4d14a1a386a82c0453bb13c1967d601706de4a2875481175f4d5f0899fe3`; test wrapped MSE `0.000823098535` versus persistence `0.00113867904`; lower MSE: `True`.

| Synthetic tag | Samples | Forecast MSE | Persistence MSE | Lower MSE |
|---|---:|---:|---:|---|
| `synthetic` | 8 | 0.000470128586 | 0.000878043562 | `True` |
| `grid_like_sim` | 8 | 0.000950616802 | 0.00107599328 | `True` |
| `eeg_like_sim` | 8 | 0.00110907622 | 0.00175816927 | `True` |
| `plasma_like_sim` | 8 | 0.000762572526 | 0.000842510037 | `True` |

## Partial observation and uncertainty

- Partial target fraction: `0.5`; observed wrapped RMSE `0.0254875208`; exact-simulator Kuramoto residual RMSE `0.210377396`.
- Split residual radius: `0.0961529067` at target coverage `0.9`; empirical sample coverage `0.90625` and value coverage `0.9921875`.

## Composition ports

- Active-sensing plan allowed: `True`; hardware execution: `False`.
- Controller proposal applied: `False`; safety decision: `False`.

## Support matrix

| Surface | Status | Evidence / boundary |
|---|---|---|
| `synthetic_multimodal_schema` | `synthetic_supported` | Immutable series, graph, event, mask, target, tag, and split custody. All four tags identify stylised simulator configurations only. |
| `missingness_aware_ridge` | `synthetic_supported` | Training-only imputation/scaling and held-out persistence comparison. Linear reference baseline, not BRITS or a production forecaster. |
| `partial_observation_objective` | `synthetic_supported` | Observed wrapped error plus exact Kuramoto forward residual. Known simulator coupling; no hidden-state or parameter inference. |
| `split_residual_intervals` | `synthetic_supported` | Independent calibration rows and empirical held-out test coverage. Not sequential EnbPI, conditional coverage, or domain transfer. |
| `active_sensing_bridge` | `bounded_supported` | Interval-width proxies enter the existing no-submit sensing planner. Not adaptive hardware sensing or optimal sensor placement. |
| `codesign_controller_initialisation` | `bounded_supported` | Terminal forecast creates a clipped existing ControllerProposal. Proposal remains unapplied and is not a safety decision. |
| `real_eeg_clinical_data` | `blocked_dependency` | No governed real EEG or clinical dataset is in custody. The eeg_like_sim tag provides no clinical or neuroscience validity. |
| `real_grid_scada_data` | `blocked_dependency` | No governed grid or SCADA dataset is in custody. The grid_like_sim tag is not a power-system operational model. |
| `real_plasma_plant_data` | `blocked_dependency` | No governed plasma diagnostic or plant dataset is in custody. The plasma_like_sim tag provides no reactor or plant evidence. |
| `hardware_qpu_execution` | `blocked_dependency` | No hardware or provider request is made by this evidence runner. Local deterministic simulation only; no QPU, provider, or spend. |

## Claim boundary

Deterministic synthetic Kuramoto trajectory evidence under explicit simulation-only domain tags. No real EEG, clinical, grid, SCADA, plasma, plant, hardware, QPU, state-estimation, control-performance, safety, deployment, or publication claim.
