# Multimodal Forecasting Under Partial Observation

This page defines the bounded multimodal forecasting product. It provides
immutable multimodal custody, a deterministic missingness-aware classical
baseline, partial-observation scoring, empirical split residual intervals, and
explicit composition ports into no-submit active sensing and unapplied
controller proposals.

## Production surfaces

The public `scpn_quantum_control.forecasting` facade exports:

- `MultimodalObservationBatch` for phase histories, graphs, exogenous events,
  masks, targets, frequencies, sample identifiers, split custody, and
  simulation-only domain tags;
- `generate_synthetic_multimodal_dataset(...)` for deterministic, independent
  train, calibration, and test Kuramoto trajectories;
- `fit_multimodal_ridge_forecaster(...)` for a linear reference model whose
  imputation means and scales come only from training custody;
- `evaluate_partial_observation_batch(...)` for observed wrapped-phase error
  plus an exact known-simulator Kuramoto forward residual;
- `fit_residual_interval_calibrator(...)` and
  `certify_interval_coverage(...)` for sample-max split residual intervals over
  independent synthetic trajectory rows;
- `plan_forecast_active_sensing(...)` for an interval-width proxy entering the
  existing no-submit active-sensing planner; and
- `forecast_to_controller_initialisation(...)` for a clipped, unapplied
  existing `ControllerProposal`.

The four allowed tags—`synthetic`, `grid_like_sim`, `eeg_like_sim`, and
`plasma_like_sim`—identify stylised generator configurations. They do not
identify real datasets or establish domain fidelity.

## Minimal deterministic workflow

```python
import numpy as np

from scpn_quantum_control.forecasting import (
    SyntheticMultimodalConfig,
    apply_residual_interval,
    certify_interval_coverage,
    evaluate_partial_observation_batch,
    evaluate_point_forecast,
    fit_multimodal_ridge_forecaster,
    fit_residual_interval_calibrator,
    generate_synthetic_multimodal_dataset,
)

dataset = generate_synthetic_multimodal_dataset(
    SyntheticMultimodalConfig(
        train_samples=64,
        calibration_samples=24,
        test_samples=32,
        seed=3701,
    )
)
model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)

calibration_forecast = model.predict(dataset.calibration)
calibrator = fit_residual_interval_calibrator(
    model,
    calibration_forecast,
    dataset.calibration,
    alpha=0.10,
)

test_forecast = model.predict(dataset.test)
accuracy = evaluate_point_forecast(test_forecast, dataset.test)
interval = apply_residual_interval(calibrator, test_forecast)
coverage = certify_interval_coverage(model, calibrator, interval, dataset.test)

partial_mask = np.zeros_like(dataset.test.target_mask)
partial_mask[:, :, ::2] = True
partial = evaluate_partial_observation_batch(
    test_forecast,
    dataset.test,
    partial_mask,
)
```

All batches and fitted arrays are copied and made read-only. Missing inputs are
normalised to `NaN` behind explicit masks. Split identifiers and SHA-256 content
digests make train/calibration/test leakage checks executable.

## Frozen evidence

The committed evidence uses 64 training, 24 calibration, and 32 test
trajectories, with 16/6/8 independent rows per synthetic tag. Input phase and
event entries are randomly masked; exact simulator couplings remain fully known
because the physics-residual certificate does not infer them.

| Held-out metric | Observed |
|---|---:|
| Test wrapped MSE | 0.000823098535 |
| Persistence wrapped MSE | 0.00113867904 |
| Partial-mask fraction | 0.5 |
| Partial observed wrapped RMSE | 0.0254875208 |
| Exact-simulator Kuramoto residual RMSE | 0.210377396 |
| Split residual radius (`alpha=0.1`) | 0.0961529067 |
| Empirical test sample coverage | 0.90625 |
| Empirical test value coverage | 0.9921875 |

The test MSE is lower than persistence overall and for each of the four frozen
test-tag groups. On calibration, the overall MSE is lower than persistence, but
the `synthetic` subgroup is not. These finite synthetic rows are regression
evidence for this exact configuration, not a general superiority result.

The nominal interval target is 0.9 and the frozen sample coverage is 0.90625.
This is an empirical held-out result over independently generated trajectory
rows. It is not a conditional-coverage, sequential EnbPI, or domain-transfer
guarantee.

Regenerate and byte-check the evidence with:

```bash
PYTHONPATH=src python scripts/run_multimodal_forecasting_evidence.py
PYTHONPATH=src python scripts/run_multimodal_forecasting_evidence.py --check
```

Committed custody:

- `data/multimodal_forecasting/multimodal_forecasting_evidence.json`
- `data/multimodal_forecasting/multimodal_forecasting_evidence.md`
- content digest
  `2d770f7d5a23b27d06717ba45fa925981c993d1b0feea20578e6b60ab52d199d`

## Composition boundaries

`plan_forecast_active_sensing(...)` converts per-node interval widths into
`InformationGainCandidate` records and calls the existing active-sensing planner. The
frozen evidence produces an allowed local dry-run plan with
`hardware_execution=False`, `would_submit=False`, and no provider cost claim.
Passing `request_hardware=True` remains refused by the active-sensing policy.

`forecast_to_controller_initialisation(...)` maps the terminal forecast order
parameter into a clipped existing `ControllerProposal`. The returned record is
always `applied=False` and `safety_decision=False`. It does not establish
closed-loop stability, control performance, or operational suitability.

## Scientific basis

- [Cao et al. (2018), BRITS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html)
  motivates preserving explicit missingness in multivariate time-series
  models. This bounded implementation does not implement or claim BRITS.
- [Xu and Xie (2023)](https://doi.org/10.1109/TPAMI.2023.3272339) motivates
  careful uncertainty claims for time series. This product uses independent synthetic
  trajectory rows and does not claim their sequential method.
- [Smith and Gottwald (2023)](https://arxiv.org/abs/2309.03545) treats dynamics
  learning under partial observation as a data-assimilation problem. This product does
  not claim their ensemble Kalman parameter-inference result.
- [Dörfler, Chertkov, and Bullo (2013)](https://doi.org/10.1073/pnas.1212134110)
  supplies oscillator-network context for the grid-like simulation tag. It
  does not validate an operational power-grid model here.

These sources constrain the architecture and claim language. They do not
validate this repository's thresholds, generated trajectories, forecasts, or
deployment suitability.

## Explicit exclusions

- No real EEG, clinical, grid, SCADA, plasma diagnostic, or plant data is in
  this product's custody.
- No hidden-state reconstruction, coupling inference, data assimilation, or
  arbitrary partial-observation solution is claimed.
- No hardware, provider, QPU, adaptive sensing, clinical, safety, stability,
  control-performance, generalisation, advantage, publication, or deployment
  claim is made.
- The deterministic ridge model is a classical reference baseline, not a
  production forecasting service.

## Related surfaces

- [Real-data synchronisation forecasting](real_data_sync_forecasting.md)
- [Active sensing product](active_sensing_product.md)
- [Quantum-classical co-design loop](quantum_classical_codesign_loop.md)
- [Differentiable API](differentiable_api.md)
