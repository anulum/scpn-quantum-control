# Experiment mitigation orchestration

`scpn_quantum_control.hardware.experiment_mitigation` defines six hardware-runner
workflows for zero-noise extrapolation, calibration drift, dynamical
decoupling, and decoherence scaling. These functions orchestrate circuit
construction, submission, result parsing, and classical comparison. They do
not authenticate a runner or choose a backend.

## Execution boundary

Every function receives an already configured `HardwareRunner`. Calls may
submit sampler work and therefore inherit the runner's credential, approval,
provider, cost, timeout, and backend policies. Importing the module performs no
submission. The functions print progress to standard output; the noise
baseline and dynamical-decoupling workflows also ask the runner to save one
result file.

Tests use deterministic boundary doubles to verify orchestration without
claiming provider or device evidence. A passing local test proves result-schema
and routing behaviour only. It does not validate hardware noise reduction,
calibration stability, or a device-specific extrapolation model.

## Zero-noise extrapolation

`kuramoto_4osc_zne_experiment` and `kuramoto_8osc_zne_experiment` build a fixed
Kuramoto evolution, fold the circuit at each requested scale, submit X/Y/Z
measurement batches, and apply a first-order extrapolation. Their default
scales are `[1, 3, 5]`; an explicit list is preserved exactly.

`zne_higher_order_experiment` extends the default scale list to
`[1, 3, 5, 7, 9]` and reports every polynomial order from one through
`poly_order`. Its outputs are fits to the measured points, not proof that the
zero-noise limit is physically accurate.

## Calibration and decoupling

`noise_baseline_experiment` submits a four-qubit near-identity circuit and
returns measured and classical order parameters plus per-qubit expectations.
It saves the first sampler result as `noise_baseline.json` through the runner.

`upde_16_dd_experiment` submits raw and dynamical-decoupling variants of the
same 16-layer evolution. It reports both order parameters and the decoupled
expectation vectors, then saves the raw sampler result as `upde_16_dd.json`.
The returned comparison does not establish that decoupling improves every
backend or calibration window.

## Decoherence scaling

`decoherence_scaling_experiment` evaluates a caller-supplied qubit-count list,
or `[2, 4, 6, 8, 10, 12]` by default. It records transpiled depth and measured
versus classical order parameters, then fits
`R_hw / R_classical = exp(-gamma * depth)` over positive ratios. Fewer than two
valid points produce `NaN` fit metrics. The fit is a bounded diagnostic for the
submitted circuits, not a universal per-gate error rate.

## Result handling

All functions return JSON-oriented dictionaries. NumPy-derived scalar values
remain numeric, and expectation arrays are converted to lists where exposed.
Callers must retain runner provenance, calibration, job identifiers, and raw
counts when using these summaries as evidence. Do not publish or compare a
result without the runner's associated custody record.

For exact signatures and return fields, see the
[Experiment mitigation API](api/experiment_mitigation.md).
