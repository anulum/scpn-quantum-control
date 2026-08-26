# Experiment mitigation API

Module: `scpn_quantum_control.hardware.experiment_mitigation`

The module exposes six orchestration functions. Each requires an existing
`HardwareRunner` and returns a dictionary. Importing the module has no provider
or filesystem side effect.

## `kuramoto_4osc_zne_experiment`

```python
kuramoto_4osc_zne_experiment(
    runner: HardwareRunner,
    shots: int = 10000,
    dt: float = 0.1,
    scales: list[int] | None = None,
) -> dict[str, Any]
```

Uses default scales `[1, 3, 5]` when `scales` is `None`. Returns the scales,
per-scale order parameters and standard deviations, linear zero-noise estimate,
classical reference, and fit residual.

## `noise_baseline_experiment`

```python
noise_baseline_experiment(
    runner: HardwareRunner,
    shots: int = 10000,
) -> dict[str, Any]
```

Runs the fixed four-qubit near-identity baseline. Returns measured/classical
order parameters and X/Y/Z expectation vectors. Requests a runner-managed save
to `noise_baseline.json`.

## `kuramoto_8osc_zne_experiment`

```python
kuramoto_8osc_zne_experiment(
    runner: HardwareRunner,
    shots: int = 10000,
    dt: float = 0.1,
    scales: list[int] | None = None,
) -> dict[str, Any]
```

Uses the same scale and extrapolation contract as the four-oscillator workflow
with an eight-oscillator circuit and an explicit `n_oscillators` field.

## `upde_16_dd_experiment`

```python
upde_16_dd_experiment(
    runner: HardwareRunner,
    shots: int = 20000,
    trotter_steps: int = 1,
) -> dict[str, Any]
```

Submits raw and dynamical-decoupling X/Y/Z batches. Returns both measured order
parameters, the classical reference, and decoupled expectation vectors.
Requests a runner-managed save to `upde_16_dd.json`.

## `zne_higher_order_experiment`

```python
zne_higher_order_experiment(
    runner: HardwareRunner,
    shots: int = 10000,
    dt: float = 0.1,
    scales: list[int] | None = None,
    poly_order: int = 2,
) -> dict[str, Any]
```

Uses default scales `[1, 3, 5, 7, 9]` and reports a named extrapolation record
for each order from one through `poly_order`.

## `decoherence_scaling_experiment`

```python
decoherence_scaling_experiment(
    runner: HardwareRunner,
    shots: int = 10000,
    qubit_counts: list[int] | None = None,
) -> dict[str, Any]
```

Uses default qubit counts `[2, 4, 6, 8, 10, 12]`. Returns per-count depth and
measured/classical order parameters plus the fitted decay coefficient and
coefficient of determination. Fewer than two positive ratios return `NaN` for
both fit metrics.

## Full autodoc

::: scpn_quantum_control.hardware.experiment_mitigation
    options:
      show_root_heading: false
      show_source: false
      members_order: source
