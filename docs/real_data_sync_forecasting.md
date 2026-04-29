# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Real-Data Synchronisation Forecasting

# Real-Data Synchronisation Forecasting

The forecasting benchmark evaluates whether an early observed
synchronisation window can calibrate a physical Kuramoto forecast without
seeing the held-out samples. It is intentionally small and replayable:
the default dataset is the committed IBM Heron r2 four-oscillator
hardware trace in `results/hw_kuramoto_4osc.json`.

## Public API

```python
from scpn_quantum_control.forecasting import (
    load_hardware_kuramoto_4osc_trace,
    run_real_data_sync_forecast_benchmark,
)

dataset = load_hardware_kuramoto_4osc_trace()
result = run_real_data_sync_forecast_benchmark(dataset)

print(result.summary)
print(result.baseline.holdout_mse)
print(result.calibrated.holdout_mse)
```

The result records:

- the dataset source path, domain, source kind, and oscillator count;
- the train and hold-out window boundaries;
- baseline predictions and calibrated predictions;
- held-out MSE, held-out MAE, and pass/fail status;
- backend provenance for the physical baseline.

## Benchmark Definition

For a dataset with observed order parameter $R_\mathrm{obs}(t)$ and a
physical baseline $R_\mathrm{base}(t)$, the benchmark fits only the first
`train_size` samples:

$$
R_\mathrm{cal}(t) = a R_\mathrm{base}(t) + b.
$$

The affine coefficients are fitted with `numpy.linalg.lstsq` on the
training window. The held-out window is never visible to the fit. The
default pass criterion is a 10% reduction in held-out mean-squared error:

$$
\frac{\mathrm{MSE}_\mathrm{base} - \mathrm{MSE}_\mathrm{cal}}
     {\mathrm{MSE}_\mathrm{base}} \ge 0.10.
$$

## Data Sources

| Dataset | Source kind | Target trace | Physical baseline |
|---------|-------------|--------------|-------------------|
| `ibm_heron_r2_kuramoto_4osc` | QPU hardware measurement | `hw_R` values from `results/hw_kuramoto_4osc.json` | `exact_R` values stored in the same campaign file |
| `ieee5bus_frequency_disturbance` | Public topology classical replay | DOP853 replay of IEEE 5-bus coupling with deterministic rotor disturbance | Rust `kuramoto_trajectory` when available, Python Euler fallback otherwise |

The IEEE 5-bus case is not a raw grid-frequency measurement. It is a
source-backed topology replay built from the public IEEE 5-bus constants
already exposed by `applications.power_grid`. The hardware trace is the
default real observed synchronisation dataset.

## CLI

```bash
.venv-linux/bin/python scripts/run_real_data_sync_forecast_benchmark.py
.venv-linux/bin/python scripts/run_real_data_sync_forecast_benchmark.py --hardware-only
```

The CLI prints JSON with plain Python values, suitable for release notes,
regression storage outside `results/`, or a future Zenodo benchmark
deposit.

## Failure Criteria

The benchmark fails when any of these conditions holds:

- fewer than three samples are available;
- `train_size` leaves no held-out sample;
- observed or baseline synchronisation values leave the interval `[0, 1]`;
- the calibrated forecast does not reduce held-out MSE by the configured
  threshold;
- a generated physical baseline cannot state its backend.

## Pipeline Position

This module sits after the source-data bridge and before application-level
control:

```text
hardware trace or source topology
        -> coupling + observed R(t)
        -> Rust/Python Kuramoto baseline
        -> train-window calibration
        -> held-out forecast metrics
```

It does not claim that a calibrated affine correction is a final hybrid
forecasting engine. It provides a reproducible, falsifiable benchmark
surface for the next forecasting work: richer correction bundles, QPU
snapshots, and domain-specific train/test registries.
