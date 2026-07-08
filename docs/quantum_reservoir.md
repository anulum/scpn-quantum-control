# Quantum Reservoir Computing

This page defines the named quantum reservoir computing (QRC) surface in this
repository and its classical baseline boundary.

## Production Surfaces

The QRC surface is split across two bounded modules:

- `scpn_quantum_control.applications.quantum_reservoir` maps classical inputs
  through Kuramoto-XY Hamiltonian evolution and Pauli expectation features, then
  fits a ridge readout.
- `scpn_quantum_control.analysis.qrc_phase_detector` uses exact dense
  ground-state Pauli features as a small-system phase-detector reference.

QWC-4.3 adds the named comparison surface:

```python
import numpy as np

from scpn_quantum_control.applications import compare_quantum_reservoir_to_esn
from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

rng = np.random.default_rng(42)
X_train = rng.uniform(size=(10, 3))
y_train = np.sin(2.0 * X_train[:, 0]) + 0.25 * X_train[:, 1]
K = build_knm_paper27(L=3)

comparison = compare_quantum_reservoir_to_esn(
    X_train,
    y_train,
    K,
    alpha=0.5,
    max_weight=1,
    seed=5,
)
```

By default the classical echo-state network (ESN) uses the same number of
features as the QRC feature map. Callers can set `reservoir_size` explicitly
when they need a deliberately unmatched-capacity comparison.

## Classical Baseline

`classical_esn_feature_matrix` implements a deterministic ESN reference:

- seeded input and recurrent weights;
- recurrent matrix rescaled to the requested spectral radius;
- leaky state update;
- ridge readout through `classical_esn_ridge_regression`.

The ESN exists to keep QRC claims honest. A QRC result is not promoted by this
repository unless it is compared against the classical baseline at a stated
feature count and task.

## Functional Evidence

Local non-isolated smoke evidence on 2026-07-08, run on the workstation while
other repository work was active:

| Operation | System | Time | Output |
|-----------|--------|------|--------|
| `compare_quantum_reservoir_to_esn(10 samples)` | 3 qubits, 9/9 matched features | 48.5 ms | QRC MSE=0.018606, ESN MSE=0.037680 |

This is functional evidence for wiring and bounded task behaviour. It is not an
isolated-core production benchmark.

## Explicit Boundaries

- The QRC feature map is exact-statevector and small-system bounded.
- The phase detector is an exact dense reference, not a scalable reservoir
  simulator.
- The ESN baseline is a deterministic NumPy reference comparator, not an
  accelerated service path.
- No hardware-QRC advantage claim is made here.
- No broad time-series benchmark claim is made until a preregistered benchmark
  beats the ESN baseline at matched task and feature count.

## Related Surfaces

- [`Analysis API`](analysis_api.md)
- [`API Overview`](api.md#applications)
- [`Pipeline Performance`](pipeline_performance.md#quantum-reservoir-computing)
