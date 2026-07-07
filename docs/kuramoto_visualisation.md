# Kuramoto Visualisation Layer

`scpn_quantum_control.accel.kuramoto_visualisation` (re-exported from the
`scpn_quantum_control.kuramoto` facade under the `visualisation` capability group) renders the
standard diagnostics for coupled-phase-oscillator trajectories directly from the shipped trajectory
and observable surfaces. Four renderers are provided:

| Renderer | Input | What it shows |
| --- | --- | --- |
| `phase_raster` | `(T, N)` trajectory | The phase of every oscillator over time as an oscillator-by-time raster with a cyclic colour map — travelling waves, locking and incoherence at a glance. |
| `order_parameter_timeseries` | `(T, N)` trajectory | The Kuramoto order parameter `r(t)`, read from the same complex mean field as `order_parameter`. |
| `chimera_snapshot` | `(T, N)` trajectory + community partition | The per-community order parameters over time (`community_order_parameters`), the title annotated with the `chimera_index`. |
| `network_phase_embedding` | `(N,)` phase snapshot + `(N, N)` coupling | A single-time phase snapshot on a circular node layout with the coupling drawn as edges. |

## Optional dependency

Matplotlib is an optional dependency, declared as the `viz` extra:

```bash
pip install 'scpn-quantum-control[viz]'
```

It is imported lazily, so importing the `accel` or `kuramoto` facade never requires matplotlib;
calling a renderer without it raises an `ImportError` naming the install command. The renderers do not
compute anything of their own — they reuse `order_parameter`, `community_order_parameters` and
`chimera_index`, so the curves and annotations are exactly the shipped diagnostics.

## Usage

Every renderer draws into a caller-supplied axis when given (so panels compose into one figure) or
creates its own otherwise, returns the axis, and never calls `show` — the caller owns display and
saving.

```python
import numpy as np
import matplotlib.pyplot as plt

from scpn_quantum_control.kuramoto import (
    kuramoto_rk4_trajectory,
    phase_raster,
    order_parameter_timeseries,
    chimera_snapshot,
    network_phase_embedding,
)

rng = np.random.default_rng(0)
count = 24
coupling = np.full((count, count), 0.8)
np.fill_diagonal(coupling, 0.0)
theta0 = rng.uniform(-np.pi, np.pi, count)
omega = rng.normal(0.0, 1.0, count)
phases = kuramoto_rk4_trajectory(theta0, omega, coupling, dt=0.02, n_steps=400)  # (401, 24)

figure, axes = plt.subplots(2, 2, figsize=(11, 8))
phase_raster(phases, ax=axes[0, 0])
order_parameter_timeseries(phases, ax=axes[0, 1])
chimera_snapshot(phases, [np.arange(0, 12), np.arange(12, 24)], ax=axes[1, 0])
network_phase_embedding(phases[-1], coupling, ax=axes[1, 1])
figure.tight_layout()
figure.savefig("kuramoto_dashboard.png", dpi=150)
```

The `times` keyword sets the horizontal axis for the trajectory renderers; when omitted the integer
step index is used.
