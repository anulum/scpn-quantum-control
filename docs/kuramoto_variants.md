# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto variants

# Higher-Order, Monitored, and PT-Symmetric Kuramoto Variants

`scpn_quantum_control.phase.kuramoto_variants` formalises three Kuramoto
extensions that previously only existed as campaign-level experiments:

- higher-order simplicial coupling through anchored triadic terms,
- monitored order-parameter feedback,
- balanced gain/loss PT-symmetric complex oscillators.

Each variant has a validated Python specification, a NumPy reference path, and a
Rust PyO3 trajectory kernel exposed through `scpn_quantum_engine`.

## Equations

The shared pairwise term is the classical Kuramoto velocity

\[
\dot\theta_i = \omega_i + \sum_j K_{ij}\sin(\theta_j-\theta_i).
\]

The higher-order variant adds anchored 2-simplex terms:

\[
\dot\theta_i \leftarrow \dot\theta_i +
\sum_{(i,j,k)\in E_3} B_{ijk}\sin(\theta_j+\theta_k-2\theta_i).
\]

The monitored variant computes the instantaneous order parameter
\(R e^{i\psi}=N^{-1}\sum_i e^{i\theta_i}\), applies a deterministic readout
\(R_m=(1-s)R+sR_\star\), and adds feedback

\[
\dot\theta_i \leftarrow \dot\theta_i +
g(R_\star-R_m)\sin(\psi-\theta_i).
\]

The PT-symmetric variant evolves complex amplitudes
\(z_i=e^{i\theta_i}\) with balanced gain/loss \(\sum_i \gamma_i=0\):

\[
\dot z_i = \left(\gamma_i + i\dot\theta_i\right) z_i.
\]

The implementation renormalises the complex vector after each Euler step and
reports both \(R(t)\) and the gain/loss diagnostics, so synchronisation and
non-Hermitian imbalance are visible separately.

## API

```python
import numpy as np

from scpn_quantum_control.phase import (
    HigherOrderKuramotoSpec,
    MonitoredKuramotoSpec,
    PTSymmetricKuramotoSpec,
    build_triadic_ring_terms,
    simulate_higher_order_kuramoto,
    simulate_monitored_kuramoto,
    simulate_pt_symmetric_kuramoto,
)

K = np.array(
    [
        [0.0, 0.45, 0.0, 0.45],
        [0.45, 0.0, 0.45, 0.0],
        [0.0, 0.45, 0.0, 0.45],
        [0.45, 0.0, 0.45, 0.0],
    ],
    dtype=np.float64,
)
omega = np.array([0.0, 0.6, 1.2, 2.4], dtype=np.float64)
theta0 = np.array([0.0, 0.8, 2.0, 4.2], dtype=np.float64)

edges, weights = build_triadic_ring_terms(4, weight=0.25)
higher = simulate_higher_order_kuramoto(
    HigherOrderKuramotoSpec(K, omega, edges, weights, theta0=theta0),
    dt=0.02,
    n_steps=64,
)

monitored = simulate_monitored_kuramoto(
    MonitoredKuramotoSpec(
        K,
        omega,
        target_r=0.85,
        monitor_gain=1.1,
        measurement_strength=0.25,
        theta0=theta0,
    ),
    dt=0.02,
    n_steps=64,
)

pt = simulate_pt_symmetric_kuramoto(
    PTSymmetricKuramotoSpec(
        K,
        omega,
        gain_loss=np.array([0.08, -0.08, 0.04, -0.04], dtype=np.float64),
        theta0=theta0,
    ),
    dt=0.02,
    n_steps=64,
)
```

The stable facade also exposes `simulate_variant_trajectory(problem, variant, ...)`
for callers that already use `KuramotoProblem`.

## Swarmalators

`scpn_quantum_control.kuramoto` also exposes the Swarmalator model family:
`swarmalator_field`, `integrate_swarmalators`, and
`swarmalator_order_parameters`. This is the O'Keeffe-Hong-Strogatz
space-phase extension of Kuramoto dynamics: each oscillator carries a planar
position and a phase, phase similarity modulates spatial attraction, and
spatial distance modulates phase synchronisation.

The shipped surface is a deterministic NumPy reference path with fail-closed
validation for finite, pairwise-distinct positions. It reports both the ordinary
phase coherence and the rainbow order parameters `S_+` and `S_-`, so static
synchrony, static phase waves, and static asynchrony are distinguishable without
claiming a phase-only reduction. This slice does not add a Rust kernel or a
performance benchmark row; any future accelerated route must ship with parity
tests and benchmark provenance before it is documented as a faster tier.

```python
import numpy as np

from scpn_quantum_control.kuramoto import (
    integrate_swarmalators,
    swarmalator_order_parameters,
)

positions = np.array(
    [[-0.5, -0.2], [0.2, -0.4], [0.4, 0.3], [-0.3, 0.5]],
    dtype=np.float64,
)
phases = np.array([0.0, 1.1, 2.3, 4.0], dtype=np.float64)

trajectory = integrate_swarmalators(
    positions,
    phases,
    coupling_phase=1.0,
    coupling_space=0.0,
    dt=0.05,
    n_steps=200,
)
order = swarmalator_order_parameters(
    trajectory.terminal_positions,
    trajectory.terminal_phases,
)
```

## Rust Kernels

| Variant | Rust function | Returned diagnostics |
|---------|---------------|----------------------|
| Higher-order | `higher_order_kuramoto_trajectory` | `times`, `R(t)` |
| Monitored | `monitored_kuramoto_trajectory` | `times`, `R(t)`, readout \(R_m(t)\), feedback gain |
| PT-symmetric | `pt_symmetric_kuramoto_trajectory` | `times`, `R(t)`, PT norm, gain/loss imbalance |

The NumPy path uses the same equations and is covered by parity tests against the
preferred Rust path when the extension is installed.

## Validation

`tests/test_kuramoto_variants.py` checks:

- periodic triadic ring construction,
- Rust/NumPy trajectory parity,
- higher-order terms changing the pairwise trajectory,
- monitored readout and feedback channels,
- zero-gain monitored feedback,
- balanced PT gain/loss norm and imbalance,
- stable-facade dispatch,
- defensive copies and read-only arrays,
- invalid shape/range rejection before simulation.

## Benchmark

See [Pipeline Performance](pipeline_performance.md) for the measured command
provenance. On the ASRock H510 Pro BTC+ / i5-11600K / Ubuntu 24.04.4 machine,
the three 4-oscillator, 64-step variant trajectories run in 2.205 ms through
the Rust PyO3 kernels.
