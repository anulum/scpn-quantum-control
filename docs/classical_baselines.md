# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Classical Baselines

# Classical Baselines

This page records the supported classical baseline surfaces for Kuramoto-XY
workflows. Each baseline returns a `ClassicalBaselineRun` envelope with the
backend name, availability flag, elapsed wall time, time grid, order-parameter
trajectory, and metadata needed for provenance.

## Baseline Matrix

| Baseline | Dependency | Purpose | Availability behaviour |
| --- | --- | --- | --- |
| SciPy ODE | Runtime dependency | Classical Kuramoto phase dynamics via `solve_ivp(RK45)`. | Always available. |
| QuTiP Lindblad | `[opensys]` or `[xvalidate]` extra | Independent density-matrix open-system reference via `qutip.mesolve`. | Returns `available=False` when QuTiP is absent. |
| MPS TEBD | `[tensor]` extra | Tensor-network time evolution through the existing quimb TEBD backend. | Returns `available=False` when quimb is absent. |

## Quick Use

```python
import numpy as np

from scpn_quantum_control.benchmarks.classical_baselines import (
    available_baselines,
    run_documented_classical_baselines,
)

K = np.array(
    [
        [0.0, 0.4, 0.0],
        [0.4, 0.0, 0.3],
        [0.0, 0.3, 0.0],
    ]
)
omega = np.array([0.8, 1.0, 1.2])

print(available_baselines())
runs = run_documented_classical_baselines(K, omega, t_max=0.5, dt=0.1)

for name, run in runs.items():
    if run.available:
        print(name, run.backend, run.r_final)
    else:
        print(name, run.unavailable_reason)
```

## SciPy ODE

`scipy_ode_baseline` integrates the classical Kuramoto equations:

```text
d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i)
```

This is the baseline for classical phase locking. It is not a quantum
Hamiltonian simulation and should be labelled as a classical ODE reference in
reports.

## QuTiP Lindblad

`qutip_lindblad_baseline` builds an independent QuTiP XY Hamiltonian and evolves
the initial product state under amplitude-damping collapse operators. Use it for
small open-system cross-checks where density-matrix scaling is acceptable.

The function does not fabricate a result when QuTiP is missing. It returns a
`ClassicalBaselineRun` with `available=False` and an unavailable reason of
`qutip missing`.

## MPS TEBD

`mps_tebd_baseline` wraps `phase.mps_evolution.tebd_evolution`. The quimb local
Hamiltonian path uses nearest-neighbour terms, matching the existing MPS module
contract. Use it to document whether a tensor-network baseline is available and
what bond dimensions the run reached.

The function returns `available=False` with `unavailable_reason="quimb missing"`
when the `[tensor]` extra is absent.
