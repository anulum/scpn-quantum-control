<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# oscillatools

A differentiable, control-oriented toolkit for coupled-phase-oscillator
(Kuramoto) dynamics. It provides the model family, integrators, exact mean-field
reductions, stability and continuation analysis, differentiation, and control
primitives as a light, standalone distribution whose only hard dependencies are
**NumPy** and **SciPy**. Every acceleration or interoperability tier — a Rust
engine, Julia, JAX, PyTorch, matplotlib, scikit-learn — is an optional extra; the
pure-Python NumPy floor always runs.

`oscillatools` is extracted from
[`scpn-quantum-control`](https://github.com/anulum/scpn-quantum-control), where the
toolkit originated, so that the coupled-oscillator surface can be installed without
that project's quantum-provider dependency stack.

## Installation

```bash
pip install oscillatools                 # NumPy + SciPy floor
pip install "oscillatools[rust]"         # optional Rust acceleration engine
pip install "oscillatools[jax]"          # JAX autodiff backend
pip install "oscillatools[torch]"        # PyTorch neural-operator surrogate
pip install "oscillatools[sklearn]"      # scikit-learn estimator interface
pip install "oscillatools[viz]"          # matplotlib renderers
pip install "oscillatools[julia]"        # Julia acceleration tier
```

## First path

```python
import numpy as np

import oscillatools as kuramoto

theta0 = np.array([0.0, 0.7, 1.6, 2.9], dtype=np.float64)
omega = np.array([0.1, -0.2, 0.15, 0.05], dtype=np.float64)
coupling = np.array(
    [
        [0.0, 0.6, 0.0, 0.2],
        [0.6, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.4],
        [0.2, 0.0, 0.4, 0.0],
    ],
    dtype=np.float64,
)

trajectory = kuramoto.kuramoto_rk4_trajectory(theta0, omega, coupling, 0.01, 64)
diagnostics = kuramoto.frequency_order_diagnostics(trajectory, dt=0.01)
value, gradient = kuramoto.synchronisation_value_and_grad(theta0, omega, coupling)
```

Call `kuramoto.capabilities()` to inspect the grouped public API and
`kuramoto.describe("analysis")` to list one group programmatically.

## Documentation map

- **[Handbook](handbook.md)** — the generated reference for the model families,
  integrators and adjoints, observables, and analysis and control surface.
- **[Capability snapshot](capabilities.md)** — the exact, diffable inventory of
  every public group and symbol at the current version.
- **[Gradient coverage matrix](gradient_coverage_matrix.md)** — the generated
  inventory of public gradient, Hessian, Jacobian, adjoint, and sensitivity
  surfaces derived from the same facade map.
- **[Example gallery](gallery.md)** — runnable, deterministic worked workflows.
- **[Multi-language tier benchmark](tier_benchmarks.md)** — the Rust/Julia/Python
  per-primitive latency evidence with full provenance.
- **[Competitive benchmark](competitive_benchmark.md)** — how the toolkit is
  compared against external solvers, and the boundary that comparison keeps.

## Licensing

`oscillatools` is dual-licensed under AGPL-3.0-or-later with a commercial license
available. For commercial licensing: protoscience@anulum.li.
