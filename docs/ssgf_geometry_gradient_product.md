# SSGF quantum-in-the-loop geometry gradient

`scpn_quantum_control.ssgf_geometry_gradient_product` governs the existing
SSGF simulator path

\[
z \rightarrow W(z) \rightarrow H(W) \rightarrow |\psi(t)\rangle
\rightarrow R \rightarrow C=1-R \rightarrow \nabla_z C.
\]

The product is a bounded local-simulation evidence surface. It does not submit
jobs, claim analytic automatic differentiation, or promote a short outer-cycle
trace to convergence or advantage evidence.

## Gradient-method policy

| Governed route | Status | Reason |
|---|---|---|
| `transform:ssgf.latent_finite_difference` | `supported` | Central finite difference evaluates the complete nonlinear latent path. |
| `transform:ssgf.latent_parameter_shift` | `permanent_boundary` | Circuit parameter-shift does not directly compute `dC/dz` because `z` enters Hamiltonian coefficients through `softplus(W(z))`. |

Unsupported latent dimensions fail before an ambient quantum evaluation. The
product accepts exactly `n_oscillators * (n_oscillators - 1) // 2` latent
parameters and caps evidence probes at six oscillators.

## Certificates

```python
import numpy as np

from scpn_quantum_control.ssgf_geometry_gradient_product import (
    certify_geometry_gradient,
    certify_quantum_cost,
    materialise_outer_cycle_evidence,
)

z = np.array([0.2, -0.4, 0.7])
theta = np.array([0.1, 0.6, 1.4])

cost = certify_quantum_cost(z, 3, theta, trotter_reps=1)
assert cost.cost == cost.c_micro
assert cost.cost == 1.0 - cost.r_global

gradient = certify_geometry_gradient(z, 3, theta, trotter_reps=1)
assert gradient.route_id == "transform:ssgf.latent_finite_difference"
assert gradient.refinement_max_abs_delta <= 5e-3
assert gradient.periodic_gradient_max_abs_delta <= 1e-9

trace = materialise_outer_cycle_evidence(
    n_oscillators=3,
    z_init=z,
    theta_init=theta,
    max_iterations=3,
)
assert trace.evidence_label == "functional_non_isolated_local_simulation"
```

The cost certificate cross-checks the public `quantum_cost` result against
`compute_quantum_costs().c_micro` and the complement law `C + R = 1`. The
gradient certificate checks the expected evaluation count, step refinement
from `epsilon` to `epsilon / 2`, and invariance under a global `2*pi` phase
shift. These are metamorphic checks, not a proof of analytic AD.

## Composition boundaries

- Geometric-control consumers use `SsgfGeometryObserverRecord` as geometry
  diagnostics.
- Co-design evaluators may consume the same immutable record as optional
  telemetry; it is not an operational controller decision.
- The negative-space registry keeps wrong latent dimensions, latent
  parameter-shift, analytic-AD promotion, hardware promotion, and convergence
  promotion as explicit refusals.
- The hardware-safe execution policy remains authoritative for any future
  hardware execution. This SSGF product performs no hardware submission.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
