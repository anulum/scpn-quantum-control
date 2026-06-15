# FRC Pulsed-Shot QAOA Scheduling

SPDX-License-Identifier: AGPL-3.0-or-later

`scpn_quantum_control.control.qaoa_pulsed_cost` and
`scpn_quantum_control.control.frc_pulsed_qaoa` schedule capacitor-bank firing for
a field-reversed-configuration (FRC) pulsed-compression shot by minimising a
control-grade physics cost with QAOA.

This is a **control adapter**, not a transport solver: the high-fidelity FRC
physics is owned by SCPN-FUSION-CORE. The cost function evaluates the published
closed forms of the quantities the scheduler trades off, and the
`FRCPlasmaSurrogate` parameters can be matched to a FUSION run.

## Cost function

`frc_pulsed_shot_cost(schedule, target_b_profile, available_capacitor_energy_J,
objective)` returns the weighted sum of four dimensionless penalties:

| Term | Physics | Reference |
|---|---|---|
| s-parameter deviation | `s = R_s / rho_i`, flux-compression scaling | Steinhauer, Phys. Plasmas 18, 070501 (2011) |
| bank energy over budget | fired banks × energy-per-bank vs budget | — |
| MRTI growth | `gamma = sqrt(A_T k g - k² B² / (mu0 (rho_h+rho_l)))`, amplitude `a0 exp(∫gamma dt)` | Velikovich, Phys. Plasmas 14, 022701 (2007); Sefkow, Phys. Plasmas 21, 072711 (2014) |
| tilt-mode margin | kinetic stabilisation via `S*/E` | Belova, Phys. Plasmas 8, 1267 (2001) |

```python
from scpn_quantum_control.control.qaoa_pulsed_cost import FRCQAOAObjective, frc_pulsed_shot_cost
import numpy as np

objective = FRCQAOAObjective(
    target_s_parameter=2.5, bank_energy_budget_J=5e5,
    mrti_amplitude_max_m=1e-2, tilt_margin_required=0.3,
)
cost, components = frc_pulsed_shot_cost(
    np.array([1, 0, 1, 1, 0, 1, 1, 0]), np.linspace(0.5, 4.0, 8), 1e6, objective,
    return_components=True,
)
```

## Optimisers

The cost is a general (non-quadratic) function of the binary schedule, so the
QAOA cost layer is the exact diagonal phase separator
`exp(-i gamma diag(cost))` (Farhi, Goldstone, Gutmann, arXiv:1411.4028).

- `optimal_schedule` — exact brute-force minimum (reference, horizon ≤ 16).
- `classical_sqp_schedule` — SciPy SLSQP relaxation plus rounding (classical
  NMPC-style baseline).
- `solve_frc_pulsed_qaoa` — multi-restart p-layer QAOA used as a sampler.

On the documented test cases QAOA reaches the brute-force optimum (within the
5 % acceptance band).

## Acceleration

The MRTI growth integral (the per-evaluation physics kernel) dispatches to a Rust
kernel that matches the NumPy reference to ``1e-12`` relative tolerance; the
scalar penalties stay in Python.

Measured per-call wall-time (release build, median of 7,
`scripts/bench_frc_pulsed_qaoa.py`, artefact
`results/frc_pulsed_qaoa_benchmark.json`, `functional_non_isolated`):

| profile length | NumPy | Rust | speed-up |
|---|---|---|---|
| 8 | 25.8 µs | 0.44 µs | 59× |
| 256 | 25.5 µs | 1.42 µs | 18× |
| 1 024 | 28.5 µs | 4.43 µs | 6.5× |
| 4 096 | 47.7 µs | 16.8 µs | 2.9× |

The Rust kernel is the fastest measured backend at every profile length (NumPy's
per-call overhead dominates on short profiles), so it sits at the top of the
dispatch chain.

## Consumers

SCPN-MIF-CORE imports `frc_pulsed_shot_cost` and `FRCQAOAObjective` for
shot-sequence optimisation; SCPN-FUSION-CORE supplies the high-fidelity FRC
physics and SCPN-CONTROL the NMPC comparison.
