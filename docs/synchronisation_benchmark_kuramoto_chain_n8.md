<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- synchronisation benchmark documentation -->

# Kuramoto Ring n=4 Synchronisation Benchmark

This no-QPU artefact records schema-compatible reference rows for the canonical four-node Kuramoto-XY ring benchmark.

Benchmark ID: `kuramoto_chain_n8_decay_omega`

| Backend | Observable | Value | Tolerance | Passed |
|---|---|---:|---:|---|
| `classical_ode_scipy_dop853` | `order_parameter_t1` | `0.364101086046` | `1e-09` | `True` |
| `dense_exact_xy_numpy_scipy` | `state_norm_t1` | `1` | `1e-10` | `True` |
| `dense_exact_xy_numpy_scipy` | `energy_expectation_t1` | `-5.77740023677` | `1e-10` | `True` |

## Claim boundary

This artefact establishes n=8 decaying-chain reference rows for the standardised synchronisation benchmark suite. It does not submit QPU jobs or claim quantum advantage.
