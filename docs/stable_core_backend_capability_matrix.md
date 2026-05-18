<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- stable core backend capability matrix -->

# Stable Core Backend Capability Matrix

This generated page records stable backend capability profiles.

| Backend | Kind | Capabilities | Hardware submission | Claim boundary |
|---|---|---|---|---|
| `classical-reference` | `classical_reference` | order_parameter, parity, fim, control | `False` | Capability profile only; adapter implementation, dependency checks, and evidence gates remain required before execution claims. |
| `hardware-replay` | `hardware_replay` | order_parameter, parity, mitigation_replay | `False` | Capability profile only; adapter implementation, dependency checks, and evidence gates remain required before execution claims. |
| `qiskit-runtime` | `qiskit` | order_parameter, parity, mitigation_replay | `False` | Capability profile only; adapter implementation, dependency checks, and evidence gates remain required before execution claims. |
| `qutip-dynamics` | `qutip` | order_parameter, hamiltonian_dynamics, lindblad | `False` | Capability profile only; adapter implementation, dependency checks, and evidence gates remain required before execution claims. |
| `pennylane-autodiff` | `pennylane` | order_parameter, parity, control, autodiff | `False` | Capability profile only; adapter implementation, dependency checks, and evidence gates remain required before execution claims. |
| `pulser-surrogate` | `pulser_surrogate` | order_parameter, analog_surrogate, pulse_schedule | `False` | Capability profile only; adapter implementation, dependency checks, and evidence gates remain required before execution claims. |

## Payload boundary

This payload records stable backend capability profiles only. It does not prove adapter implementation, dependency availability, or hardware execution readiness.
