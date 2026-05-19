# Logical DLA Parity Roadmap

This note is a post-NISQ planning artefact. It estimates resources for
logical-level DLA parity work and keeps the survival claim blocked until
the theory and simulation prerequisites are closed.

## Boundary

roadmap and resource estimate only; no claim that DLA parity survives logical encoding and no hardware submission

## Resource Table

| N | d | Flat surface-code qubits | Repetition scaffold qubits | QEC rounds | Wall-clock us/step | p_L/round | Step fidelity | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 3 | 272 | 80 | 3 | 3.000 | 9.000e-03 | 0.649209377 | theory_required_before_simulation_or_hardware_promotion |
| 16 | 5 | 784 | 144 | 5 | 5.000 | 2.700e-03 | 0.805735302 | theory_required_before_simulation_or_hardware_promotion |
| 16 | 7 | 1552 | 208 | 7 | 7.000 | 8.100e-04 | 0.913273392 | theory_required_before_simulation_or_hardware_promotion |

## Multiscale QEC Cross-Check

- Flat d=7 surface-code qubits: `1552`
- Flat d=7 logical error rate: `8.100e-04`
- MS-QEC qubits for distances [3, 3, 3, 3, 3]: `1360`
- MS-QEC effective logical rate: `1.000e+00`
- Overhead ratio MS-QEC/flat: `0.876289`
- Conclusion: `hierarchical_lower_qubit_overhead_but_logical_rate_not_viable`

## Prerequisites
- representation-theory review of XY Hamiltonian generators under the stabiliser group
- logical observable definition for DLA parity before Monte Carlo promotion
- noise-channel calibration separated from hardware-execution claims
- negative-result framing if the logical code destroys the physical parity signal

## Gate

Regenerate and compare this roadmap with:

```bash
scpn-bench s7-logical-dla-roadmap
```
