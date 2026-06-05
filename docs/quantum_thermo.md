# Quantum Thermodynamics Readiness

This is the S9 no-submit readiness surface for thermodynamic signatures
of synchronisation transitions. It records calibrated observables and
protocol prerequisites with no hardware submission and no thermodynamic
peak claim.

## Why this page exists

This page is for teams evaluating thermodynamic observables on synchronization
models under bounded evidence rules. It captures readiness inputs before any
hardware-run claim is promoted, and it keeps the route to thermodynamic peak
promotion visible.

## Boundary

readiness and calibrated-protocol estimate only; no thermodynamic peak claim and no hardware submission

## K-Sweep Protocol

| K | entropy production nat/s | heat current J/s | classical reference nat/s |
| ---: | ---: | ---: | ---: |
| 0.4 | 0.127091 | 0.0433863 | 0.124549 |
| 0.6 | 0.27261 | 0.0615763 | 0.267158 |
| 0.8 | 0.42 | 0.08 | 0.4116 |
| 1 | 0.27261 | 0.0615763 | 0.267158 |
| 1.2 | 0.127091 | 0.0433863 | 0.124549 |

## Readiness

- Peak K candidate: `0.8`
- Hardware submission allowed: `False`
- Thermodynamic peak claim allowed: `False`

## Falsifier

no statistically significant peak above the classical baseline across the K-sweep

## Prerequisites
- formal theory pass for DLA-sector entropy-production decomposition
- hardware backend and readout-mitigation plan approved before execution
- classical Lindblad or QuTiP reference fixed before hardware comparison
- raw-count execution and Zenodo archive required before thermodynamic peak claims

## Gate

Regenerate and compare this readiness artefact with:

```bash
scpn-bench s9-quantum-thermo-readiness
```
