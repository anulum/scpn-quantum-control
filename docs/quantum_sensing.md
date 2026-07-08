# Quantum Sensing Readiness

This is the S11 no-submit readiness surface for DLA-driven quantum
sensing via the synchronisation order parameter. It records QFI and
classical Fisher proxy rows without hardware submission or sensing
advantage promotion.

## Why this page exists

This page supports teams comparing sensing hypotheses against classical
baselines. It captures reproducible gain estimates and the required
prerequisites before any promotion of sensing-advantage claims.

## Boundary

QFI and sync-order sensing readiness estimate only; no hardware submission and no sensing-advantage claim

## Gain Scan

| K | QFI | spectral gap | R | classical Fisher proxy | gain ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.758419 | 0.031124 | 0.790569 | 7.03137 | 0.107862 |
| 0.8 | 0.178387 | 0.0327567 | 0.901388 | 0.338042 | 0.527706 |
| 1.2 | 0.0762066 | 0.033075 | 0.935414 | 0.0620041 | 1.22906 |

## Readiness

- Peak-QFI K: `0.4`
- Optimal readiness K: `1.2`
- Best gain ratio estimate: `1.22906`
- Hardware submission allowed: `False`
- sensing advantage claim allowed: `False`

## QFI-Criticality Tail

QFI-criticality operating-point recommendation only; no probe has been run on hardware and no sensing-advantage claim is allowed

- Operating K: `0.4`
- Selected coupling pair: `(1, 2)`
- Pair QFI: `0.758419`
- Spectral gap at operating point: `0.031124`
- Cramer-Rao variance bound: `0.000131853`
- Cramer-Rao standard-deviation bound: `0.0114827`
- Gap-minimum K: `0.4`
- QFI/gap K delta: `0`
- Spectral/geometric cross-check agrees: `True`

## Falsifier

ratio of QFI-based Fisher information to classical Fisher information is below 1 on the pre-registered perturbation benchmark

the QFI peak does not survive the spectral/geometric cross-check or the pre-registered perturbation benchmark

## Prerequisites
- pre-registered perturbation benchmark and classical Fisher estimator fixed
- hardware shot budget and shadow-tomography estimator approved before execution
- raw counts and uncertainty intervals archived before sensing-advantage claims
- applied EEG or Josephson replay target selected before real-world promotion

## Gate

Regenerate and compare this readiness artefact with:

```bash
scpn-bench s11-quantum-sensing-readiness
```
