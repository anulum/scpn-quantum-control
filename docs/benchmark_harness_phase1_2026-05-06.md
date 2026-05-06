# S5 Phase 1 Benchmark Harness

This artefact records a no-QPU open-data reproduction of the Phase 1 DLA-parity dataset.

## Command

```bash
scpn-bench s5-benchmark-suite
```

## Dataset
- Runs: `4`
- Circuits: `342`
- Backends: `ibm_kingston`
- Integrity verification: `False`

## Reproduced Statistics
- Depth points: `8`
- Peak asymmetry depth: `6`
- Peak relative asymmetry: `0.17477782`
- Mean relative asymmetry: `0.09245549`
- Fisher chi2: `123.40011441`
- Fisher df: `16`

## Classical Reference
- Backend: `numpy`
- Max absolute leakage: `0.000e+00`
- Zero within tolerance: `True`

## Claim Boundary
- no new hardware execution
- no quantum advantage claim
- classical baseline is noiseless parity-conservation reference
- published hardware statistics are reproduced from committed raw counts
