<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — S2 Scaling Readiness Index -->

# S2 Scaling Readiness Index

Date: 2026-05-06

This index records the current no-QPU readiness state for S2 quantum-advantage
scaling benchmarks. It is a claim-boundary document, not an advantage claim.

## Current Status

S2 is ready at the lite rehearsal layer. The protocol, required baselines, row
schema, validator, lite harness, and claim-boundary report exist. The full N=20
campaign and any hardware rows remain blocked until the heavier baselines,
resource accounting, and QPU approval gates are complete.

## Canonical Command

```bash
scpn-bench s2-scaling-lite
```

This regenerates:

- `data/s2_advantage_scaling/s2_scaling_protocol_2026-05-06.json`;
- `data/s2_advantage_scaling/s2_scaling_lite_rows_2026-05-06.json`;
- `data/s2_advantage_scaling/s2_scaling_claim_boundary_2026-05-06.json`;
- `data/s2_advantage_scaling/s2_scaling_claim_boundary_2026-05-06.md`.

## Lite Baseline Rows

| n | Baseline | Status | Wall time |
|---|----------|--------|-----------|
| 4 | classical ODE | ok | 4.621 ms |
| 4 | dense exact diagonalisation | ok | 0.185 ms |
| 4 | sparse eigensolver | ok | 16.122 ms |
| 4 | MPS/TN spoofability | ok | 0.303 ms |
| 4 | Aer/statevector | ok | 29.627 ms |
| 6 | classical ODE | ok | 10.599 ms |
| 6 | dense exact diagonalisation | ok | 20.920 ms |
| 6 | sparse eigensolver | ok | 14.413 ms |
| 6 | MPS/TN spoofability | ok | 0.924 ms |
| 6 | Aer/statevector | ok | 29.307 ms |

The timings are opportunistic local measurements on the current workstation.
They prove the protocol path and row schema, not publication-grade crossover
positions.

## Allowed Claims

- The S2 protocol and validation gate are implemented.
- The lite rehearsal path measures all required baseline families at small sizes.
- The current artefacts are suitable for debugging the scaling workflow.
- The MPS/TN row exists as a spoofability diagnostic, not as hardness evidence.
- No hardware was submitted and no QPU time was spent.

## Forbidden Claims

- Do not claim broad quantum advantage from S2 lite rows.
- Do not claim hardware scaling from S2 lite rows.
- Do not extrapolate crossover positions from the N=4 and N=6 lite rehearsal.
- Do not claim tensor-network hardness until larger MPS/TN rows are measured and
  interpreted.
- Do not compare QPU budget to classical baselines until preregistered hardware
  rows exist.

## Full-campaign Blockers

- [ ] Extend rows to the protocol grid N=4,6,8,10,12,14,16,18,20.
- [x] Add lite-harness memory instrumentation beyond simple estimates.
- [ ] Run MPS/TN diagnostics at larger N with explicit bond caps.
- [ ] Run Aer/statevector rows until the statevector memory gate is reached.
- [ ] Record GPU dense reference rows where CUDA is available.
- [ ] Convert skipped rows into measured rows or justified size-gated failures.
- [ ] Add plot generation only after row validation passes.
- [ ] Add hardware rows only after preregistration, QPU approval, raw-count
      storage, and analysis gates.

## Hardware Boundary

Hardware rows are optional in the protocol until credits and a preregistered
campaign exist. Missing hardware must degrade the work to a classical
scaling/spoofability study. It must not be represented as a hardware quantum
advantage result.

## Next S2 Step

The next non-QPU step is to add full-grid execution controls around the measured
classical and simulator baselines, keeping expensive rows behind explicit size
gates.
