# S2 Scaling Claim-boundary Report

Protocol: `s2_quantum_advantage_scaling_2026-05-06`

Validation: `True`

## Allowed Claims
- Report that the S2 protocol, schema, and lite no-QPU rehearsal path are operational.
- Report measured small-size classical ODE and dense exact-diagonalisation timings where rows are ok.
- Report explicit skipped rows as coverage of the validation contract, not as performance data.
- Discuss MPS/TN spoofability only for measured MPS/TN rows.

## Forbidden Claims
- Do not claim broad quantum advantage from lite rows.
- Do not claim hardware scaling because no QPU rows are present.
- Do not claim tensor-network hardness unless MPS/TN rows are measured and non-spoofable.
- Do not extrapolate skipped rows into crossover estimates.

## Remaining Blockers
- Run full required MPS/TN baseline rows.
- Run Aer/statevector rows for the selected size grid.
- Measure sparse eigensolver rows or record justified size-gated failures.
- Add hardware rows only after preregistration, QPU approval, raw-count storage, and validation gates.

## Protocol Claim Boundary
S2 may publish scaling, memory, and spoofability boundaries. It must not claim broad quantum advantage without preregistered hardware data and classical tensor-network baselines at the same problem family.
