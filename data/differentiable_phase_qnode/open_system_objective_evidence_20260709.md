# Open-System Objective Evidence

- Schema: `scpn_qc_open_system_objective_evidence_v1`
- Artifact id: `open-system-objective-evidence-local`
- Classification: `functional_non_isolated`
- Passed: `True`
- Claim boundary: Bounded open-system objective evidence uses scipy Lindblad density-matrix evolution and seeded MCWF trajectory ensembles on small local systems. Gradients are deterministic central finite differences over scalar coupling and damping scales; they are not hardware gradients, adjoint Lindblad gradients, unbiased stochastic-gradient estimators, or isolated performance benchmarks.

## Summary

- Objective cases: `2`
- Executable rows: `4`
- Boundary rows: `2`
- Backends: `lindblad_density, mcwf_ensemble`

## Executable Rows

| Case | Backend | Objective | Gradient | Final R | Certificate |
| --- | --- | ---: | --- | ---: | --- |
| `two_qubit_relaxing_sync` | `lindblad_density` | 0.0765113033147 | `-0.0231338, -0.0240455` | 0.696606726933 | `passed` |
| `two_qubit_relaxing_sync` | `mcwf_ensemble` | 0.0995521818529 | `-0.0265226, 0.0105741` | 0.734984012799 | `passed` |
| `two_qubit_dephasing_balance` | `lindblad_density` | 0.0108696349554 | `-0.00217773, -0.00859234` | 0.604130284367 | `passed` |
| `two_qubit_dephasing_balance` | `mcwf_ensemble` | 0.0171438463515 | `-0.00242835, 0.00309363` | 0.627509283544 | `passed` |

## Boundary Rows

| Case | Backend | Failure class | Boundary |
| --- | --- | --- | --- |
| `adjoint_lindblad_gradient_boundary` | `lindblad_adjoint` | `unsupported_adjoint_lindblad_gradient` | Only central finite differences over bounded scalar scales are executed. Continuous adjoint Lindblad sensitivities require a separate validated solver and invariant-preserving gradient checks. |
| `hardware_open_system_gradient_boundary` | `qpu_open_system_gradient` | `no_live_provider_attestation` | No hardware-submitted open-system gradient or provider attestation is included. Provider execution remains behind the live-ticket gate. |
