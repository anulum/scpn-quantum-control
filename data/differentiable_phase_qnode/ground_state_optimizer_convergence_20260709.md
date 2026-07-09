# Ground-State Optimizer Convergence Evidence

- Schema: `scpn_qc_ground_state_optimizer_convergence_v1`
- Artifact id: `ground-state-optimizer-convergence-local`
- Classification: `functional_non_isolated`
- Passed: `True`
- Claim boundary: local functional optimizer-convergence evidence on deterministic small phase ground-state objectives; not isolated-core timing evidence, not hardware execution, and not a global optimality claim

## Summary

- Objective cases: `2`
- Executable rows: `10`
- Boundary rows: `1`
- Optimizers: `natural_gradient, adam, lbfgs, spsa, cobyla`

## Executable Rows

| Case | Optimizer | Best energy | Energy error | Parameter distance | Evaluations | Passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `single_qubit_z_rotation_ground` | `natural_gradient` | -1 | 0 | 9.05e-09 | 187 | `True` |
| `single_qubit_z_rotation_ground` | `adam` | -0.999975896126 | 2.41e-05 | 0.00694 | 396 | `True` |
| `single_qubit_z_rotation_ground` | `lbfgs` | -1 | 0 | 6.86e-10 | 33 | `True` |
| `single_qubit_z_rotation_ground` | `spsa` | -0.999976233639 | 2.38e-05 | 0.00689 | 265 | `True` |
| `single_qubit_z_rotation_ground` | `cobyla` | -0.999994413339 | 5.59e-06 | 0.00334 | 17 | `True` |
| `two_qubit_product_ising_ground` | `natural_gradient` | -1.7 | 0 | 1.38e-08 | 239 | `True` |
| `two_qubit_product_ising_ground` | `adam` | -1.69992461342 | 7.54e-05 | 0.0147 | 475 | `True` |
| `two_qubit_product_ising_ground` | `lbfgs` | -1.69999999969 | 3.15e-10 | 2.86e-05 | 31 | `True` |
| `two_qubit_product_ising_ground` | `spsa` | -1.69993242438 | 6.76e-05 | 0.0136 | 130 | `True` |
| `two_qubit_product_ising_ground` | `cobyla` | -1.69996598476 | 3.4e-05 | 0.015 | 17 | `True` |

## Boundary Rows

| Case | Optimizer | Failure class | Boundary |
| --- | --- | --- | --- |
| `qng_qjit_class_metric_fusion` | `qng_qjit_class_boundary` | `unsupported_qjit_metric_fusion` | The local BL-15 suite exposes Python parameter-shift natural-gradient evidence only. A QNG-QJIT-class route needs a compiler-owned metric fusion and executable lowering contract before it can be compared. |
