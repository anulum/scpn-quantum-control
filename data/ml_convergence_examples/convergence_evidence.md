# QNN/QGNN/QSNN convergence evidence

- Schema: `ml_convergence_examples.v1`
- Passed: `true`
- Content digest: `c98b312b370955cfe21a73c870e42a56ec7b1857b359991e2c7e1cb4687e62a3`
- Execution: local synthetic simulator only; no provider or hardware execution.

## Convergence certificates

| Family | Example | Initial loss | Best loss | Target | Loss drop | Replay | Passed |
|---|---|---:|---:|---:|---:|---|---|
| qnn | qnn_phase_separable_binary | 0.022996705 | 7.10821488e-05 | 0.0001 | 0.0229256229 | True | True |
| qgnn | qgnn_kuramoto_graph_regression | 0.508950006 | 0.00341457457 | 0.005 | 0.505535431 | True | True |
| qsnn | qsnn_single_synapse_silencing | 0.772880023 | 5.61583759e-07 | 1e-05 | 0.772879461 | True | True |

## Framework matrix

| Family | Framework | Status | Required | Executed | Passed | Reason |
|---|---|---|---|---|---|---|
| qnn | scpn_parameter_shift | ran | True | True | True | canonical bounded phase-QNN trainer executed |
| qnn | jax | ran | False | True | True | native bounded-QNN gradient agrees with parameter shift |
| qnn | pytorch | ran | False | True | True | native bounded-QNN gradient agrees with parameter shift |
| qnn | tensorflow | unavailable | False | False | None | optional dependency 'tensorflow' is not installed |
| qnn | provider_hardware_gradient | unsupported | False | False | None | provider hardware gradients require separate job, shot, and approval evidence |
| qgnn | scpn_message_passing_phase_qnode | ran | True | True | True | existing bounded QGNN trainer executed its exact chained gradient |
| qgnn | jax | not_applicable | False | False | None | no framework-native adapter is registered for the bounded QGNN surface |
| qgnn | pytorch | not_applicable | False | False | None | no framework-native adapter is registered for the bounded QGNN surface |
| qgnn | tensorflow | not_applicable | False | False | None | no framework-native adapter is registered for the bounded QGNN surface |
| qsnn | scpn_qsnn_statevector | ran | True | True | True | existing QSNN parameter-shift trainer and final spike readout executed |
| qsnn | jax | not_applicable | False | False | None | no framework-native adapter is registered for the QSNN dense-layer trainer |
| qsnn | pytorch | not_applicable | False | False | None | no framework-native adapter is registered for the QSNN dense-layer trainer |
| qsnn | tensorflow | not_applicable | False | False | None | no framework-native adapter is registered for the QSNN dense-layer trainer |
| qsnn | neuromorphic_hardware | unsupported | False | False | None | the local statevector example is not a neuromorphic-hardware training route |

## Claim boundary

deterministic synthetic local QNN/QGNN/QSNN training evidence on frozen small tasks; no arbitrary-architecture, generalisation, SOTA, provider, QPU, neuromorphic-hardware, or production convergence claim
