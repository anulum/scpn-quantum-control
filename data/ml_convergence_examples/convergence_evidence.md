# QNN/QGNN/QSNN convergence evidence

- Schema: `ml_convergence_examples.v1`
- Passed: `true`
- Content digest: `97e31026e81267c2cae42cb0254e380fe9b49cd288a33693e44e40f652d8ee22`
- Execution: local synthetic simulator only; no provider or hardware execution.

## Convergence certificates

| Family | Example | Step budget | Initial loss | Best loss | Target | Loss drop | Replay | Passed |
|---|---|---:|---:|---:|---:|---:|---|---|
| qnn | qnn_phase_separable_binary | 80 | 0.022996705 | 7.10821488e-05 | 0.0001 | 0.0229256229 | True | True |
| qgnn | qgnn_kuramoto_graph_regression | 60 | 0.508950006 | 0.00341457457 | 0.005 | 0.505535431 | True | True |
| qsnn | qsnn_single_synapse_silencing_v2 | 1024 | 0.772880023 | 1.50475808e-06 | 1e-05 | 0.772878518 | True | True |

## Retained predecessor diagnostics

The predecessor budget is replayed with the current gradients and its original loss gates.

| Example | Step budget | Target | Best loss | Replay | Passed |
|---|---:|---:|---:|---|---|
| qsnn_single_synapse_silencing | 16 | 1e-05 | 0.00883347095 | True | False |

## Framework matrix

| Family | Framework | Status | Required | Executed | Passed | Reason |
|---|---|---|---|---|---|---|
| qnn | scpn_parameter_shift | ran | True | True | True | executed QNN protocol qnn_phase_separable_binary; convergence passed=True |
| qnn | jax | ran | True | True | True | native bounded-QNN gradient agrees with parameter shift |
| qnn | pytorch | ran | True | True | True | native bounded-QNN gradient agrees with parameter shift |
| qnn | tensorflow | ran | True | True | True | native bounded-QNN gradient agrees with parameter shift |
| qnn | provider_hardware_gradient | unsupported | False | False | None | provider hardware gradients require separate job, shot, and approval evidence |
| qgnn | scpn_message_passing_phase_qnode | ran | True | True | True | executed QGNN protocol qgnn_kuramoto_graph_regression; convergence passed=True |
| qgnn | jax | not_applicable | False | False | None | no framework-native adapter is registered for the bounded QGNN surface |
| qgnn | pytorch | not_applicable | False | False | None | no framework-native adapter is registered for the bounded QGNN surface |
| qgnn | tensorflow | not_applicable | False | False | None | no framework-native adapter is registered for the bounded QGNN surface |
| qsnn | scpn_qsnn_statevector | ran | True | True | True | executed QSNN protocol qsnn_single_synapse_silencing_v2; convergence passed=True |
| qsnn | jax | not_applicable | False | False | None | no framework-native adapter is registered for the QSNN dense-layer trainer |
| qsnn | pytorch | not_applicable | False | False | None | no framework-native adapter is registered for the QSNN dense-layer trainer |
| qsnn | tensorflow | not_applicable | False | False | None | no framework-native adapter is registered for the QSNN dense-layer trainer |
| qsnn | neuromorphic_hardware | unsupported | False | False | None | the local statevector example is not a neuromorphic-hardware training route |

## Claim boundary

deterministic synthetic local QNN/QGNN/QSNN training evidence on frozen small tasks; no arbitrary-architecture, generalisation, SOTA, provider, QPU, neuromorphic-hardware, or production convergence claim
