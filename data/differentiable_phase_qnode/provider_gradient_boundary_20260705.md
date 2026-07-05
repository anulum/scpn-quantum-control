# Provider Gradient Boundary Evidence

- artifact_id: `provider-gradient-boundary-20260705`
- source_commit: `b67dabef`
- classification: `functional_no_submit`
- no_submit: `True`
- promotion_ready: `False`
- hardware_execution_count: `0`
- gradient_available_count: `0`

| Surface | Passed | Records | Supported/Approved | Blocked | Hardware executions | Gradient results |
|---|---:|---:|---:|---:|---:|---:|
| `provider_gradient_readiness` | `True` | 6 | 3 | 3 | 0 | 0 |
| `hardware_gradient_policy_readiness` | `True` | 6 | 1 | 5 | 0 | 0 |
| `provider_hardware_gradient_preparation` | `True` | 6 | 2 | 4 | 0 | 0 |

Promotion blockers:

- live execution ticket missing
- raw-count replay artefact missing
- calibration snapshot artefact missing
- statevector comparison artefact missing
- isolated benchmark artefact missing
- validated provider hardware evidence chain missing

Claim boundary: No-submit provider-gradient boundary evidence only; supported records are local callback, policy, or preparation checks and blocked records are fail-closed governance evidence. This artifact does not promote live provider execution, QPU execution, hardware-gradient results, isolated benchmark status, or performance claims.
