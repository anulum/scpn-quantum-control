# BL-45 Quantum Reservoir and Surrogate Evidence

Schema: `scpn.quantum_reservoir_surrogates.v1`
Content digest: `9db8f66f5f930cc06b67e0a2a9cd04fdd77393f0f60a256efab6ee49da2244bc`

## Held-out reservoir certificates

| Task | Train / validation | QRC / ESN features | QRC validation MSE | ESN validation MSE | Lower MSE |
|---|---:|---:|---:|---:|---|
| `classification` | 18 / 8 | 6 / 6 | 0.069647 | 0.091391 | `qrc` |
| `forecast` | 18 / 8 | 6 / 6 | 0.890832 | 0.019553 | `esn` |

## Classical surrogate fidelity

- Held-out value fidelity: `passed=True`, RMSE `0.000423`, maximum error `0.000792`, R² `0.999995`.
- Analytic-gradient fidelity: `passed=True`, RMSE `0.000525`, maximum error `0.000894` against exact local central differences.
- Exact proposal validation: `exact_local_improvement`; the BL-33 proposal remains unapplied.

## Support matrix

| Surface | Status | Evidence / boundary |
|---|---|---|
| `qrc_heldout_certificates` | `local_exact_supported` | Two disjoint synthetic task certificates. Small-system exact statevector only. |
| `matched_esn_comparator` | `bounded_supported` | QRC and ESN use equal readout feature counts. No winner or advantage assumption. |
| `gaussian_rbf_value_fidelity` | `local_exact_supported` | Disjoint held-out values pass frozen thresholds. One frozen two-parameter simulator objective. |
| `analytic_rbf_gradient_fidelity` | `local_exact_supported` | Analytic RBF gradients match exact central differences. Finite-difference reference, not hardware gradients. |
| `codesign_exact_validated_proposal` | `bounded_supported` | Surrogate proposal followed by exact local objective query. ControllerProposal remains unapplied. |
| `multimodal_forecasting_adapter` | `blocked_dependency` | BL-37 multimodal schema is not implemented. No invented domain adapter or operational data. |
| `notebook_programme` | `blocked_dependency` | BL-40 notebook stretch is outside the BL-45 product. No notebook is represented as complete. |

## Claim boundary

Synthetic local exact-statevector and classical-reference evidence only. No hardware QRC, provider execution, unseen-domain generalisation, closed-loop control, optimisation advantage, publication, or deployment claim.
