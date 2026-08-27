# Quantum Reservoir and Surrogate Evidence

Schema: `scpn.quantum_reservoir_surrogates.v1`
Content digest: `e08bae2ba437f0ab4ab479ffc81d5afd1915e3feb86bcdf59b8018e14e166328`

## Held-out reservoir certificates

| Task | Train / validation | QRC / ESN features | QRC validation MSE | ESN validation MSE | Lower MSE |
|---|---:|---:|---:|---:|---|
| `classification` | 18 / 8 | 6 / 6 | 0.069647 | 0.091391 | `qrc` |
| `forecast` | 18 / 8 | 6 / 6 | 0.890832 | 0.019553 | `esn` |

## Classical surrogate fidelity

- Held-out value fidelity: `passed=True`, RMSE `0.000423`, maximum error `0.000792`, R² `0.999995`.
- Analytic-gradient fidelity: `passed=True`, RMSE `0.000525`, maximum error `0.000894` against exact local central differences.
- Exact proposal validation: `exact_local_improvement`; the controller proposal remains unapplied.

## Support matrix

| Surface | Status | Evidence / boundary |
|---|---|---|
| `qrc_heldout_certificates` | `local_exact_supported` | Two disjoint synthetic task certificates. Small-system exact statevector only. |
| `matched_esn_comparator` | `bounded_supported` | QRC and ESN use equal readout feature counts. No winner or advantage assumption. |
| `gaussian_rbf_value_fidelity` | `local_exact_supported` | Disjoint held-out values pass frozen thresholds. One frozen two-parameter simulator objective. |
| `analytic_rbf_gradient_fidelity` | `local_exact_supported` | Analytic RBF gradients match exact central differences. Finite-difference reference, not hardware gradients. |
| `codesign_exact_validated_proposal` | `bounded_supported` | Surrogate proposal followed by exact local objective query. ControllerProposal remains unapplied. |
| `multimodal_forecasting_adapter` | `blocked_dependency` | The multimodal forecasting adapter is not implemented. No invented domain adapter or operational data. |
| `differentiable_notebook_curriculum` | `blocked_dependency` | Differentiable notebook curriculum expansion is outside the quantum-reservoir surrogate evidence scope. No notebook is represented as complete. |

## Claim boundary

Synthetic local exact-statevector and classical-reference evidence only. No hardware QRC, provider execution, unseen-domain generalisation, closed-loop control, optimisation advantage, publication, or deployment claim.
