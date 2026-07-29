# BL-45 Quantum Reservoir and Surrogate Evidence

Schema: `scpn.quantum_reservoir_surrogates.v1`
Content digest: `9efb44e1ee268f49e4ebb789750aac733d7b4f55c2cc7a540c71377d98c36f8b`

## Held-out reservoir certificates

| Task | Train / validation | QRC / ESN features | QRC validation MSE | ESN validation MSE | Lower MSE |
|---|---:|---:|---:|---:|---|
| `classification` | 18 / 8 | 6 / 6 | 0.0696474284 | 0.0913905636 | `qrc` |
| `forecast` | 18 / 8 | 6 / 6 | 0.890831726 | 0.0195529047 | `esn` |

## Classical surrogate fidelity

- Held-out value fidelity: `passed=True`, RMSE `0.00042286964`, maximum error `0.000791764885`, R² `0.999994691`.
- Analytic-gradient fidelity: `passed=True`, RMSE `0.000525290566`, maximum error `0.000893768362` against exact local central differences.
- Exact proposal validation: `exact_local_improvement`; the BL-33 proposal remains unapplied.

## Support matrix

| Surface | Status | Evidence / boundary |
|---|---|---|
| `qrc_heldout_certificates` | `local_exact_supported` | Two disjoint synthetic task certificates. Small-system exact statevector only. |
| `matched_esn_comparator` | `bounded_supported` | QRC and ESN use equal readout feature counts. No winner or advantage assumption. |
| `gaussian_rbf_value_fidelity` | `local_exact_supported` | Disjoint held-out values pass frozen thresholds. One frozen two-parameter simulator objective. |
| `analytic_rbf_gradient_fidelity` | `local_exact_supported` | Analytic RBF gradients match exact central differences. Finite-difference reference, not hardware gradients. |
| `bl33_exact_validated_proposal` | `bounded_supported` | Surrogate proposal followed by exact local objective query. ControllerProposal remains unapplied. |
| `bl37_multimodal_adapter` | `blocked_dependency` | BL-37 multimodal schema is not implemented. No invented domain adapter or operational data. |
| `bl40_notebook` | `blocked_dependency` | BL-40 notebook stretch is outside the BL-45 product. No notebook is represented as complete. |

## Claim boundary

Synthetic local exact-statevector and classical-reference evidence only. No hardware QRC, provider execution, unseen-domain generalisation, closed-loop control, optimisation advantage, publication, or deployment claim.
