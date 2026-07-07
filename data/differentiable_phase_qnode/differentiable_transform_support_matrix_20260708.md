<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable Transform-Algebra Support Matrix
-->

# Differentiable Transform-Algebra Support Matrix

- Schema: `scpn_qc_differentiable_transform_support_matrix_v1`
- Artifact ID: `diff-transform-support-matrix-20260708`
- Supported rows: `9/13`
- Fail-closed blocked rows: `4`
- Source audit cases: `24` (`20` passed, `4` blocked)
- Claim boundary: generated from bounded local transform-algebra audit rows; supported rows are local conformance evidence only, and blocked rows are not promoted to analytic, framework-native, compiler, provider, hardware, or performance claims

| Row | Lane | Transform stack | Status | Residual | Tolerance | Blockers | Notes |
|---|---|---|---|---|---|---|---|
| `native_grad_vmap` | `native` | grad<br>vmap | `passed` | `1.388e-11` | `5.0e-05` | — | native local finite-difference diagnostic transform composition |
| `native_vmap_grad` | `native` | vmap<br>grad | `passed` | `8.916e-11` | `5.0e-05` | — | native local row-wise gradient vectorisation |
| `native_jacfwd_jacrev` | `native` | jacfwd<br>jacrev | `passed` | `0.000e+00` | `5.0e-05` | — | forward and reverse finite-difference Jacobian routes agree |
| `native_hessian` | `native` | hessian | `passed` | `0.000e+00` | `5.0e-05` | — | smooth scalar-objective Hessian symmetry |
| `native_jvp_vjp` | `native` | jvp<br>vjp | `passed` | `4.885e-11` | `5.0e-05` | — | directional transform adjoint identity |
| `registered_custom_rules` | `custom_rules` | vmap<br>custom_jvp<br>custom_vjp | `passed` | `0.000e+00` | `5.0e-05` | — | registered exact custom JVP/VJP rules under native vmap |
| `program_ad_jvp_vjp` | `program_ad` | jvp<br>vjp<br>vmap<br>whole_program_grad | `passed` | `5.412e-15` | `5.0e-05` | — | directional transforms over vmap of whole-program AD gradients |
| `program_ad_hessian` | `program_ad` | hessian<br>whole_program_value_and_grad | `passed` | `7.661e-15` | `5.0e-05` | — | Hessian over a whole-program AD scalar objective |
| `quantum_gradient_native_nesting` | `quantum_gradients` | vmap<br>grad<br>parameter_shift | `passed` | `5.551e-17` | `5.0e-05` | — | deterministic native local phase-QNode manual vmap(grad) |
| `unsupported_custom_rule_registration` | `unsupported_boundary` | custom_jvp<br>custom_vjp | `blocked` | `n/a` | `5.0e-05` | custom JVP/VJP composition requires registered exact rules and adjoint identity evidence before promotion | unregistered custom rules remain fail-closed |
| `unsupported_complex_valued_objective` | `unsupported_boundary` | complex_step<br>wirtinger | `blocked` | `n/a` | `5.0e-05` | complex-step support is limited to real-valued analytic objectives; complex-valued objectives need Wirtinger-specific contracts | complex-valued objectives need Wirtinger-specific contracts |
| `unsupported_structured_container` | `unsupported_boundary` | pytree<br>structured_container | `blocked` | `n/a` | `5.0e-05` | structured parameter containers need explicit PyTree/container metadata before transform composition can be promoted | structured containers need explicit metadata before promotion |
| `unsupported_nondifferentiable_boundary` | `unsupported_boundary` | grad<br>nondifferentiable | `blocked` | `n/a` | `5.0e-05` | abs at zero has a cusp; central finite difference may produce a number but cannot promote differentiability | nondifferentiable cusps are diagnostic-only boundaries |

Blocked rows are explicit fail-closed boundaries, not failures. The
artefact mirrors the executable audit and cannot promote any row beyond
its generated status.
