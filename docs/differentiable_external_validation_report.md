# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable external-validation technical report

# Differentiable External-Validation Technical Report

This report defines the public comparison and reproducibility package for the
differentiable-programming lane. It is a technical report, not a promotion
claim. Every row remains bounded by the committed claim ledger until external
comparison rows, the claim ledger, and isolated benchmark artefacts all pass.

## Claim Boundary

Current evidence is `functional_non_isolated` unless an artefact explicitly says
otherwise. The report does not claim production performance, quantum advantage,
hardware execution, provider execution, QPU execution, GPU execution, or
`isolated_affinity` benchmark status.

The public wording source is
`data/differentiable_phase_qnode/public_claim_table_20260616.md`. The reviewer
ledger is `data/differentiable_phase_qnode/claim_ledger.md`.

## Reproducibility Package

| Artefact | Role | Boundary |
|---|---|---|
| `data/differentiable_phase_qnode/external_validation_artifact_bundle_20260616.json` | SHA-256 manifest over committed differentiable evidence files. | Checksum provenance only. |
| `data/differentiable_phase_qnode/external_validation_environment_lock_20260616.json` | Exact runtime, developer, CI, framework-overlay, and Enzyme-runner lockfile manifest. | Reviewer reproduction only. |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/diff-qnode-external-comparison.json` | Bounded JAX, PyTorch, PennyLane, Enzyme, TensorFlow, and Catalyst comparison rows, including the dedicated Catalyst compiler-workflow profile added during the 2026-07-04 schema refresh. | Functional non-isolated comparison evidence. |
| `data/differentiable_phase_qnode/identical_circuit_gradient_comparison_20260616.json` | Same-circuit gradient comparison artefact for Qiskit and PennyLane routes. | Correctness comparison, not hardware execution. |
| `data/differentiable_phase_qnode/domain_benchmark_dataset_closure_20260616.json` | Exact-answer domain dataset closure artefact. | Dataset validation, not production benchmark promotion. |
| `data/differentiable_phase_qnode/differentiable_architecture_map_20260627.json` | Architecture and Rustification routing map over inventory rows and scorecard categories. | Routing evidence only, not Rust or performance promotion. |
| `data/differentiable_phase_qnode/differentiable_dependency_environment_map_20260627.json` | Dependency and environment evidence map over runtime, development, CI matrix, framework overlay, and Enzyme runner lock profiles. | Lockfile provenance only, not framework, Enzyme, hardware, or benchmark promotion. |
| `data/differentiable_phase_qnode/differentiable_isolated_benchmark_plan_20260627.json` | Reserved-host batch plan over current non-isolated differentiable benchmark/evidence artefacts. | Planning evidence only, not benchmark execution or `isolated_affinity` promotion. |
| `data/differentiable_phase_qnode/provider_gradient_boundary_20260705.json` | No-submit provider-gradient, hardware-policy, and provider-preparation boundary evidence. | Local callback and governance evidence only, not live provider, QPU, hardware-gradient result, isolated benchmark, or performance promotion. |
| `data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json` | Enzyme/MLIR maturity audit with MLIR-runtime correctness evidence, native Enzyme scalar evidence, and missing raw compiler-AD breadth artifact plus derived breadth evidence recorded as hard gaps. Partial breadth captures are represented through complete artifacts with explicit case hard gaps before promotion. | Hard-gap evidence, not Enzyme parity promotion. |

## Provider Comparison Status

| Provider family | Current SCPN evidence | Still blocked before promotion |
|---|---|---|
| JAX | CPU overlay comparison rows and bounded bridge evidence exist for declared routes. | Mature native arbitrary Phase-QNode lowering, broad transform algebra, and isolated benchmark artefacts. |
| PyTorch | Bounded parameter-shift, custom autograd, `torch.func`, `torch.compile`, module/layer wrapper, and live-overlay evidence exist for declared routes. | Promotion-grade isolated benchmark artefacts and arbitrary simulator/compiler lowering. |
| TensorFlow | Bounded `GradientTape`, `tf.function`, XLA, and Keras-layer evidence exist for declared routes. | Arbitrary Phase-QNode lowering and isolated benchmark artefacts. |
| PennyLane | Export/import and identical-circuit comparison evidence exists for bounded local circuits. | Plugin/provider hardware routes and promotion-grade isolated benchmark evidence. |
| Qiskit | Shifted-circuit generation, local Statevector, finite-shot surrogate, and no-submit provider-gradient/preparation boundary evidence exist. | Live execution, raw-count replay, calibration/statevector comparison, and hardware-gradient artefacts. |
| Enzyme/LLVM/MLIR | MLIR-runtime correctness evidence is attached; the local `enzyme`, `opt`, `mlir-opt`, and `clang` command/version snapshot is recorded; a bounded native LLVM Enzyme scalar derivative probe passes. | Raw Enzyme/MLIR compiler-AD breadth artifacts, derived arbitrary Enzyme/program AD breadth evidence, the Enzyme-JAX external-comparison runtime gap, and isolated benchmark IDs. |
| Catalyst | The external-comparison artefact now emits a Catalyst hard-gap row with qjit/MLIR/QIR workflow scope, compiled differentiation scope, control-flow gaps, finite-shot limitations, and unsupported provider routes. | Configured Catalyst runner success, arbitrary compiled quantum-classical workflow parity, finite-shot jobs, provider submission, hardware execution, and isolated benchmark IDs. |

## Reviewer Procedure

1. Validate the public claim table before quoting differentiable capability
   language.
2. Validate the environment lock before replaying framework comparisons.
3. Validate the artefact bundle before using any evidence file in review.
4. Treat local benchmark rows as regression and functional comparison evidence
   unless the artefact itself reports `isolated_affinity`.

The current package is therefore suitable for reviewer reproduction and gap
triage. It is not yet suitable for state-of-art performance promotion.
