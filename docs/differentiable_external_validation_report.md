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
otherwise. The report does not claim hardware execution, provider execution,
QPU execution, GPU execution, production performance, quantum advantage, or
`isolated_affinity` benchmark status.

The public wording source is
`data/differentiable_phase_qnode/public_claim_table_20260616.md`. The reviewer
ledger is `data/differentiable_phase_qnode/claim_ledger.md`.

## Reproducibility Package

| Artefact | Role | Boundary |
|---|---|---|
| `data/differentiable_phase_qnode/external_validation_artifact_bundle_20260616.json` | SHA-256 manifest over committed differentiable evidence files. | Checksum provenance only. |
| `data/differentiable_phase_qnode/external_validation_environment_lock_20260616.json` | Exact runtime, developer, CI, framework-overlay, and Enzyme-runner lockfile manifest. | Reviewer reproduction only. |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/diff-qnode-external-comparison.json` | Bounded JAX, PyTorch, TensorFlow, PennyLane, and Enzyme comparison rows. | Functional non-isolated comparison evidence. |
| `data/differentiable_phase_qnode/identical_circuit_gradient_comparison_20260616.json` | Same-circuit gradient comparison artefact for Qiskit and PennyLane routes. | Correctness comparison, not hardware execution. |
| `data/differentiable_phase_qnode/domain_benchmark_dataset_closure_20260616.json` | Exact-answer domain dataset closure artefact. | Dataset validation, not production benchmark promotion. |
| `data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json` | Enzyme/MLIR maturity audit with MLIR-runtime correctness evidence, native Enzyme scalar evidence, and missing raw compiler-AD breadth artifact plus derived breadth evidence recorded as hard gaps. | Hard-gap evidence, not Enzyme parity promotion. |

## Provider Comparison Status

| Provider family | Current SCPN evidence | Still blocked before promotion |
|---|---|---|
| JAX | CPU overlay comparison rows and bounded bridge evidence exist for declared routes. | Mature native arbitrary Phase-QNode lowering, broad transform algebra, and isolated benchmark artefacts. |
| PyTorch | Bounded parameter-shift, custom autograd, `torch.func`, `torch.compile`, module/layer wrapper, and live-overlay evidence exist for declared routes. | Promotion-grade isolated benchmark artefacts and arbitrary simulator/compiler lowering. |
| TensorFlow | Bounded `GradientTape`, `tf.function`, XLA, and Keras-layer evidence exist for declared routes. | Arbitrary Phase-QNode lowering and isolated benchmark artefacts. |
| PennyLane | Export/import and identical-circuit comparison evidence exists for bounded local circuits. | Plugin/provider hardware routes and promotion-grade isolated benchmark evidence. |
| Qiskit | Shifted-circuit generation, local Statevector, finite-shot surrogate, and no-submit preparation evidence exist. | Live execution, raw-count replay, calibration/statevector comparison, and hardware-gradient artefacts. |
| Enzyme/LLVM/MLIR | MLIR-runtime correctness evidence is attached; the local `enzyme`, `opt`, `mlir-opt`, and `clang` command/version snapshot is recorded; a bounded native LLVM Enzyme scalar derivative probe passes. | Raw Enzyme/MLIR compiler-AD breadth artifacts, derived arbitrary Enzyme/program AD breadth evidence, the Enzyme-JAX external-comparison runtime gap, and isolated benchmark IDs. |

## Reviewer Procedure

1. Validate the public claim table before quoting differentiable capability
   language.
2. Validate the environment lock before replaying framework comparisons.
3. Validate the artefact bundle before using any evidence file in review.
4. Treat local benchmark rows as regression and functional comparison evidence
   unless the artefact itself reports `isolated_affinity`.

The current package is therefore suitable for reviewer reproduction and gap
triage. It is not yet suitable for state-of-art performance promotion.
