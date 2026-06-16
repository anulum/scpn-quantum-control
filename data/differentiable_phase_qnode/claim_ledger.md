<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable Phase-QNode Claim Ledger
-->

# Differentiable Phase-QNode Claim Ledger

| Claim | Status | Artefact IDs | Benchmark IDs | Known gaps |
|---|---|---|---|---|
| framework_overlay_parity | SOTA-candidate | diff-qnode-framework-overlay-profile-v1 | diff-qnode-framework-overlay-profile-v1 | Local machines must still run the manifest install command before parity can execute.<br>The overlay is CPU-only and intentionally excludes the repository jax[cuda12] extra. |
| ci_benchmark_evidence | SOTA-candidate | diff-qnode-ci-evidence-schema-v1 | diff-qnode-ci-evidence-schema-v1 | GitHub-hosted runners are downgraded to functional_non_isolated.<br>Production isolated_affinity promotion requires a self-hosted isolated-benchmark runner artefact.<br>Accelerator benchmark claims require explicit CUDA/ROCm device metadata; missing visible devices are classified as silent_accelerator_fallback. |
| external_framework_comparison | SOTA-candidate | diff-qnode-external-comparison-schema-v1 | diff-qnode-external-comparison-schema-v1 | Missing framework dependencies are hard_gap rows.<br>Unconfigured LLVM/Enzyme tooling remains dependency_missing; configured runners must pass strict JSON, timeout, toolchain, and correctness gates. |
| phase_qnode_claim_boundary | SOTA-candidate | diff-qnode-claim-ledger-v1 | diff-qnode-claim-ledger-v1 | The lane remains SOTA-candidate until isolated CI benchmark and external comparison artefacts pass.<br>Provider and QPU execution remain explicitly outside this claim. |
| support_surface_alignment | SOTA-candidate | diff-support-surface-alignment-audit-v1 | diff-support-surface-alignment-audit-v1 | The audit checks committed generated-manifest inventory and does not replace isolated benchmark artefacts.<br>It validates support-surface wiring, not provider hardware execution or production performance. |
| hardening_slice_gate | SOTA-candidate | diff-hardening-slice-gate-v1 | diff-hardening-slice-gate-v1 | The gate builds an auditable verification checklist but does not execute shell commands.<br>It checks benchmark-classification invariants and does not upgrade functional_non_isolated rows to isolated_affinity evidence. |
| module_hardening_audit | SOTA-candidate | diff-module-hardening-audit-v1 | diff-module-hardening-audit-v1 | The audit verifies inventory, tests, and declared diagnostics; it does not prove full formal correctness.<br>Provider execution, hardware execution, and isolated benchmark promotion remain separate evidence gates. |
| external_validation_environment_lock | SOTA-candidate | diff-external-validation-environment-lock-20260616 | diff-external-validation-environment-lock-20260616 | The lock manifest records reproducibility inputs only and does not execute external validation jobs.<br>The artefact remains functional_non_isolated and cannot promote benchmark or hardware claims. |
| public_claim_table | SOTA-candidate | diff-public-claim-table-20260616 | diff-public-claim-table-20260616 | The table constrains wording but does not execute external validation or isolated benchmarks.<br>Rows remain bounded candidates unless the claim ledger promotes them with passing evidence. |
| external_validation_artifact_bundle | SOTA-candidate | diff-external-validation-artifact-bundle-20260616 | diff-external-validation-artifact-bundle-20260616 | The manifest records committed evidence checksums but does not bundle private or ignored artefacts.<br>The artefact remains functional_non_isolated and cannot promote benchmark or hardware claims. |
| external_validation_technical_report | SOTA-candidate | diff-external-validation-technical-report-20260616 | diff-external-validation-technical-report-20260616 | The report is descriptive and does not execute external validation jobs or isolated benchmarks.<br>Rows remain bounded candidates unless the claim ledger promotes them with passing evidence. |

Bounded language: the differentiable lane is SOTA-candidate unless isolated CI benchmark evidence and external comparison artefacts pass.
