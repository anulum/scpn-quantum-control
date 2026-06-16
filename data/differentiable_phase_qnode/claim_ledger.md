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

Bounded language: the differentiable lane is SOTA-candidate unless isolated CI benchmark evidence and external comparison artefacts pass.
