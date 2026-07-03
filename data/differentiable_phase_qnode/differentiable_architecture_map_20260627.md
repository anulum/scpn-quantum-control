<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable Architecture and Rustification Map
-->

# Differentiable Architecture and Rustification Map

- Schema: `scpn_qc_differentiable_architecture_map_v1`
- Artifact ID: `diff-architecture-rustification-map-20260627`
- Rustification ready: `False`
- Ready layers: `0/6`
- Claim boundary: Differentiable architecture and Rustification routing map only; no broad Rustification promotion, provider execution, hardware execution, GPU execution, LLVM/JIT execution, or isolated benchmark claim is implied.

| Layer | Inventory rows | SOTA categories | Blockers | Next rounds |
|---|---|---|---|---|
| `public_api_facade` | unified_differentiable_api | docs_api_maintainability<br>adoption_licensing | public orchestration remains Python-first<br>dashboard rows still include metadata-only and blocked routes | Round 0 surface integrity and maintainability<br>Round 5 Rustification readiness |
| `qnode_framework_bridges` | pennylane_plugin_matrix<br>qiskit_runtime_provider_gradients | jax_native_transforms<br>pytorch_autograd_compile<br>pennylane_qnode_device_plugin<br>qiskit_runtime_provider_gradients | provider-plugin execution artefacts are missing<br>hardware-plugin and provider-gradient parity artefacts are missing<br>live-ticket Runtime/QPU evidence is missing<br>provider-gradient methods are not attached to a live-ticket run | Round 0 surface integrity and maintainability<br>Round 5 Rustification readiness |
| `program_ad_core` | rust_program_ad_ir<br>whole_program_frontend | rust_native_program_ad | non-lowered dynamic indexing semantics, non-sum/mean reduction families, and broad linalg array adjoints are missing<br>executable registry promotion, LLVM/JIT lowering, provider/hardware evidence, and isolated benchmark evidence are missing<br>static Python source/bytecode metadata has no executable Rust parity requirement<br>compiler lowering remains blocked until a real Program AD backend exists | Round 0 surface integrity and maintainability<br>Round 5 Rustification readiness |
| `compiler_ad_native_execution` | rust_compiler_ad_primitives<br>enzyme_mlir_compiler_ad<br>catalyst_compiler_comparison | catalyst_compiler_workflows<br>enzyme_compiler_ad | isolated compiler-AD benchmark ID is missing<br>broad LLVM/JIT lowering remains claim-blocked<br>11-case compiler-AD breadth evidence is incomplete<br>isolated Enzyme/MLIR benchmark attachment is missing<br>configured Catalyst qjit/MLIR/QIR runner evidence is missing<br>compiled quantum-classical workflow parity is unimplemented | Round 0 surface integrity and maintainability<br>Round 5 Rustification readiness |
| `provider_hardware_boundary` | qiskit_runtime_provider_gradients<br>hardware_gradient_campaigns | provider_hardware_gradients<br>qiskit_runtime_provider_gradients | live-ticket Runtime/QPU evidence is missing<br>provider-gradient methods are not attached to a live-ticket run<br>live-ticket hardware execution is missing<br>raw counts, calibration, and reference simulator attachments are missing | Round 0 surface integrity and maintainability<br>Round 5 Rustification readiness |
| `benchmark_and_claim_governance` | differentiable_sota_scorecard | benchmark_promotion<br>docs_api_maintainability<br>adoption_licensing | governance evidence only; no executable Rust surface is required | Round 0 surface integrity and maintainability<br>Round 5 Rustification readiness |

This map is routing evidence for the Rustification queue. It does not promote Rust, LLVM/JIT, provider, hardware, GPU, performance, or isolated benchmark claims.
