<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable Rust/Python Surface Inventory
-->

# Differentiable Rust/Python Surface Inventory

- Schema: `scpn_qc_differentiable_rust_python_inventory_v1`
- Artifact ID: `diff-rust-python-inventory-20260620`
- Rustification ready: `False`
- Ready surfaces: `0/9`
- Claim boundary: Differentiable Rust/Python surface inventory for rustification planning; no broad rustification promotion, provider execution, hardware execution, LLVM/JIT execution, GPU execution, or isolated benchmark claim is implied.

| Surface | Classification | Rust parity | Polyglot | Benchmark | Blockers |
|---|---|---|---|---|---|
| `unified_differentiable_api` | `python_reference` | `partial` | `partial` | `functional_non_isolated` | public orchestration remains Python-first<br>dashboard rows still include metadata-only and blocked routes |
| `rust_program_ad_ir` | `rust_backed` | `partial` | `partial` | `functional_non_isolated` | primitive-family replay and array adjoints are missing<br>registry metadata mirror, LLVM/JIT lowering, and isolated benchmark evidence are missing |
| `rust_compiler_ad_primitives` | `rust_backed` | `partial` | `partial` | `functional_non_isolated` | isolated compiler-AD benchmark ID is missing<br>broad LLVM/JIT lowering remains claim-blocked |
| `differentiable_sota_scorecard` | `metadata_only` | `not_applicable` | `not_applicable` | `not_applicable` | governance evidence only; no executable Rust surface is required |
| `pennylane_plugin_matrix` | `provider_blocked` | `not_applicable` | `partial` | `blocked` | provider-plugin execution artefacts are missing<br>hardware-plugin and provider-gradient parity artefacts are missing |
| `qiskit_runtime_provider_gradients` | `provider_blocked` | `not_applicable` | `partial` | `blocked` | live-ticket Runtime/QPU evidence is missing<br>provider-gradient methods are not attached to a live-ticket run |
| `hardware_gradient_campaigns` | `hardware_blocked` | `not_applicable` | `partial` | `blocked` | live-ticket hardware execution is missing<br>raw counts, calibration, and reference simulator attachments are missing |
| `catalyst_compiler_comparison` | `deprecate_before_promotion` | `partial` | `missing` | `blocked` | dedicated Catalyst comparison row is missing<br>compiled quantum-classical workflow parity is unimplemented |
| `enzyme_mlir_compiler_ad` | `compiler_native_not_rust` | `partial` | `partial` | `functional_non_isolated` | 11-case compiler-AD breadth evidence is incomplete<br>isolated Enzyme/MLIR benchmark attachment is missing |

Rows are planning evidence for the rustification queue. A row becoming `rustification_ready` does not by itself promote public performance, provider, hardware, GPU, LLVM/JIT, or isolated benchmark claims.
