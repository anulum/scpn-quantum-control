<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable competitive baseline refresh
-->

# Differentiable Competitive Baseline Refresh

- Schema: `scpn_qc_differentiable_competitive_baseline_refresh_v1`
- Artifact ID: `diff-competitive-baseline-refresh-20261011`
- Generated on: `2026-10-11`
- Max age: `45` days
- Claim boundary: Competitive baseline freshness evidence only; it does not promote category-leadership, provider, hardware, GPU, QPU, production-performance, or isolated_affinity claims.

| Baseline | Version/source stream | Checked | Refresh due | Categories | Source |
|---|---|---|---|---|---|
| `jax` | PyPI jax 0.11.2; official docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | jax_native_transforms, benchmark_promotion | [source](https://docs.jax.dev/) |
| `pytorch` | PyPI torch 2.14.1; official stable docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | pytorch_autograd_compile, benchmark_promotion | [source](https://docs.pytorch.org/docs/2.14/index.html) |
| `tensorflow` | PyPI tensorflow 2.21.0; official docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | pytorch_autograd_compile, docs_api_maintainability | [source](https://www.tensorflow.org/guide/autodiff) |
| `pennylane` | PyPI pennylane 0.45.1; official docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | pennylane_qnode_device_plugin, provider_hardware_gradients | [source](https://docs.pennylane.ai/) |
| `qiskit_algorithms` | PyPI qiskit-algorithms 0.4.0; official docs and discontinued-support notice checked 2026-10-11 | `2026-10-11` | `2026-11-25` | qiskit_runtime_provider_gradients, provider_hardware_gradients | [source](https://qiskit-community.github.io/qiskit-algorithms/) |
| `catalyst` | PyPI pennylane-catalyst 0.15.0; official docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | catalyst_compiler_workflows, enzyme_compiler_ad | [source](https://docs.pennylane.ai/projects/catalyst/en/stable/) |
| `enzyme_mlir` | GitHub EnzymeAD/Enzyme v0.0.302; official docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | enzyme_compiler_ad, rust_native_program_ad | [source](https://enzyme.mit.edu/) |
| `julia_ad` | ChainRulesCore.jl v1.26.1; Zygote.jl v0.7.13; Reactant.jl v0.2.291; PyPI juliacall 0.9.36 | `2026-10-11` | `2026-11-25` | rust_native_program_ad, docs_api_maintainability | [source](https://juliadiff.org/ChainRulesCore.jl/stable/) |
| `emerging_ad` | PyPI tinygrad 0.14.0; MLIR EmitC docs checked 2026-10-11 | `2026-10-11` | `2026-11-25` | adoption_licensing, docs_api_maintainability | [source](https://mlir.llvm.org/docs/Dialects/EmitC/) |

This artefact is a freshness gate for comparison baselines. It does not promote any scorecard category; promotion still requires implementation, tests, docs, fresh baseline evidence, claim-ledger promotion, and benchmark artefacts.