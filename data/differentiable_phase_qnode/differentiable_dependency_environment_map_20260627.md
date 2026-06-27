<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable Dependency and Environment Evidence Map
-->

# Differentiable Dependency and Environment Evidence Map

- Schema: `scpn_qc_differentiable_dependency_environment_map_v1`
- Artifact ID: `diff-dependency-environment-map-20260627`
- Environment ready: `False`
- Ready profiles: `4/5`
- Claim boundary: Differentiable dependency and environment evidence map only; no dependency or benchmark promotion, framework parity promotion, provider execution, hardware execution, GPU execution, Enzyme promotion, or isolated benchmark claim is implied.

| Profile | Status | Lockfiles | Pinned packages | Blockers |
|---|---|---|---|---|
| `runtime_baseline` | `locked` | requirements.txt | 11 | none |
| `development_verification` | `locked` | requirements-dev.txt | 27 | none |
| `ci_python_matrix` | `locked` | requirements-ci-cross-platform-smoke.txt<br>requirements-ci-py311-linux.txt<br>requirements-ci-py312-linux.txt<br>requirements-ci-py313-linux.txt | 484 | none |
| `framework_overlay_cpu` | `locked` | data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt | 54 | none |
| `enzyme_runner_py39` | `hard_gap` | data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt | 10 | Enzyme/JAX runner lockfiles exist, but native Enzyme/LLVM/MLIR toolchain execution remains explicit hard-gap evidence until configured runner artefacts pass. |

This map is dependency and environment evidence only. It does not promote framework parity, Enzyme/MLIR parity, provider execution, hardware execution, GPU execution, performance, or isolated benchmark claims.
