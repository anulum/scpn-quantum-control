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

- Schema: `scpn_qc_differentiable_dependency_environment_map_v2`
- Artifact ID: `diff-dependency-environment-map-20260627`
- Environment ready: `False`
- Ready profiles: `4/5`
- Ready evidence rows: `10/16`
- Claim boundary: Differentiable dependency and environment evidence map only; no dependency or benchmark promotion, framework parity promotion, provider execution, hardware execution, GPU execution, Enzyme promotion, or isolated benchmark claim is implied.

| Profile | Status | Lockfiles | Pinned packages | Blockers |
|---|---|---|---|---|
| `runtime_baseline` | `locked` | requirements.txt | 11 | none |
| `development_verification` | `locked` | requirements-dev.txt | 28 | none |
| `ci_python_matrix` | `locked` | requirements-ci-cross-platform-smoke.txt<br>requirements-ci-py311-linux.txt<br>requirements-ci-py312-linux.txt<br>requirements-ci-py313-linux.txt | 491 | none |
| `framework_overlay_cpu` | `locked` | data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt | 54 | none |
| `enzyme_runner_py39` | `hard_gap` | data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt | 10 | Enzyme/JAX runner lockfiles exist, but native Enzyme/LLVM/MLIR toolchain execution remains explicit hard-gap evidence until configured runner artefacts pass. |

| Evidence | Category | Classification | Status | Versions/constraints | Blockers |
|---|---|---|---|---|---|
| `python_versions` | `toolchain` | `locked_versions` | `locked` | python==3.11<br>python==3.12<br>python==3.13<br>enzyme-runner-python==3.9 | none |
| `rust_crates` | `toolchain` | `locked_versions` | `locked` | anyhow==1.0.104<br>chrono==0.4.45<br>clap==4.6.3<br>criterion==0.5.1<br>scpn-quantum-engine==0.2.0<br>pyo3==0.29.0<br>ndarray==0.16.1<br>numpy==0.29.0<br>nalgebra==0.35.0<br>num-complex==0.4.6<br>rand==0.10.2<br>rayon==1.11.0<br>reqwest==0.13.4<br>scpn-quantum-program-ad-replay==0.1.0<br>serde==1.0.229<br>serde_json==1.0.151 | none |
| `jax_cpu` | `toolchain` | `locked_versions` | `locked` | jax==0.10.1<br>jaxlib==0.10.1 | none |
| `pytorch_cpu` | `toolchain` | `locked_versions` | `locked` | torch==2.12.0+cpu | none |
| `tensorflow_cpu` | `toolchain` | `locked_versions` | `locked` | tensorflow_cpu==2.21.0 | none |
| `pennylane_cpu` | `toolchain` | `locked_versions` | `locked` | pennylane==0.45.0<br>pennylane_lightning==0.45.0 | none |
| `qiskit` | `toolchain` | `locked_versions` | `locked` | qiskit==2.5.0<br>qiskit-aer==0.17.2<br>qiskit-qasm3-import==0.6.0<br>qiskit-ibm-runtime==0.47.0 | none |
| `catalyst` | `toolchain` | `locked_versions` | `locked` | pennylane-catalyst==0.15.0<br>catalyst==0.15.0 | none |
| `enzyme_llvm_mlir` | `toolchain` | `locked_versions` | `locked` | enzyme-ad==0.0.6<br>Enzyme LLVM plugin 0.0.79<br>LLVM==18.1.3<br>MLIR==18.1.3 | none |
| `gpu_overlay` | `toolchain` | `declared_unlocked` | `hard_gap` | cupy-cuda12x>=13.0<br>jax[cuda12]>=0.4.30<br>torch>=2.2,<3.0 | The optional CUDA requirements are constrained but have no exact GPU lock or compatible modern-GPU execution artefact. |
| `local_cpu` | `execution_route` | `locally_runnable` | `locked` | n/a | none |
| `jarvislabs_cloud` | `execution_route` | `cloud_only` | `hard_gap` | n/a | JarvisLabs is a dispatch plan only until returned CUDA, XLA, pmap, device, and isolated-benchmark artefacts validate. |
| `provider_execution` | `execution_route` | `provider_only` | `hard_gap` | n/a | Provider execution requires an approved provider workflow and captured provider artefacts. |
| `hardware_ticket` | `execution_route` | `hardware_ticket_only` | `hard_gap` | n/a | Live hardware execution requires an owner-approved ticket, job metadata, raw counts, calibration, and replay evidence. |
| `gtx1060_workstation` | `execution_route` | `unsupported_local_hardware` | `hard_gap` | n/a | The GTX 1060 is explicitly incompatible with the modern CUDA and multi-device promotion route. |
| `ml350_isolated` | `execution_route` | `isolated_host_only` | `hard_gap` | n/a | ML350 bounded CPU evidence exists, but promotion-grade isolated reruns remain blocked on the host-readiness and RAM gate. |

This map is dependency and environment evidence only. It does not promote framework parity, Enzyme/MLIR parity, provider execution, hardware execution, GPU execution, performance, or isolated benchmark claims.
