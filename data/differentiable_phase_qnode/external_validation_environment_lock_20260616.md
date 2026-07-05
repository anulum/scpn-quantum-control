<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Differentiable external-validation environment lock
-->

# Differentiable External-Validation Environment Lock

- Artefact ID: `diff-external-validation-environment-lock-20260616`
- Classification: `functional_non_isolated`
- Python: `3.12.3`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `f29141ead7e6df9e0c08731b3c0f72c0060a8d1be6238cf7c206377fc7359bda` | 0 |
| `requirements.txt` | Runtime dependency lock input | `31ef4f50c62078f89052d4e6dc8081f70cdff6715231a3f1f8e4b6be508dc886` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `fac6b5c68ad210d635ab1f759a2af603d60c2a14a3c7e246eb4ee7cb93d6b870` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `114d6ad85f736daefd3307bad5ab77483d404031331232b7c5db111d2a82caa5` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `52d27854428cb771fe3dd7f4ac4821a3e8c13ef6b19288287d329d07c3e96616` | 156 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `d37732fc987171990d9c6a3bf279e15b91d0e09063f9039cfaacbbec3ea3e126` | 156 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `c716d3c5325c8f14891dc7a8cd4f297054c0332176c68b4d2a30c0e5f8d02c5f` | 156 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
