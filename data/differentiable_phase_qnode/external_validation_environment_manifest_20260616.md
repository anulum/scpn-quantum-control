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

- Artefact ID: `differentiable-external-validation-environment-manifest-20260616`
- Classification: `functional_non_isolated`
- Python: `3.12.15`
- Platform: `Linux-7.0.0-42-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `02de8a01fc25158c29aeab5f55edd023727980459ad169668c5b2a29cfdb1b5e` | 0 |
| `requirements.txt` | Runtime dependency lock input | `ebf8fed298a3c59d7d1123c9e1f9d561ba16f6e3790d06344f5f3dd74977a7fe` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `8db72efec1fb0058a2732cb22d82e09e99c047cd42e252cabc343afba31f72dc` | 33 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `ec995d031989afe891bb9b4ef15312ee083231012697b90366028a27708a9d5a` | 17 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `49d15fc1ce62a45129e8892f1f1f3033cabba2061f1b047b2cafb68338725433` | 157 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `d98f0abd12f128786702bb7f5ce632c7ef9570dd9ba193252564ca03580958b0` | 157 |
| `requirements-ci-py315-linux.txt` | Python 3.15 Linux CI lockfile | `e8d56ec80b3e809a4f1dbd8f17a36be5a189a2ca2c9f85625185bdd3a568d4bf` | 157 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
