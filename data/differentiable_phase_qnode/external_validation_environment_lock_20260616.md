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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `2b9c34c32bd159925b3727f640a5ecda507cf87a9874a78a6ce9f4a069da353b` | 0 |
| `requirements.txt` | Runtime dependency lock input | `9a3ec3b57c0d1c0bd8647e82dea8ddcf54b94f02be3992a94bff39edb3a78e12` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `0c9e3e9bcc484c94d7776d50554aa65f81f744f1f646c8958cbf8fb7bd460b7e` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `13a2b72471e6bf1cdc41d1c9807f5ad6a3cca71523f03c711c7d372d0507c110` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `4ed2fcf388b6e65e694adb68de6a7585aa9c8af947e18df6492dce5610136d81` | 156 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `6edccd08bfe1f8dcf8c0a8ad34183ecc7c4047ab4487b5213a88b60b81e48087` | 156 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `acdf851e04a72aab42b27d7bf815b0fd1afa2d046c16bde2a5db511e734c77e1` | 156 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |