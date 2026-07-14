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
- Python: `3.12.13`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `6730fbd347ced82ef5d870793b34cf26ac66bc78dc54aefc0b7891d545af2e74` | 0 |
| `requirements.txt` | Runtime dependency lock input | `739228b5225dd2e52ac26aa66ae2d03783b3de0794460b059d8b825717d85f8c` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `d5586f0157b889bf8037214df6d2d7500bd0c90ebf9ef05c8e36fe1cf281d231` | 28 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `7dbb41967d1fa074b90e1b6c6801f5c92f09b2edea7aeb6467192764a12758f0` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `4c7d06efcf6956c6aec2e0f305cf063503993561d7c54c0f0d7d722a3dc3e522` | 158 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `62a83c5e005282fa7e2a71d81779771dfbdd28a59835b35e71f02deb98318562` | 158 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `af1cdfe5833ef159f73dbd352ca9e151606ad51d7624b83147e373883ebc9786` | 158 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
