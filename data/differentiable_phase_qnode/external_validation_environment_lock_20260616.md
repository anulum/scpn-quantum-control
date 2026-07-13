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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `4df6075901e28f74abc2994db40b0575274b98399065bd00f56b22cbe26b5a82` | 0 |
| `requirements.txt` | Runtime dependency lock input | `739228b5225dd2e52ac26aa66ae2d03783b3de0794460b059d8b825717d85f8c` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `201c35c1d64dc527f4b6949d0158ecac1b3a4003ffa17e912d6df901925979cb` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `7dbb41967d1fa074b90e1b6c6801f5c92f09b2edea7aeb6467192764a12758f0` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `026d5e208f3ae6b58a7bf507a7b711e873e89b55a7d03111e30932521a77a9ea` | 156 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `a340cb953b566990c9cc2220bdaa229afc0a20532adcfde07fad9b08728acb75` | 156 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `7e3dda1c70eb8ea38104c5c04b4cced9e8959b9289ba9995ace9376468db8e28` | 156 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |