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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `33c12f682479f3712a58a44558de82d28976a49eba4f209e6d9100df7563fb2f` | 0 |
| `requirements.txt` | Runtime dependency lock input | `9861cad54a8eb39dba41eebfe886a097529969444384f7ace56de3926e27b809` | 10 |
| `requirements-dev.txt` | Developer verification dependency lock input | `9f8355a32fa6762931df15257bfa38a6304e83bd44ed008f8e45cb4a678ee01f` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `1bdb1fe6fca2e94f9d379af7d1bb3df223d0d38d4b3fa126932e135fbaa49a89` | 17 |
| `requirements-ci-py310-linux.txt` | Python 3.10 Linux CI lockfile | `631f57a206dafcb53c6423db8d903a8e25748b502cc1f5c6be1bf561e20e2d6c` | 157 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `a4e8cfa3d5ba52f448c063f9dc401f6ecb46a29ecd5430911afbcb239ec479f8` | 155 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `d2ecd9238ee3f5e1b730edac4aabd99dc03f6b4b85d7e5e1a04717b2d902527d` | 155 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `f213a5d4ab2d3cfbd99c950d0cfa7bff79d57a5def584e0567bec86647cf8afc` | 155 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
