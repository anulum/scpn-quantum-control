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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `00a182af3d2bce6b43221e9b686165c49d37762a4443b9a0e3191e847e875195` | 0 |
| `requirements.txt` | Runtime dependency lock input | `0abcc72f24fedc10b57533334c5510e39430a67b8567100dc879c8b4107febd0` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `c4ccd5989d909293fc81e7ffa8f8b5c81d68c34feac9aa11dd9bd11f6e824907` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `42be13f99861c241f2fc980e95988cf7f6e9c7b132d8db24e00de92aa6f63464` | 17 |
| `requirements-ci-py310-linux.txt` | Python 3.10 Linux CI lockfile | `416145a6ca9a37196044512b329313e198a243177519cf19c4e3311494a7d6e2` | 157 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `157f169402f7d1edd22b661c20638c8a9cbe6799039fe7199f516ec46b1583a0` | 155 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `f820cf27319f63e0301e84ee3af00fc510c8cf0771357ee7ce0c2a36c82b8943` | 155 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `2d800930f635e2dc54889afa2258197e90a06c3e236bf8032ea30af770d0b4fb` | 155 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
