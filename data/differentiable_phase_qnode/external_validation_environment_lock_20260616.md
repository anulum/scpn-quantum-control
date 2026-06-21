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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `6e3566be3e652fc8c6f60bca06d225fdaa93a3cbe2c66aec7721a93acbc43bc2` | 0 |
| `requirements.txt` | Runtime dependency lock input | `0abcc72f24fedc10b57533334c5510e39430a67b8567100dc879c8b4107febd0` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `c4ccd5989d909293fc81e7ffa8f8b5c81d68c34feac9aa11dd9bd11f6e824907` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `42be13f99861c241f2fc980e95988cf7f6e9c7b132d8db24e00de92aa6f63464` | 17 |
| `requirements-ci-py310-linux.txt` | Python 3.10 Linux CI lockfile | `1335b6badc28f0195928d655f4c7a41631c2b60178e798329e481edf79cfe6c2` | 157 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `f339f47ef39e1490074064bbe10d87fbc97050043c667bd4f6629340876c7922` | 155 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `fca0774e8a50b05c3f5021e30171cfd58e508ac885e794f40fbeaddbc68cf6e1` | 155 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `33b1f0b10d7cb78c7ec4ad1db8d3a30d7ffc4dbe1bbe4f68c9b00fc4c08b239f` | 155 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
