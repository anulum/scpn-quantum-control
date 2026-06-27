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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `424cfef0ce5b12dddbafadf80ac4354c96cb9a13df87739dcc356e69af324bd0` | 0 |
| `requirements.txt` | Runtime dependency lock input | `1b6003edef261b2877b6de9ebef9c5a23441428d7b65d8becd6fc12e83ead3ce` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `78c1b9ab666af031ef2620fc3e7753c81a83f75a7818adbcd08e04208e82fea4` | 27 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `5817fd38f104b55dd82864e6b51cbf4ed96054f0533eb6a1ee81612112919f3c` | 19 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `157f169402f7d1edd22b661c20638c8a9cbe6799039fe7199f516ec46b1583a0` | 155 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `f820cf27319f63e0301e84ee3af00fc510c8cf0771357ee7ce0c2a36c82b8943` | 155 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `2d800930f635e2dc54889afa2258197e90a06c3e236bf8032ea30af770d0b4fb` | 155 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |