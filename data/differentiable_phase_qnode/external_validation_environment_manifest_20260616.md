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
- Python: `3.12.14`
- Platform: `Linux-7.0.0-34-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `f4be8f59adf971790fa70c1fe75797aacc6aca2f133cfdcfb67305375765cf3c` | 0 |
| `requirements.txt` | Runtime dependency lock input | `fa33d0f2d273e0fbcc7878b899637c543a67cca4cd2d639989a4c26ee317f6eb` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `18434ac1023426a65605f3f02e0a0aa83e1a27a7e4c4ebb11b88b348a5bcfc6c` | 31 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `73411b493d920d4e3bcba6fdf9bd881b1fa79d4b72c7080df3e76c6a58aeca9a` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `38b5a98f6cd3029a3b766bd1d04d06c433b0c200d8b7da0bdb919d0fc76a72b0` | 157 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `0506a54097e39743dd7a29636908d2064371340dfe22953ddbbad9a6bb67bdf7` | 157 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `fe1d99842d1345c791aba12e0a7a9a4ed4c9c9c617d56ed3322ed597e484456b` | 157 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
