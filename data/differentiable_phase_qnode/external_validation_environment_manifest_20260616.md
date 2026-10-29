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
- Platform: `Linux-7.0.0-34-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `dd0218c1b02e382ac60796e7031228d91eb324cd6da687683575a7a25fb0c550` | 0 |
| `requirements.txt` | Runtime dependency lock input | `fa33d0f2d273e0fbcc7878b899637c543a67cca4cd2d639989a4c26ee317f6eb` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `abc35b3c04d1b415ee4d5b2776a2522ef14d3117786df6e25f8ab1147e72dd52` | 33 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `73411b493d920d4e3bcba6fdf9bd881b1fa79d4b72c7080df3e76c6a58aeca9a` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `c6aa0b6022914ef1b46c6401149b0cb8a339437b9ed269dc5003a7b6c8d54c39` | 157 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `12cbd7b4c37ccc08360789019d76915e821032a4f2d21c348d822a9448eedd37` | 157 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `b76cff67931f372d0d28dd3bc2d24bd84f36cd153dd220946fdde6269da45496` | 157 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |