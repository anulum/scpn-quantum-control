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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `141c512cad007b3033225be4b08a8dc3dc7a545417880f97861b5f460754f342` | 0 |
| `requirements.txt` | Runtime dependency lock input | `fa33d0f2d273e0fbcc7878b899637c543a67cca4cd2d639989a4c26ee317f6eb` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `37d893ddfdde255e138fc7d36e28689efc686752be7aed0c8fe4e0b66ca0f59a` | 31 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `73411b493d920d4e3bcba6fdf9bd881b1fa79d4b72c7080df3e76c6a58aeca9a` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `b912c0e4c4370cf77e7d9181fa0927a60c9de505329e2987c8aa4de6327907fb` | 158 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `284449e2b3778706e010bc3a77319f7f33f4811b8f941ab34d1fc4f71db711e2` | 158 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `71dde916385e5cab71ddd87c03d53c439d35da17ed05908eb05e6d097120aa4f` | 158 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
