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
- Python: `3.12.3`
- Platform: `Linux-7.0.0-31-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `6687519e01994267f2841f70537a794326ba08b4a753d26b0877bc8b7bb5d44a` | 0 |
| `requirements.txt` | Runtime dependency lock input | `67d30486ee7b3d478dcdab2c44ed932dada3a2fcda326b2cda425a057bc62618` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `c95c5dceba3f04f1c4b1153174ab9315f42f09918d432bc21e0ce7132b5f8e87` | 29 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `73411b493d920d4e3bcba6fdf9bd881b1fa79d4b72c7080df3e76c6a58aeca9a` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `3f3bb3299ef9a49dd940fb73e67629369a33c7d5662b307366fabfdc3eb98a61` | 158 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `f31f9322e51830b7b8061d463ecac4ca9a4e4c1b6a266edd97b0a7975c9f330d` | 158 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `73e804b604c49a0fbb1891617b8502b3406d9b9df051c2ec5a406347e430188c` | 158 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
