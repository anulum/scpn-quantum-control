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
- Platform: `Linux-7.0.0-28-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `35660e01a5aa20c65e54a1d892675f78aaf37fd2204232587aadee07a43dd986` | 0 |
| `requirements.txt` | Runtime dependency lock input | `67d30486ee7b3d478dcdab2c44ed932dada3a2fcda326b2cda425a057bc62618` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `88ed9a13e3ddf447a8db27cbea2f5345b13d294fce24e028718da4a0149b2fb4` | 28 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `73411b493d920d4e3bcba6fdf9bd881b1fa79d4b72c7080df3e76c6a58aeca9a` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `e4184396cf87e2c22905bccd199ac12d078930ff783ff1b8eec740c9408f4cd8` | 157 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `d7c95a30ff728d3fcea5f7739d48c5b10446503c153b7773038a992f73bc9756` | 157 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `3d912d4ef65e5f20965e05abc3861c8222e220a43fde3690279237734e22c81b` | 157 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
