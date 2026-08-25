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
- Platform: `Linux-7.0.0-28-generic-x86_64-with-glibc2.39`
- Claim boundary: Exact environment lockfile manifest for reviewer reproduction only; it does not promote performance, provider, QPU, GPU, hardware, or isolated_affinity benchmark claims.

| Lockfile | Role | SHA-256 | Pinned packages |
|---|---|---|---|
| `pyproject.toml` | Package metadata and bounded dependency ranges | `35660e01a5aa20c65e54a1d892675f78aaf37fd2204232587aadee07a43dd986` | 0 |
| `requirements.txt` | Runtime dependency lock input | `67d30486ee7b3d478dcdab2c44ed932dada3a2fcda326b2cda425a057bc62618` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `5cc61b41d8eed90e3d8dd1ca93763fecef4a7f795e566f89f28151f4c4228b0e` | 28 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `6d3e3afd3c60251a6b0e858c338e0ddc14a7e31948f0811fccd07efb395b181d` | 17 |
| `requirements-ci-py311-linux.txt` | Python 3.11 Linux CI lockfile | `c3384a5df0ce28a051eaada0d125e60339bd3d0b94e177787ad3aebc439d5a2b` | 157 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `c798f6d219d99e7f4aa945667c42d6081dac46cb37140d44b35ea2eda6837f9f` | 157 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `7b9ee8bacca351db197e200b4e4ed4d5a6ade20a194ade4c319037acef4f4b07` | 157 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
