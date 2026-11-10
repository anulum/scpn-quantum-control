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
| `pyproject.toml` | Package metadata and bounded dependency ranges | `59c87002be4109b71619ee52ae4f8b706fc45bf083fecfd3c2c48cb0401df002` | 0 |
| `requirements.txt` | Runtime dependency lock input | `9010e7a298032aee39afc563b211c84270f883a2fed032be361294aba87aa245` | 11 |
| `requirements-dev.txt` | Developer verification dependency lock input | `8db72efec1fb0058a2732cb22d82e09e99c047cd42e252cabc343afba31f72dc` | 33 |
| `requirements-ci-cross-platform-smoke.txt` | Cross-platform smoke CI lockfile | `e81114467e033f179cd6f655c83f1f9ee1231324b0b716d2f68409696aa52864` | 17 |
| `requirements-ci-py312-linux.txt` | Python 3.12 Linux CI lockfile | `f678b0ed3c5807a6e3ebb2b344eea5f625e5199ac28d07351ebd2263af01e9cb` | 157 |
| `requirements-ci-py313-linux.txt` | Python 3.13 Linux CI lockfile | `85054f036589847b45b850d709aaf0c3f792cebc07c027bae41f4c4938163f05` | 157 |
| `requirements-ci-py315-linux.txt` | Python 3.15 Linux CI lockfile | `7bebac844ffe2e793cf8fcb06e9217e466a2640a7279473fbc863d0c515833b7` | 157 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/framework_overlay_freeze.txt` | CPU framework overlay freeze used for JAX, PyTorch, TensorFlow, and PennyLane rows | `11a15a483d2f8f602b8d052dc1cf0824d37a86a47853a66b1cda1ed93caa56c6` | 54 |
| `data/differentiable_phase_qnode/local_benchmark_20260616T0955Z/enzyme_py39_freeze.txt` | Python 3.9 Enzyme/JAX runner freeze used for installed-toolchain hard-gap evidence | `2770738675e8ac3fbf3edd5f8b004a3c0d2621fd3324b77aa3a238437b947d32` | 10 |
