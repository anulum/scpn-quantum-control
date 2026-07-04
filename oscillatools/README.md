<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# oscillatools

A differentiable, control-oriented toolkit for coupled-phase-oscillator (Kuramoto)
dynamics. It provides the model family, integrators, exact mean-field reductions,
stability and continuation analysis, differentiation, and control primitives as a
light, standalone distribution whose only hard dependencies are **NumPy** and
**SciPy**. Every acceleration or interoperability tier (a Rust engine, Julia, JAX,
PyTorch, matplotlib, scikit-learn) is an optional extra; the pure-Python NumPy
floor always runs.

`oscillatools` is extracted from
[`scpn-quantum-control`](https://github.com/anulum/scpn-quantum-control), where the
toolkit originated, so that the coupled-oscillator surface can be installed without
that project's quantum-provider dependency stack.

## Installation

```bash
pip install oscillatools                 # NumPy + SciPy floor
pip install "oscillatools[rust]"         # optional Rust acceleration engine
pip install "oscillatools[jax]"          # JAX autodiff backend
pip install "oscillatools[torch]"        # PyTorch neural-operator surrogate
pip install "oscillatools[sklearn]"      # scikit-learn estimator interface
pip install "oscillatools[viz]"          # matplotlib renderers
pip install "oscillatools[julia]"        # Julia acceleration tier
```

## Capabilities

- **Model family** — classical, higher-order (hyperedge/simplicial force),
  time-delayed, noisy, inertial (second-order), adaptive-coupling, Sakaguchi,
  Winfree, Stuart–Landau, networked/multiplex, and next-generation QIF neural-mass
  Kuramoto dynamics.
- **Integrators** — explicit Euler, RK4, DOPRI adaptive step, symplectic (inertial),
  method-of-steps (delayed), and Euler–Maruyama (stochastic).
- **Exact reductions** — Ott–Antonsen and Watanabe–Strogatz manifolds, and the
  noisy/finite-width mean-field order-parameter theory.
- **Analysis** — order parameters and Daido harmonics, linear-stability spectra,
  Lyapunov exponents, pseudo-arclength continuation, saddle-node/fold location, and
  basin-stability estimation.
- **Differentiation** — a hand-written reverse-mode adjoint and forward sensitivity
  over the integrators (NumPy and, optionally, JAX), for gradient-based design.
- **Control** — optimal coupling design, pinning control, coordinated reset,
  SDRE/receding-horizon control, and system identification.
- **Interoperability** — a scikit-learn-style estimator interface and a
  NumPy-array-first API.

A generated capability handbook and manifest enumerate the full public surface.

## Optional acceleration and interoperability

The Rust acceleration engine ships as the separate
[`scpn-quantum-engine`](https://pypi.org/project/scpn-quantum-engine/) wheel and is
used automatically when installed (`[rust]` extra). Without it, the dispatchers fall
through to the NumPy implementation. Measured tier selection is reported per call.

## Licensing

`oscillatools` is dual-licensed under AGPL-3.0-or-later with a commercial license
available. See [`NOTICE.md`](NOTICE.md) and
[`LICENSES/AGPL-3.0-or-later.txt`](LICENSES/AGPL-3.0-or-later.txt). For commercial
licensing: protoscience@anulum.li.

## Citation

If you use `oscillatools`, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).
