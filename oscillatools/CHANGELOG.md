<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Changelog

All notable changes to `oscillatools` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `train_kuramoto_neural_operator` initialises its layers from a model-local
  `torch.Generator` seeded by `seed`, on the active default device, with the
  same draw order as `nn.Linear`. Training no longer draws from or advances the
  global PyTorch random state, and it runs on the declared `torch>=2.2` floor.

### Changed

- `mpc_optimum_parameter_sensitivity` and `mpc_plan_energy_gradient` refuse a
  horizon solve whose stationarity residual exceeds `residual_tolerance`
  (default `1e-6`) before the pullback, instead of returning the implicit
  gradient at an unconverged point. A caller that accepts a looser solve
  passes a larger tolerance; `MpcOptimumSensitivity.residual_tolerance`
  records it.

### Added

- `evaluate_control_under_disturbances` (`ControlDisturbance`,
  `DisturbedControlOutcome`, `ControlRobustnessReport`) scores a fixed
  open-loop network control on held-out plant disturbances against the same
  plants uncontrolled, checks the amplitude bound and lists losing cases.
- `infer_network_bayesian` reports `parameter_covariances`, `data_rank`,
  `unidentified_directions` and `identifiable`; a rank-deficient design is no
  longer hidden by a ridge, and a noise-free series is refused.
- `fit_with_held_out` and `evaluate_held_out` (`InverseProblemSplit`,
  `HeldOutEvaluation`) fit on a training segment and score one-step
  predictive intervals on disjoint calibration and test segments.
- `noisy_expectation_value_and_grad` estimates `E[L]` and `∇E[L]` over
  independent noise seeds, with the standard error of each mean
  (`NoisyExpectationEstimate`, `EXPECTATION_ESTIMAND`); one path's gradient
  remains `noisy_terminal_value_and_grad`.
- `jax_kuramoto_delayed_hessian_vector_product` returns a Hessian-vector
  product of a terminal objective through the integer-delay method-of-steps
  map (JAX extra).
- `sparse_coupling_from_edges` builds a sparse coupling from an explicit edge
  list. Its declared duplicate policy is `sum` or `refuse`, and
  `sparse_coupling_from_scipy` accepts the same `duplicates` argument.
- `SparseKuramotoCoupling.storage_report()` measures the edge-linear stored
  arrays.
- `sparse_kuramoto_execution_report()` records CPU float64 placement. It reports
  no GPU path, so the sparse kernels never stand in for GPU evidence.

## [0.1.1] - 2026-10-29

### Fixed

- Preserve solver time and termination contracts in fixed-step, delayed,
  adaptive and differentiable Kuramoto integrations. Reject non-finite model
  parameters and malformed callback velocities at their public boundaries.
- Require optional Rust engine 0.3.0 so accelerated installations receive the
  corresponding native solver corrections.
- Align the standalone distribution with the solver implementation qualified by
  `scpn-quantum-control` 1.3.0.

## [0.1.0] - 2026-07-17

### Added

- Initial extraction of the coupled-phase-oscillator (Kuramoto) toolkit from
  `scpn-quantum-control` into a standalone, NumPy+SciPy-floor distribution: the model
  family, integrators, exact mean-field reductions, stability/continuation analysis,
  reverse-mode and forward differentiation, and control primitives.
- Optional acceleration and interoperability extras: `[rust]`, `[julia]`, `[jax]`,
  `[torch]`, `[sklearn]`, `[viz]`.
- A documentation site (mkdocs-material): a generated handbook, a capability
  snapshot, tier and competitive benchmark pages, and an example gallery, with the
  renderers under `tools/` and a deterministic worked-workflow example under
  `examples/`. The `[docs]` extra and a `mkdocs build --strict` CI job build it.
