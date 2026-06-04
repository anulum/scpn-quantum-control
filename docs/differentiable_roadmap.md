# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Roadmap

# Differentiable Roadmap

This roadmap defines the staged work needed to turn differentiable programming into a complete public product surface. It complements the internal TODO without exposing private execution notes.

## Critical path

| Stage | Deliverable | Promotion evidence |
|---|---|---|
| 1 | Parameter-shift core for Kuramoto-XY/VQE objectives | Analytic checks, finite-difference checks, convergence tests. |
| 2 | Public gradient API | `grad`, `value_and_grad`, support reports, typed errors. |
| 3 | Gradient tape | MVP context manager available for supported phase parameter-shift records; expand to nesting semantics and programme-IR traces. |
| 4 | Backend gradient planner | MVP available for statevector, finite-shot simulator, fail-closed hardware routes, and callback-based provider parameter-shift execution with shot/variance accounting; expand to provider-specific job submission policies. |
| 5 | Framework adapters | JAX host-callback, JAX gradient-agreement certificates, PyTorch tensor, TensorFlow tensor, PennyLane agreement checks, PennyLane caller-supplied QNode round-trip certificates, and Qiskit shifted-circuit generation/local Statevector gradients are available for supported phase parameter-shift calls. Native framework autodiff-through-simulator kernels, full provider job submission, and unrestricted provider-backed gradient execution remain open. |
| 6 | QNN/QGNN/QSNN training | Seeded convergence notebooks and benchmarks. |
| 7 | Compiler-backed AD | Executable MLIR/LLVM/JIT kernels beyond bounded scalar/vector/matrix paths. |
| 8 | Benchmark oracle | Quantum Sync Challenge fixtures, baselines, leaderboard-ready output. |
| 9 | Advanced control | Analog mapping, open-system gradients, MCWF, feedback control, dashboards. |

## Future-leading lanes

- Generalized quantum gradient calculus: multi-frequency parameter-shift, adjoint simulator gradients, stochastic finite-shot estimators, Hessians, quantum Fisher information, natural gradients, and Wirtinger semantics.
- Differentiable circuit transforms: gate decomposition, measurement grouping, symmetry projection, error mitigation, adaptive Trotter compensation, and transform provenance.
- Hardware-safe execution: dry-run cost estimates, shot allocation, batching, cache keys, timeouts, rate limits, and no-hardware-by-default policies.
- Data-driven coupling learning: EEG-like, power-grid-like, oscillator-array, and multimodal time-series adapters with privacy and claim-boundary rules.
- Analog oscillator mapping: neutral-atom, Rydberg, trapped-ion, photonic, and other continuous-time hardware mappings for Kuramoto/XY dynamics.
- Advanced witnesses: persistent homology, topological summaries, Krylov complexity, OTOC scaling, synchronization order parameters, tomography, and classical shadows.
- Fault-tolerant path: logical encodings, resource estimation, and error-corrected oscillator simulation assumptions.
- Visualization: phase portraits, order parameters, hardware bitstrings, shadow estimates, DLA/topological views, and feedback-control telemetry.

## Release rule

A differentiable feature becomes public-production only when the implementation, tests, docs, examples, support matrix, benchmark evidence, security checks, and failure-mode documentation agree. Until then, the docs must label it as experimental, planned, or unsupported.
