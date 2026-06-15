# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Roadmap

# Differentiable Roadmap

This roadmap defines the staged work needed to turn differentiable programming into a complete public product surface. It complements the private execution tracker without exposing private execution notes.

## Critical path

| Stage | Deliverable | Promotion evidence |
|---|---|---|
| 1 | Parameter-shift core for Kuramoto-XY/VQE objectives | Analytic checks, finite-difference checks, multi-frequency parameter-shift rules, second-order Hessian certificates, convergence tests. |
| 2 | Public gradient API | `grad`, `value_and_grad`, support reports, typed errors. |
| 3 | Gradient tape | MVP context manager and QNode-style tape records available for supported phase parameter-shift, seeded finite-shot replay, and provider-boundary evidence; expand to nesting semantics, arbitrary-QNode transforms, and programme-IR traces. |
| 4 | Backend gradient planner | MVP available for statevector, finite-shot simulator, fail-closed hardware routes, term-aware multi-frequency planning, and callback-based provider parameter-shift execution with shot/variance accounting; expand to provider-specific job submission policies. |
| 5 | Framework adapters | JAX host-callback, JAX gradient-agreement certificates, native JAX bounded phase-QNN value-and-gradient evidence, PyTorch tensor bridges, PyTorch bounded phase-QNN tensor-gradient evidence, TensorFlow tensor bridges, TensorFlow bounded phase-QNN tensor-gradient evidence, bounded TensorFlow `GradientTape` evidence, bounded TensorFlow `tf.function` evidence, bounded TensorFlow XLA evidence, bounded TensorFlow Keras layer evidence, a bounded-QNN framework bridge matrix, PennyLane agreement checks, PennyLane caller-supplied QNode round-trip certificates, bounded PennyLane QNode conversion from registered local `PhaseQNodeCircuit` declarations with explicit device/shot/diff-method metadata, Qiskit shifted-circuit generation/local Statevector gradients, a reproducible CPU framework overlay, real Phase-QNode external comparison rows for installed local frameworks, and explicit Enzyme/compiler AD dependency-gap rows are available for supported calls. Arbitrary native framework autodiff-through-simulator kernels, full provider job submission, unrestricted covariance-observable conversion, dynamic-circuit conversion, and unrestricted provider-backed gradient execution remain open. |
| 6 | QNN/QGNN/QSNN training | Bounded phase-QNN classifier, QNN-specific finite-difference gradient verifier, deterministic multi-seed convergence envelopes, bounded loss-landscape grids, seeded finite-shot gradient uncertainty and noisy-convergence evidence, named external-gradient agreement records with source-class/native-autodiff provenance, dedicated caller-supplied framework-gradient agreement checks, deterministic convergence-suite evidence, conformance-suite evidence with required-evidence unsuitable-scenario records, non-isolated optimizer-baseline comparisons, QSNN parameter-shift evidence, and registered medium QNN/QGNN/QSNN/Kuramoto-XY evidence are available; broader seeded convergence notebooks and benchmarks remain the promotion gate for arbitrary architectures. |
| 7 | Compiler-backed AD | Executable MLIR/LLVM/JIT kernels beyond bounded scalar/vector/matrix paths, registered Phase-QNode MLIR-runtime value/gradient lowering adapters with dialect operation metadata, runtime shape/type verification, blocked interpreter-fallback success claims, and bounded LLVM/Enzyme runner comparison rows with strict JSON, timeout, toolchain, and correctness gates. Rust/PyO3/native-JIT gaps remain declared explicitly. |
| 8 | Benchmark oracle | Quantum Sync Challenge fixtures, baselines, leaderboard-ready output, CI-only benchmark evidence bundles, self-hosted `isolated-benchmark` runner setup, isolated-affinity benchmark metadata gates, and fail-closed CUDA/ROCm accelerator metadata guards. |
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

A differentiable feature becomes public-production only when the implementation, tests, docs, examples, support matrix, benchmark evidence, security checks, and failure-mode documentation agree. The formal claim ledger must name artefact IDs and benchmark IDs for promoted rows. Until then, the docs must label it as experimental, planned, unsupported, or SOTA-candidate.
