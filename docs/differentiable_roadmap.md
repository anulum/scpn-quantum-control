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
| 3 | Gradient tape | Context manager and QNode-style tape records available for supported phase parameter-shift, seeded finite-shot replay, and provider-boundary evidence; expand to nesting semantics, arbitrary-QNode transforms, and programme-IR traces. |
| 4 | Backend gradient planner | Available for statevector, finite-shot simulator, fail-closed hardware routes, term-aware multi-frequency planning, registered Phase-QNode gate-aware evaluation planning, explicit opaque-callable 2N fallback records, no-submit hardware-gradient campaign specs for XY parameter-shift VQE and seeded SPSA validation, publication-package scaffolding for the planned hardware-gradient paper, and callback-based provider parameter-shift execution with shot/variance accounting; expand to provider-specific job submission policies. |
| 5 | Framework adapters | JAX host-callback, JAX gradient-agreement certificates, native JAX bounded phase-QNN value-and-gradient evidence, registered local Phase-QNode JAX statevector value-and-gradient plus flat, PyTree, and pmap/sharding native-transform evidence including PyTree Hessian symmetry checks, PyTorch tensor bridges, PyTorch bounded phase-QNN tensor-gradient evidence, bounded PyTorch module training-loop parity through `torch.func.grad` and `torch.compile`, registered local Phase-QNode PyTorch statevector plus `torch.func.grad`/`jacrev`/`vmap` and non-fullgraph `torch.compile` transform evidence, PyTorch module/transform/compiler/device maturity routing, PyTorch cloud validation batch planning for incompatible local accelerator routes, TensorFlow tensor bridges, TensorFlow bounded phase-QNN tensor-gradient evidence, bounded TensorFlow `GradientTape` evidence, bounded TensorFlow `tf.function` evidence, bounded TensorFlow XLA evidence, bounded TensorFlow Keras layer evidence, a bounded-QNN framework bridge matrix, PennyLane agreement checks, PennyLane caller-supplied QNode round-trip certificates, bounded PennyLane QNode conversion from registered local `PhaseQNodeCircuit` declarations with explicit device/shot/diff-method metadata, Qiskit shifted-circuit generation/local Statevector gradients including tied-parameter multi-frequency evaluation-count parity, a reproducible CPU framework overlay, real Phase-QNode external comparison rows for installed local frameworks, explicit Enzyme/compiler AD dependency-gap rows, and explicit Catalyst qjit/MLIR/QIR dependency-gap rows are available for supported calls. Arbitrary provider/native framework autodiff-through-simulator kernels, registered PyTorch fullgraph `torch.compile` lowering, incompatible CUDA/device execution without cloud artefacts, finite-shot native framework lowering, full provider job submission, unrestricted covariance-observable conversion, dynamic-circuit conversion, and unrestricted provider-backed gradient execution remain open. |
| 6 | QNN/QGNN/QSNN training | Bounded phase-QNN classifier, QNN-specific finite-difference gradient verifier, deterministic multi-seed convergence envelopes, bounded loss-landscape grids, seeded finite-shot gradient uncertainty and noisy-convergence evidence, named external-gradient agreement records with source-class/native-autodiff provenance, dedicated caller-supplied framework-gradient agreement checks, exact-answer bounded-QNN/Kuramoto-XY synthetic domain datasets, published public-domain Kuramoto artefact references, deterministic convergence-suite evidence, conformance-suite evidence with required-evidence unsuitable-scenario records, non-isolated optimizer-baseline comparisons across parameter-shift, finite-difference, SGD, Adam, L-BFGS-B, diagonal-Fisher natural-gradient, seeded SPSA, and derivative-free grid routes, QSNN parameter-shift evidence, and registered medium QNN/QGNN/QSNN/Kuramoto-XY evidence are available; broader seeded convergence notebooks and benchmarks remain the promotion gate for arbitrary architectures. |
| 7 | Compiler-backed AD | Executable MLIR/LLVM/JIT kernels beyond bounded scalar/vector/matrix paths, registered Phase-QNode MLIR-runtime value/gradient lowering adapters with dialect operation metadata, runtime shape/type verification, blocked interpreter-fallback success claims, bounded LLVM/Enzyme runner comparison rows, and bounded Catalyst qjit/MLIR/QIR runner comparison rows with strict JSON, timeout, toolchain, and correctness gates. Rust/PyO3/native-JIT gaps remain declared explicitly. |
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
