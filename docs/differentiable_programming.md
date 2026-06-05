# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Differentiable Programming

# Differentiable Programming

`scpn-quantum-control` treats differentiability as a product surface, not a hidden implementation detail. The goal is to make coupled-oscillator quantum control trainable, testable, and explainable across quantum gradients, classical program AD, compiler-backed kernels, and future ML-framework adapters.

## Why this matters

A useful quantum-control framework must answer four questions quickly:

1. Can the objective produce a gradient?
2. Is that gradient analytic, approximate, stochastic, or unsupported?
3. Does the gradient help an optimiser converge on a known problem?
4. Can another tool or framework reproduce the same derivative?

This repository now documents those questions directly. Current support is deliberately bounded so users can trust the surfaces that are advertised.

## Current capability map

| Surface | Status | Evidence route |
|---|---|---|
| Parameter-shift gradients | Available for callable scalar objectives, structured `PhaseVQE` gradients, local gradient-descent VQE examples, metric-aware natural-gradient descent, multi-start optimizer comparison evidence, and compatible composed phase-control objectives through `scpn_quantum_control.phase`. | [Quantum Gradients](quantum_gradients.md), [Variational Methods](variational.md) |
| Objective evidence | Available for composed phase-control objectives through finite-difference agreement, compatibility-gate checks, and local training certificates. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Objective execution planning | Available for composed objectives through fail-closed routing between pure parameter-shift, hybrid term-gradient, hardware-blocked, and unsupported backend routes. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Gradient support matrix | Available for executable planning across registered gates, observables, backends, transforms, and adapters with explicit alternatives for blocked combinations. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Transform nesting governance | Available for bounded first-order, value-and-gradient, deterministic local Hessian, nested-grad Hessian, tape, scalar local JVP/VJP, scalar local jacfwd/jacrev, deterministic native vector-output Jacobian execution, native manual `vmap(grad)`, and provider-callback QNode transforms with finite-shot uncertainty propagation, with fail-closed framework vectorization, adapter-nesting, malformed-provider, finite-shot curvature, and hardware boundaries. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Provider-gradient readiness | Available as an executable support matrix for deterministic callbacks, finite-shot callbacks, multi-frequency rules, hardware-blocked routes, unknown backends, malformed samples, and policy-bound hardware-preparation evidence. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Hardware-gradient policy readiness | Available as a fail-closed dry-run policy layer for provider/backend allowlists, shot and evaluation budgets, evidence IDs, and live-ticket gating before hardware-gradient job preparation. Provider preparation records can be generated without executing QPU jobs. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |
| Compiler/program AD | Available for supported scalar, vector, and matrix primitives with registry contracts, lowering reports, and native executable kernels on bounded paths. | [Differentiable API](differentiable_api.md), [Quickstart](quickstart.md) |
| Primitive registry | Available for derivative, batching, lowering, shape, dtype, and nondifferentiability contracts on supported primitive identities. | `scpn_quantum_control.differentiable` |
| Reverse replay and program traces | Available for supported captured operations; unsupported arbitrary Python remains fail-closed. | Support reports and module-specific tests |
| JAX, PyTorch, TensorFlow adapters | Optional parameter-shift value-and-gradient bridges plus a fail-closed ML parity audit are available for supported phase objectives; native framework autodiff through arbitrary simulators remains open. | [Differentiable Roadmap](differentiable_roadmap.md), [Quantum Gradients](quantum_gradients.md) |
| Gradient tape | MVP available for supported phase parameter-shift records, plus QNode-style tape records for deterministic, seeded finite-shot, and provider-boundary evidence; arbitrary Python and programme-IR tape semantics remain open. | [Quantum Gradients](quantum_gradients.md), [Differentiable Roadmap](differentiable_roadmap.md) |
| QNN/QGNN/QSNN training lane | A bounded phase-QNN binary classifier, QNN-specific finite-difference gradient verification, seeded finite-shot gradient uncertainty and noisy-convergence evidence, named external-gradient agreement records, dedicated caller-supplied framework-gradient agreement checks, deterministic convergence-suite evidence, conformance-suite evidence with unsuitable-scenario records, non-isolated optimizer-baseline comparisons, and QSNN parameter-shift training evidence are available locally; arbitrary QNN/QGNN/QSNN stacks and production convergence notebooks remain planned. | [Differentiable API](differentiable_api.md), [Quantum Gradients](quantum_gradients.md) |

## User routes

| Goal | Recommended path |
|---|---|
| Train a small VQE objective | `phase.param_shift` -> [Quantum Gradients](quantum_gradients.md) -> [Variational Methods](variational.md) |
| Train and verify a bounded QNN classifier | `phase.qnn_training` -> `train_parameter_shift_qnn_classifier(...)` -> `verify_parameter_shift_qnn_classifier_gradient(...)` -> `estimate_parameter_shift_qnn_finite_shot_gradient(...)` -> `run_parameter_shift_qnn_conformance_suite(...)` -> `run_parameter_shift_qnn_convergence_suite(...)` -> `run_parameter_shift_qnn_finite_shot_convergence_suite(...)` -> `run_parameter_shift_qnn_framework_agreement_suite(...)` -> `run_parameter_shift_qnn_optimizer_benchmark_suite(...)` -> [Quantum Gradients](quantum_gradients.md) |
| Inspect compiler-backed AD | [Quickstart](quickstart.md) differentiable primitive path -> [Differentiable API](differentiable_api.md) |
| Build a custom primitive | `CustomDerivativeRule` -> `CustomDerivativeRegistry` -> primitive contract tests |
| Decide whether a gradient stack can run | `plan_gradient_support(...)`, `plan_gradient_transform_nesting(...)`, `plan_quantum_gradient_backend(...)`, `run_phase_qnode_tape_readiness_suite()`, `run_provider_gradient_readiness_audit(...)`, and `run_hardware_gradient_policy_readiness_suite()` |
| Prepare ML-framework integration | Follow [Differentiable Roadmap](differentiable_roadmap.md) until adapter tests land |

## Production-Readiness Rubric

A differentiable workflow should be treated as production-ready only when all
of these checks are true:

| Check | Required evidence |
|---|---|
| Mathematical derivative contract | Parameter-shift, analytic derivative, adjoint replay, or compiler-AD rule is named and registered. |
| Shape and dtype contract | Primitive registry or support matrix admits the target tensor rank, dtype, backend, and static arguments. |
| Verification | Finite-difference, analytic, or independent framework agreement is recorded for a small representative case. |
| Optimiser behaviour | Descent or convergence diagnostics exist for the target objective class. |
| Backend policy | Simulator, finite-shot simulator, hardware, or adapter route declares shot, variance, budget, evidence-ID, ticket, and blocked-state policy. |
| Documentation | Unsupported or unsuitable scenarios are listed with alternatives and no silent fallback. |

This is the current standard for claiming enterprise-grade differentiable
behaviour in this repository. Anything below that bar is documented as staged,
experimental, or unsupported.

## Design principles

- Fail closed when a derivative mode is unsupported.
- Separate exact, approximate, finite-shot, and roadmap gradient modes.
- Do not silently treat analytic classical penalties as parameter-shift quantum terms.
- Keep shape, dtype, backend, and primitive support inspectable.
- Compare gradients against finite differences, analytic references, and cross-framework references where practical.
- Document failed or unsuitable scenarios because they are research evidence.

## Immediate production targets

The next differentiable-programming implementation rounds should prioritise:

1. broader finite-difference verification for larger circuits;
2. multi-start convergence studies on known ground states and VQE systems, extending the current phase-optimizer comparison audit with derivative-free baselines;
3. native JAX agreement beyond QNN caller-supplied framework-gradient records and the existing host-boundary bridge;
4. broader PennyLane adapter round-trip tests beyond caller-supplied framework-gradient agreement checks;
5. broader QNN/QGNN/QSNN convergence notebooks beyond the bounded local phase-QNN conformance, deterministic convergence, seeded finite-shot, and optimizer-baseline suites plus QSNN tests;
6. public tutorials for Kuramoto-XY VQE gradients and coupling learning;
7. executable implementations for still-blocked framework-native nested routes where the physics contract is clear; native vector-output Jacobian, provider-callback QNode transforms, and manual `vmap(grad)` now have bounded local evidence.

## Unsupported boundaries

Unsupported does not mean ignored. Current public boundaries include:

- arbitrary Python/NumPy program AD beyond supported trace operations;
- full native compiler AD for every MLIR/LLVM/JIT path;
- complete gradient tape semantics beyond supported phase parameter-shift and QNode-style phase records;
- public JAX/PyTorch/TensorFlow adapters;
- hardware gradient jobs without hardware-gradient policy approval, required evidence IDs, and live-execution ticketing where applicable;
- provider callbacks that omit finite-shot variance or shifted-sample provenance;
- unsupported gate, observable, transform, adapter, or backend combinations returned by `plan_gradient_support(...)`;
- unsupported transform nesting returned by `plan_gradient_transform_nesting(...)`;
- arbitrary quantum neural architectures beyond the bounded local phase-QNN classifier, its QNN-specific gradient verifier, finite-shot simulator evidence, conformance, convergence, framework-agreement, and optimizer-baseline suites, and declared QSNN training routes;
- gates without registered generator spectra;
- dynamic topology changes that invalidate parameter indexing;
- wide native quotient-linalg traces beyond the documented support profile.

See [Differentiable Roadmap](differentiable_roadmap.md) for the staged closure plan.
