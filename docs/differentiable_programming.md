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
| Parameter-shift gradients | Available for callable scalar objectives, structured `PhaseVQE` gradients, local gradient-descent VQE examples, and metric-aware natural-gradient descent through `scpn_quantum_control.phase`. | [Quantum Gradients](quantum_gradients.md), [Variational Methods](variational.md) |
| Compiler/program AD | Available for supported scalar, vector, and matrix primitives with registry contracts, lowering reports, and native executable kernels on bounded paths. | [Differentiable API](differentiable_api.md), [Quickstart](quickstart.md) |
| Primitive registry | Available for derivative, batching, lowering, shape, dtype, and nondifferentiability contracts on supported primitive identities. | `scpn_quantum_control.differentiable` |
| Reverse replay and program traces | Available for supported captured operations; unsupported arbitrary Python remains fail-closed. | Support reports and module-specific tests |
| JAX, PyTorch, TensorFlow adapters | Optional parameter-shift value-and-gradient bridges plus a fail-closed ML parity audit are available for supported phase objectives; native framework autodiff through arbitrary simulators remains open. | [Differentiable Roadmap](differentiable_roadmap.md), [Quantum Gradients](quantum_gradients.md) |
| Gradient tape | MVP available for supported phase parameter-shift records; arbitrary Python and programme-IR tape semantics remain open. | [Quantum Gradients](quantum_gradients.md), [Differentiable Roadmap](differentiable_roadmap.md) |
| QNN/QGNN/QSNN training lane | Partly represented in existing QSNN and neural-state work; production convergence notebooks remain planned. | [Notebooks](notebooks.md) |

## User routes

| Goal | Recommended path |
|---|---|
| Train a small VQE objective | `phase.param_shift` -> [Quantum Gradients](quantum_gradients.md) -> [Variational Methods](variational.md) |
| Inspect compiler-backed AD | [Quickstart](quickstart.md) differentiable primitive path -> [Differentiable API](differentiable_api.md) |
| Build a custom primitive | `CustomDerivativeRule` -> `CustomDerivativeRegistry` -> primitive contract tests |
| Decide whether a backend can support gradients | Use current support docs; backend gradient planner remains roadmap work |
| Prepare ML-framework integration | Follow [Differentiable Roadmap](differentiable_roadmap.md) until adapter tests land |

## Design principles

- Fail closed when a derivative mode is unsupported.
- Separate exact, approximate, finite-shot, and roadmap gradient modes.
- Keep shape, dtype, backend, and primitive support inspectable.
- Compare gradients against finite differences, analytic references, and cross-framework references where practical.
- Document failed or unsuitable scenarios because they are research evidence.

## Immediate production targets

The next differentiable-programming implementation rounds should prioritise:

1. broader finite-difference verification for larger circuits;
2. multi-start convergence studies on known ground states, including natural-gradient and derivative-free baselines;
3. JAX agreement with manual parameter-shift;
4. PennyLane adapter round-trip tests;
5. QSNN training convergence tests;
6. public tutorials for Kuramoto-XY VQE gradients and coupling learning;
7. a support matrix for gates, observables, backends, transforms, and adapters.

## Unsupported boundaries

Unsupported does not mean ignored. Current public boundaries include:

- arbitrary Python/NumPy program AD beyond supported trace operations;
- full native compiler AD for every MLIR/LLVM/JIT path;
- complete gradient tape semantics beyond supported phase parameter-shift records;
- public JAX/PyTorch/TensorFlow adapters;
- hardware gradient jobs without explicit backend policy;
- gates without registered generator spectra;
- dynamic topology changes that invalidate parameter indexing;
- wide native quotient-linalg traces beyond the documented support profile.

See [Differentiable Roadmap](differentiable_roadmap.md) for the staged closure plan.
