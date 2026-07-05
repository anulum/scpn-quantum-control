---
title: 'oscillatools: a differentiable, control-oriented toolkit for coupled-phase-oscillator (Kuramoto) dynamics'
tags:
  - Python
  - Rust
  - Kuramoto model
  - synchronisation
  - coupled oscillators
  - dynamical systems
  - differentiable simulation
  - nonlinear dynamics
authors:
  - name: Miroslav Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
affiliations:
  - name: ANULUM, Marbach SG, Switzerland
    index: 1
date: 5 July 2026
bibliography: paper.bib
---

<!--
DRAFT — not yet submitted to JOSS. `oscillatools` is the standalone,
numpy+scipy-floor distribution of the coupled-phase-oscillator toolkit, extracted
from the `scpn-quantum-control` repository (which carries its own software note,
submission 001). The accompanying Zenodo archive DOI is reserved and minted at
finalisation; the `doi:` front-matter field is intentionally omitted until then.
Numerical inventory figures below are the static file-system counts reported by
the repository capability manifest and are regenerated, not hand-entered.
-->

# Summary

The Kuramoto model of coupled phase oscillators is a standard paradigm for
studying synchronisation across physics, neuroscience, and engineering
[@Kuramoto1984; @Strogatz2000; @Acebron2005; @Rodrigues2016]. `oscillatools` is a
coupled-phase-oscillator toolkit built around three properties that are usually
assembled by hand for each study: end-to-end **differentiability** of the
simulated dynamics, first-class **control and inference** routines, and a uniform
**model and observable API** across a broad family of oscillator variants. It
installs on a numpy + scipy floor with optional extras for accelerated, JAX, and
plotting tiers, and exposes 392 public symbols across 85 single-responsibility
modules, with performance-sensitive kernels dispatched through a
Rust → Julia → Python fallback chain so that a pure Python floor is always
available and a faster measured backend is used when present.

The model family spans the classical mean-field and networked/graph couplings,
the Sakaguchi phase-frustrated variants, higher-order (triadic, simplicial, and
hyperedge) interactions, Daido mean-field modes, and the delayed, stochastic,
second-order inertial, and adaptive (Hebbian) extensions, alongside the
next-generation quadratic integrate-and-fire neural-mass reduction and related
Stuart–Landau, Winfree, swarmalator, and multiplex systems. Fixed- and
adaptive-step integrators (Euler, classical RK4, Dormand–Prince, a symplectic
scheme for the inertial model, a method-of-steps scheme for the delayed model,
and Euler–Maruyama for the stochastic model) share a common contract, and the
exact Ott–Antonsen [@OttAntonsen2008] and Watanabe–Strogatz [@WatanabeStrogatz1993]
low-dimensional reductions are provided for the cases where they apply.

# Statement of need

Kuramoto studies routinely require three capabilities that existing tooling
provides only partially, forcing researchers to write bespoke code that is hard
to reproduce and to verify:

1. **Gradients through the dynamics.** Control design, parameter estimation, and
   coupling inference all need derivatives of trajectory-level observables with
   respect to frequencies, coupling, and phase-lag. Hand-derived adjoints are
   error-prone and are rarely shared alongside the forward model.
2. **Control and inference as first-class operations**, not post-processing:
   optimal and pinning coupling design, coordinated-reset and feedback
   desynchronisation, and the inverse problem of recovering the coupling function
   or governing equations from observed phases.
3. **A consistent interface** across the many oscillator variants, so that a
   measurement, an integrator, or a sweep written for one model applies to the
   others without rewriting.

`oscillatools` addresses these needs directly. The core integrators ship with
matching reverse-mode adjoint and forward-sensitivity paths, so trajectory-level
objectives are differentiable without finite differences. The control and
inference layer includes optimal and pinning coupling design, coordinated-reset
and state-dependent Riccati feedback, differentiable model-predictive control,
learned and interval-bound-verified Lyapunov stability certificates,
coupling-function inference, sparse identification of the governing phase dynamics
in the SINDy sense [@Brunton2016], and dynamical Bayesian coupling inference. The
intended users are researchers in nonlinear dynamics, computational neuroscience,
and network control who need an auditable path from an oscillator model to a
gradient, a controller, or an inferred coupling.

# State of the field

General-purpose libraries cover parts of this space. `DynamicalSystems.jl`
provides a mature Julia interface for nonlinear dynamics and chaos, and its
`CoupledODEs` state/parameter/rule contract is the convention adopted here for
the unified system object [@Datseris2018]. `neurolib` offers a whole-brain
neural-mass simulation framework with a parameter-exploration workflow whose
`BoxSearch` / `ParameterSpace` convention motivates the sweep API in this package
[@Cakan2023]. Standard references establish the synchronisation theory the
package implements [@Pikovsky2001; @Acebron2005].

The contribution here is a Python package that specialises this general
machinery for coupled-phase-oscillator research and couples it to a
differentiable substrate. Rather than treating the simulator and the
control/inference tooling as separate concerns, `oscillatools` makes the forward
model, its gradient, and the control and inference operations share one model
representation and one observable interface. To keep the package usable inside
existing scientific pipelines, the system object integrates with
`scipy.integrate.solve_ivp` for adaptive and stiff solvers, exposes the
inference routines through a scikit-learn-style `fit`/`predict` estimator
interface, and provides a parameter-grid sweep over the shipped observables.
These interoperability layers are thin adapters over the core model and add no
hard dependency on the external libraries.

# Functionality

A minimal workflow constructs a system, sweeps a parameter, and measures an
observable across the grid:

```python
import numpy as np
from oscillatools import (
    KuramotoSystem,
    KuramotoParameterGrid,
    mean_order_parameter,
    sweep_parameter_grid,
)

rng = np.random.default_rng(0)
system = KuramotoSystem.mean_field(
    initial_phases=rng.uniform(-np.pi, np.pi, size=64),
    natural_frequencies=rng.normal(0.0, 0.5, size=64),
    coupling=0.0,
    dt=0.05,
)
grid = KuramotoParameterGrid({"coupling": np.linspace(0.0, 4.0, 21)})
result = sweep_parameter_grid(
    system, grid, [mean_order_parameter()], n_steps=2000, transient=1000
)
coherence = result.grid_values("mean_order_parameter")  # the synchronisation curve
```

The synchronisation curve rises from an incoherent floor to near-complete
coherence as the coupling crosses the critical value. The same model can be
integrated through `solve_ivp`, differentiated with the toolkit's adjoint
integrators, analysed for linear stability and Lyapunov exponents, continued along
a coupling branch to trace a hysteresis loop, or handed to the control and
inference routines. Numerical inventory figures quoted above are the static
file-system counts produced by the repository capability manifest and are
regenerated rather than transcribed; benchmark, coverage, and scientific-fidelity
claims are governed by their own dedicated evidence artefacts in the repository
and are not asserted here.

# AI usage disclosure

AI-assisted tools were used for drafting and editing support. Numerical claims,
citation accuracy, scientific framing, authorship, and final responsibility were
verified and retained by the author.

# References
