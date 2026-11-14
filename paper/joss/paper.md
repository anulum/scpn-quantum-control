---
title: 'scpn-quantum-control: reproducible Kuramoto-XY quantum simulation and control workflows with Rust acceleration and auditable hardware evidence'
tags:
  - Python
  - Rust
  - quantum computing
  - Qiskit
  - Kuramoto model
  - XY Hamiltonian
  - synchronisation
  - NISQ
  - reproducibility
authors:
  - name: Miroslav Šotek
    orcid: 0009-0009-3560-0851
    affiliation: 1
affiliations:
  - name: ANULUM CH&LI, Marbach SG, Switzerland
    index: 1
date: 13 November 2026
bibliography: paper.bib
---

# Summary

Many natural and engineered systems behave like networks of coupled
oscillators: power grids, neurons, chemical reactions and arrays of
superconducting junctions. Such networks can fall into step, or synchronise,
and the Kuramoto model is the standard mathematical description of how that
happens [@Kuramoto1984; @Acebron2005]. The same coupling structure can be
written as an XY spin Hamiltonian, which a quantum processor can simulate
directly.

`scpn-quantum-control` is a Python package with Rust acceleration that takes
the two inputs of a Kuramoto network, a coupling matrix $K_{nm}$ and a vector
of natural frequencies $\omega$, and turns them into quantum Hamiltonians,
circuits, simulator runs and hardware experiments. It also records every result
in a form that another researcher can check: raw measurement counts, job
identifiers, circuit metadata and SHA-256 digests, together with an explicit
statement of what kind of claim each result supports. The package is released
under the AGPL-3.0 licence on PyPI (version 1.3.0) and archived on Zenodo
[@SotekSoftware2026].

# Statement of need

Quantum-simulation studies on noisy intermediate-scale quantum (NISQ) hardware
often leave a gap between the mathematical model and the evidence needed to
verify a hardware claim. To reproduce such a claim, a reader needs to know how
the coupling matrix was generated, how the Hamiltonian and circuits were built
and compiled, which classical baselines were used, which hardware jobs ran, and
how the raw counts were analysed. Small changes in ansatz topology,
transpilation or readout processing can dominate a reported effect.

`scpn-quantum-control` closes this gap for Kuramoto-XY dynamics. Its users are
quantum-software researchers, NISQ experimentalists and computational
physicists who need one reproducible path from an oscillator-network model to
circuits, simulators, hardware runs and published artefacts. The package does
not claim a quantum speed-up: at the sizes reachable today (up to 16 qubits)
classical solvers are faster and more accurate, and the documentation states
this first. Its value lies in characterising
hardware behaviour (synchronisation witnesses, symmetry-sector leakage,
entanglement checks) under honest classical baselines and complete data
provenance.

# State of the field

General-purpose frameworks cover the layers below this package. Qiskit
[@JavadiAbhari2024] provides circuits, operators, transpilation and execution
on IBM hardware; PennyLane [@Bergholm2018] provides differentiable hybrid
programs; QuTiP [@Lambert2026] provides open-system and master-equation
dynamics; Mitiq [@LaRose2022] provides error-mitigation techniques. Variational
methods such as VQE [@Peruzzo2014] with hardware-efficient ansatze
[@Kandala2017] are well established.

None of these tools joins the Kuramoto-to-XY mapping, coupling-topology-aware
ansatz construction, classical Kuramoto baselines, multi-provider hardware
execution and a claim-level evidence record in one package. Those pieces encode
one scientific model and one evidence policy, so they do not belong upstream in
a general framework. `scpn-quantum-control` therefore builds on the existing
ecosystem instead of replacing it: Qiskit remains the circuit and operator
layer, PennyLane, JAX, PyTorch and TensorFlow are optional gradient back ends,
and QuTiP serves as an open-system comparison. The classical oscillator toolkit
was extracted into the standalone `oscillatools` distribution, so that users
who do not need the quantum layer can use it on a NumPy and SciPy floor.

# Software design

Four design decisions shape the package.

**Domain layer over shared infrastructure.** Typed problem inputs (a
`KuramotoProblem` holding the symmetric coupling matrix and frequencies) are
compiled to Qiskit operators and circuits. The topology-informed ansatz places
entangling gates only where the coupling matrix is non-zero, so a sparse
network needs fewer entangling gates than an all-to-all ansatz.

**Python orchestration, Rust kernels.** The public interface is Python, because
that is where quantum researchers work. Hot paths where allocation or object
dispatch dominate, such as coupling-matrix construction, Hamiltonian assembly,
dynamical-Lie-algebra dimension and Monte Carlo XY sampling, are implemented in
Rust through PyO3 and shipped as the separate optional wheel
`scpn-quantum-engine`. Most accelerated routines fall back to pure Python when
the engine is absent, so the core package installs without a Rust toolchain,
and parity tests compare the two paths. The companion methods preprint describes the benchmark protocol and the
measured speed-ups [@SotekMethods2026].

**Claims as data.** A hardware status ledger separates theory, simulator,
unmitigated hardware, mitigated hardware and noise-limited results. A result is
promoted to a hardware claim only when its raw counts are committed. Hardware
result packs bind raw counts, reproduced summaries, job identifiers, SHA-256
digests, reproduction commands and an explicit claim boundary into one
manifest, and benchmark tables are regenerated from scripts into JSON and CSV
rather than edited by hand.

**Fail closed.** The differentiable-programming surface returns exact,
finite-shot or framework-comparison gradients only where they are supported;
unsupported paths raise an error instead of silently approximating. Provider
SDKs (IBM Quantum, IQM and others) are optional extras behind a common
hardware-abstraction layer, so the core package installs without them.

Quality is enforced by 18,657 test functions in 1,613 test files, a CI
coverage gate of 99.05%, mutation testing, static type checking, CodeQL
and SBOM workflows among 45 GitHub Actions workflows, and an OpenSSF Best
Practices passing badge.

# Research impact statement

The package has produced the circuits, simulations and hardware data of seven
preprints: synchronisation on a 156-qubit IBM superconducting processor
[@SotekSync2026]; parity-sector decoherence asymmetry [@SotekDLA2026], whose
raw counts are committed as hardware result packs, and reduced-Pauli
entanglement checks [@SotekPauli2026] on IBM Heron processors; monitored feedback
versus open-loop control on IBM dynamic circuits [@SotekFeedback2026]; a
hardware falsification of Fisher-information-inspired feedback
[@SotekFisher2026]; five preregistered campaigns on IQM Garnet
[@SotekIQM2026]; and the methods study of the Rust and VQE workflow
[@SotekMethods2026]. Two of these report negative or boundary results, which
the evidence ledger is designed to make publishable rather than hide.

The software has been developed in public since February 2026, with 22
releases on PyPI between 28 February and 29 October 2026 and Zenodo archives. Documented use is so far within the author's research programme; the
documentation, tutorials, notebooks and contribution guide are written for
outside users.

# AI usage disclosure

Generative AI was used substantively in all parts of this submission. Large
language model coding assistants generated and refactored source code, tests
and documentation from the first public release, and since July 2026 all code
in the repository has been written by AI agents. The assistants also drafted
and revised the text of this paper. From June 2026 the agents were coordinated
through SYNAPSE CHANNEL (versions 0.17.0 to 0.99.51) [@SotekSynapse2026], an
open-source agent-coordination system developed within the ANULUM Agentic
Coordination, Assurance and Continuity Systems programme.

The author framed the research problem, chose the architecture and the evidence
policy described above (the claim classes, the raw-count promotion rule, the
fail-closed gradient paths and the split between Python orchestration and Rust
kernels), specified the experiments and preregistrations, and decided which
changes were accepted. Each change was checked by an agent that did not write
it and is validated by the test suite and the continuous-integration gates; the
author verified the reported numbers against the committed raw data and
analysis artefacts. The author takes full responsibility for the software and
this paper.

# Acknowledgements

Access to IBM Quantum hardware was provided through the IBM Quantum Open Plan.
Access to IQM Garnet was provided by IQM through the IQM Resonance platform
under an extended-access research grant. The author received no other external
support for this work. The author is the founder of ANULUM, which offers
commercial licences for this software alongside the AGPL-3.0 licence.

# References
