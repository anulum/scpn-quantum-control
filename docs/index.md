# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — scpn-quantum-control

[![CI](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://github.com/anulum/scpn-quantum-control/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Qiskit 2.2+](https://img.shields.io/badge/qiskit-2.2%2B-6929C4.svg)](https://qiskit.org)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12290/badge)](https://www.bestpractices.dev/projects/12290)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/anulum/scpn-quantum-control/badge)](https://securityscorecards.dev/viewer/?uri=github.com/anulum/scpn-quantum-control)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![PyPI](https://img.shields.io/pypi/v/scpn-quantum-control)](https://pypi.org/project/scpn-quantum-control/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scpn-quantum-control)](https://pypi.org/project/scpn-quantum-control/)
[![All-time Downloads](https://static.pepy.tech/badge/scpn-quantum-control)](https://pepy.tech/project/scpn-quantum-control)

Quantum simulation of coupled Kuramoto oscillator networks on IBM superconducting
hardware, with a hardware evidence ledger separating theory, simulator,
unmitigated hardware, mitigated hardware, and noise-limited claims.

## Positioning and purpose

`scpn-quantum-control` is an evidence-first platform for teams turning
coupled-oscillator models into reproducible computational evidence. It is aimed at
research groups and product teams that need a clearer path from model statement to
auditable result than ad-hoc notebooks can provide.

In practice, this means:

- explicit problem-to-experiment boundaries (`K_nm`, `omega`, solver surface);
- separation of simulator and hardware evidence classes;
- stable public interfaces for integration;
- explicit, non-silenced boundaries on unsupported or blocked modes.

## Start Here

If you are new to the repository, read these pages in order:

1. [Onboarding](onboarding.md) — what the software is, who it serves, what is
   mature, and what remains claim-bound.
2. [Quickstart](quickstart.md) — run a local no-credential Kuramoto-XY
   simulation.
3. [Tutorials](tutorials.md) — follow the learning path from first simulation
   to research workflow.
4. [Stable Facades API](stable_facades_api.md) — use supported public entry
   points before advanced internals.
5. [Hardware Status Ledger](hardware_status_ledger.md) — understand which
   hardware and scientific claims are promoted, bounded, or blocked.

The short version: provide a coupling matrix `K_nm` and natural frequencies
`omega`; the package validates the problem, compiles the XY Hamiltonian,
builds circuits, runs local or provider-backed execution when allowed, and
analyses synchronisation, entanglement, topology, and control observables under
explicit evidence boundaries.

## Product Map

| Lane | What users get now | Why it matters |
|---|---|---|
| Physics compiler | A reproducible `K_nm`/`omega` to XY-Hamiltonian path with simulator execution and classical references. | Researchers can compare oscillator-network hypotheses without rewriting the quantum stack. |
| Differentiable optimisation | Parameter-shift VQE, composed phase objectives, gradient evidence, and bounded compiler/program AD kernels. | Optimisation and ML users can see when gradients are exact, approximate, blocked, or still roadmap. |
| Hardware governance | Provider capability records, raw-count ledgers, result packs, and release gates. | Hardware claims stay auditable and safe to cite. |
| Integration surface | Stable facades, notebooks, tutorials, API maps, and Rust acceleration hooks. | External projects can adopt supported entry points without depending on internal layout. |

Use [Differentiable Programming](differentiable_programming.md) and
[Quantum Gradients](quantum_gradients.md) when the central question is whether a
workflow can be trained or optimised. Use [Hardware Status Ledger](hardware_status_ledger.md)
when the central question is whether a result can be promoted as hardware evidence.

## Status Snapshot — 2026-05-18

| Area | Public status |
|---|---|
| Package line | Version `0.9.11`, Python `>=3.10`, Qiskit `>=2.2,<3.0`. |
| Generic compiler surface | `scpn_quantum_control.kuramoto_core` validates arbitrary `K_nm`/`omega` inputs and compiles Hamiltonians, dense matrices, Trotter circuits, order-parameter measurements, and Kuramoto variant trajectories. |
| Hardware evidence | Promoted raw-count campaigns: `ibm_kingston` DLA parity Phase 1, selected Phase 2 A+G/B-C/popcount controls, and the SCPN/FIM negative/falsification result for the tested digital circuit family. Legacy `ibm_fez` rows require artefact-level citation. |
| Paper 0 source-validation register | Fully promoted through the source-accounting register from `P0R00001` through `P0R06211`; the planner reports `0` remaining work orders and `0` remaining records. This is source-bounded ingestion and fixture preservation, not external validation evidence. |
| Paper 0 downstream programme | [Paper 0 Experimental Pathway](paper0/paper0_experimental_pathway.md) defines the longer-term experimental agenda and methodology-paper route. Paper 27 is treated as a bounded implementation candidate, not the definitive programme source. |
| Claim source | [Hardware Status Ledger](hardware_status_ledger.md). |

## What this package does

The classical Kuramoto model for coupled oscillators maps directly to the quantum XY
spin Hamiltonian. Superconducting qubits are native simulators of this physics: each
qubit is an oscillator on the Bloch sphere, and the XX+YY coupling between qubits
reproduces the $\sin(\theta_j - \theta_i)$ interaction of the Kuramoto model.

This package provides three things:

1. **A compiler** that takes any coupling matrix $K_{nm}$ and natural frequencies
   $\omega_i$ and produces executable Qiskit circuits for IBM hardware.

2. **35 research modules** probing the synchronization phase
   transition — synchronization witnesses, topological diagnostics, chaos
   measures, computational complexity bounds, and open-system dynamics. ~4 are
   novel constructions; ~8 are first applications of existing tools to
   Kuramoto-XY; the rest are standard many-body diagnostics.

3. **The SCPN 16-layer network** as a built-in benchmark — the coupling matrix from
   Paper 27 of the Sentient-Consciousness Projection Network framework, where
   synchronization is the mechanism by which consciousness emerges across 16
   ontological layers.

4. **The Paper 0 source-validation register** as a source-accounting layer —
   generated validation modules, spec loaders, fixtures, and tests preserve
   ledger-bounded Paper 0 claims under an explicit non-hardware, non-external
   validation boundary.

5. **A differentiable computation lane** for supported scalar, vector, and
   matrix primitive kernels, including compiler-AD metadata and native Rust
   backend parity for selected primitives. Supported scalar program traces also
   expose native lowering reports before LLVM/JIT compilation, with strict
   no-tie native selection lowering for `np.where`, `maximum`, `minimum`, and
   `clip`, plus scalar 2x2/3x3/4x4/5x5 expression determinant,
   helper-backed 6x6 through 19x19 determinant, static square/rectangular trace,
   static diagonal gather/scatter, static dense inverse through 6x6, static
   vector and matrix-RHS solve through 6x6, and 2x2 product lowering.
   The native linalg support contract is introspectable before compilation, and
   wide determinant helpers are regression-tested on non-diagonal dense matrices.
   Unsupported compiler and program-AD paths fail closed rather than silently
   fabricating gradients.

6. **A documented quantum-gradient route** that starts from parameter-shift VQE
   and extends toward backend-aware gradient planning, gradient tapes,
   framework adapters, QNN/QGNN/QSNN workflows, analog oscillator mapping,
   open-system gradients, and benchmark evidence. Current capabilities and
   planned surfaces are separated in the support documentation.

Think of it as a quantum microscope for synchronization. Classical Kuramoto tells you
*when* oscillators lock in step. This package tells you *what the quantum state looks
like* at the transition, *how hard it is* to prepare, *what its topology reveals*, and
*where classical simulation fails*.

## Key results

| Result | Value |
|--------|-------|
| VQE ground-state row | **0.05%** (4-qubit, legacy ibm_fez artifact) |
| 16-layer UPDE snapshot | 46% error at depth 770 (NISQ-consistent) |
| Coherence wall | depth 250–400 (Heron r2) |
| DLA dimension formula | $2^{2N-1} - 2$ (exact, all $N$) |
| Research modules | See generated capability inventory for current package counts |
| IBM hardware evidence | Legacy ibm_fez artifact rows + 342-circuit ibm_kingston Phase 1 DLA-parity raw-count dataset |
| DLA parity asymmetry (hardware) | $+10.8\,\%$ mean for depths $\ge 4$, peak $+17.5\,\%$ at depth 6, reproduced from `data/phase1_dla_parity/` |
| Test suite | CI-gated suite, 91.5% aggregate coverage gate |
| Python modules | 766 Python source modules + 1 Rust crate (55 PyO3 bindings) + Julia tier (`accel/julia/*.jl`) |

## Package map

| Subpackage | Modules | Purpose |
|------------|:-------:|---------|
| `paper0` | 471 | Source-accounting validation modules and fixtures for processed Paper 0 records |
| `analysis` | 58 | Synchronisation probes: witnesses, witness discovery, QFI, PH, OTOC, Krylov, magic, BKT, DLA |
| `hardware` | 63 | IBM Quantum runner, plugin backends registry, AsyncHardwareRunner, trapped-ion backend, GPU offload, circuit cutting, fast sparse, qubit mapper (DynQ), provenance |
| `phase` | 29 | Time evolution: Trotter, VQE, ADAPT-VQE, VarQITE, AVQDS, QSVT, Floquet DTC, Lindblad, Kuramoto variants |
| `bridge` | 13 | $K_{nm}$ → Hamiltonian, cross-repo adapters (sc-neurocore, SSGF, orchestrator) |
| `applications` | 13 | FMO photosynthesis, power grid, Josephson array, EEG, ITER, quantum EVS, application benchmark plugins |
| `mitigation` | 12 | ZNE, PEC, dynamical decoupling, Z₂ parity, CPDR, symmetry verification, GUESS, compound |
| `qec` | 13 | Toric code, repetition code UPDE, surface code, biological surface code, DLA-protected memory/scar prototypes, error budget, multi-scale, syndrome flow |
| `control` | 11 | QAOA-MPC, residual-certified VQLS Grad-Shafranov, Petri nets, ITER disruption, topological optimiser |
| `identity` | 6 | VQE attractor, coherence budget, entanglement witness, fingerprint |
| `qsnn` | 7 | Quantum spiking neural networks (LIF, STDP, synapses, dynamic coupling, training) |
| `crypto` | 6 | BB84, Bell tests, topology-authenticated QKD, key hierarchy |
| `gauge` | 5 | U(1) Wilson loops, vortex detection, CFT, universality, confinement |
| `ssgf` | 4 | SSGF quantum integration |
| `benchmarks` | 7 | Classical vs quantum scaling, MPS baseline, GPU baseline, AppQSim |
| `psi_field` | 4 | U(1) compact lattice gauge: lattice, infoton, observables, SCPN mapping |
| `forecasting` | 1 | Held-out synchronisation forecasting over hardware traces and source-backed topology replays |
| `accel` | 3 | Multi-language dispatcher + Julia tier (Rust → Julia → Python fallback chain) |
| `fep` | 2 | Friston Free Energy Principle: variational free energy, predictive coding |
| `tcbo` | 1 | TCBO quantum observer |
| `pgbo` | 1 | PGBO quantum bridge |
| `l16` | 1 | Layer 16 quantum director |

## Quick example

Any coupling topology — bring your own $K$ and $\omega$:

```python
from scpn_quantum_control import QuantumKuramotoSolver, build_kuramoto_ring

K, omega = build_kuramoto_ring(6, coupling=0.5, rng_seed=42)
solver = QuantumKuramotoSolver(6, K, omega)
result = solver.run(t_max=1.0, dt=0.1, trotter_per_step=2)
print(f"R(t): {result['R']}")
```

Detect synchronization on hardware with witness operators:

```python
from scpn_quantum_control.analysis.sync_witness import evaluate_all_witnesses

# After running X-basis and Y-basis circuits on IBM hardware:
results = evaluate_all_witnesses(x_counts, y_counts, n_qubits=4)
for name, w in results.items():
    print(f"{name}: {'SYNCHRONIZED' if w.is_synchronized else 'incoherent'}")
```

## Limitations

- **NISQ benchmarking only.** Circuit depths >400 hit the coherence wall on Heron r2.
- **SCPN coupling matrix is from unpublished work.** The $K_{nm}$ parameterisation
  comes from Paper 27 (2025 working paper, no external citations). The Kuramoto→XY
  mapping is standard; the specific coupling structure is not independently validated.
- **No quantum advantage at this scale.** At $N=4$–16, classical exact diagonalisation
  is faster. Advantage requires $N \gg 20$ with error-corrected qubits.
- **IBM hardware claim hygiene.** The promoted raw-count dataset is Phase 1
  DLA parity on `ibm_kingston`; legacy `ibm_fez` rows must cite their
  committed artifact path. V2/frontier/queued-job outputs are unpromoted.

## Documentation

- [Onboarding](onboarding.md) — project purpose, user routes, application value, and claim boundaries
- [Installation](installation.md) — pip install + all optional extras
- [Quickstart](quickstart.md) — first experiment in 5 minutes
- [Kuramoto Core Facade](kuramoto_core_facade.md) — stable `K_nm`/`omega` compiler entry point
- [Stable Facades API](stable_facades_api.md) — mkdocstrings reference for first-path public facades
- [Physics-First Kuramoto-XY](physics_first_kuramoto_xy.md) — start from arbitrary oscillator networks before SCPN-specific layers
- [Differentiable Programming](differentiable_programming.md) — current AD capabilities, boundaries, and user routes
- [Quantum Gradients](quantum_gradients.md) — parameter-shift gradients, VQE convergence route, and verification evidence
- [Differentiable API](differentiable_api.md) — public `scpn_quantum_control.differentiable` namespace map
- [Differentiable Roadmap](differentiable_roadmap.md) — staged plan for framework adapters, advanced gradients, benchmarks, verification, and dashboards
- [API Overview](api.md) — stable facade route first, advanced module references second
- [Paper 0 Validation Register](paper0/paper0_validation_register.md) — completed Paper 0 source-accounting register and generated API contract
- [Paper 0 Processing Methodology](paper0/paper0_processing_methodology.md) — repeatable extraction, ledger, fixture, gate, and claim-boundary method for future Book II papers
- [Paper 0 Experimental Pathway](paper0/paper0_experimental_pathway.md) — downstream methodology-paper route and experimental programme derived from Paper 0 ingestion
- [Campaign Artefacts](campaigns/README.md) — dated preregistration, readiness, manifest, and result notes
- [Publication Operations](publication/README.md) — submission checklists, source packaging, and venue-readiness notes
- [Release Readiness Gate](release_readiness.md) — deterministic tag-readiness audit for version, coverage, behavioural quality, and claim-boundary artefacts
- [Research Gems](research_gems.md) — **33 analysis modules with theory and API**
- [Literature Catalogue](literature/README.md) — project-relevant literature surveys and citation-planning material
- [Equations](equations.md) — every equation in the codebase
- [Architecture](architecture.md) — dependency graph + 20 subpackages
- [Hardware Status Ledger](hardware_status_ledger.md) — claim classes and campaign evidence paths
- [Analysis API](analysis_api.md) — advanced reference for 46 analysis modules
- [Witness Discovery](witness_discovery.md) — Bayesian/bandit search over synchronisation witness candidates
- [Application Benchmark Plugins](application_benchmarks.md) — EEG, plasma, power-grid, and FEP datasets through the QPU artifact contract
- [Phase API](phase_api.md) — advanced reference for 29 evolution algorithms
- [Kuramoto Variants](kuramoto_variants.md) — higher-order, monitored, and PT-symmetric trajectory APIs
- [Classical Baselines](classical_baselines.md) — SciPy ODE, QuTiP Lindblad, and MPS TEBD provenance surfaces
- [Hardware Guide](hardware_guide.md) — IBM Quantum setup
- [Bridges](bridges_api.md) — cross-repo integrations
- [Tutorials](tutorials.md) — 4-level learning path, 14 tutorials
- [Notebooks](notebooks.md) — 98 tracked notebooks (47 core + 51 Colab)
- [Language Policy](language_policy.md) — Rust / Julia / Go / Mojo accel-chain rules
- [Pipeline Performance](pipeline_performance.md) — measured wall-times + multi-language benchmarks
- [Methods Benchmark Dashboard](methods_benchmark_dashboard.md) — one-command artefact regeneration and paper-supporting benchmark provenance
- [Issue Triage](triage.md) — label taxonomy, SLAs, routing
- [Falsification](falsification.md) — 8 named claims + falsifiers

---

**Contact:** [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[GitHub Discussions](https://github.com/anulum/scpn-quantum-control/discussions) |
[www.anulum.li](https://www.anulum.li)

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>

**Contact:** [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[GitHub Discussions](https://github.com/anulum/scpn-quantum-control/discussions) |
[www.anulum.li](https://www.anulum.li)

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
