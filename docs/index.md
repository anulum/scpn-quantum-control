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
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0%2B-6929C4.svg)](https://qiskit.org)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12290/badge)](https://www.bestpractices.dev/projects/12290)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/anulum/scpn-quantum-control/badge)](https://securityscorecards.dev/viewer/?uri=github.com/anulum/scpn-quantum-control)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)

Quantum simulation of coupled Kuramoto oscillator networks on IBM superconducting
hardware, with 33 research modules probing the synchronization phase transition.

## What this package does

The classical Kuramoto model for coupled oscillators maps directly to the quantum XY
spin Hamiltonian. Superconducting qubits are native simulators of this physics: each
qubit is an oscillator on the Bloch sphere, and the XX+YY coupling between qubits
reproduces the $\sin(\theta_j - \theta_i)$ interaction of the Kuramoto model.

This package provides three things:

1. **A compiler** that takes any coupling matrix $K_{nm}$ and natural frequencies
   $\omega_i$ and produces executable Qiskit circuits for IBM hardware.

2. **33 research modules** (the "gems") probing the synchronization phase
   transition — synchronization witnesses, topological diagnostics, chaos
   measures, computational complexity bounds, and open-system dynamics. ~4 are
   novel constructions; ~8 are first applications of existing tools to
   Kuramoto-XY; the rest are standard many-body diagnostics.

3. **The SCPN 16-layer network** as a built-in benchmark — the coupling matrix from
   Paper 27 of the Sentient-Consciousness Projection Network framework, where
   synchronization is the mechanism by which consciousness emerges across 16
   ontological layers.

Think of it as a quantum microscope for synchronization. Classical Kuramoto tells you
*when* oscillators lock in step. This package tells you *what the quantum state looks
like* at the transition, *how hard it is* to prepare, *what its topology reveals*, and
*where classical simulation fails*.

## Key results

| Result | Value |
|--------|-------|
| VQE ground-state error | **0.05%** (4-qubit, ibm_fez) |
| 16-layer UPDE snapshot | 46% error at depth 770 (NISQ-consistent) |
| Coherence wall | depth 250–400 (Heron r2) |
| DLA dimension formula | $2^{2N-1} - 2$ (exact, all $N$) |
| Research modules | 35 (≈ 5 novel, ≈ 10 first-application) |
| IBM hardware jobs | 33 on ibm_fez (Feb 2026) + 348 on ibm_kingston (Apr 2026, Phase 1 DLA-parity campaign) |
| DLA parity asymmetry (hardware) | $+10.8\,\%$ mean for depths $\ge 4$, peak $+17.5\,\%$ at depth 6 (Welch combined $p \ll 10^{-16}$) |
| Test suite | **4,828 passing**, 97%+ coverage |
| Python modules | 201 + 1 Rust crate (36 functions) |

## Package map

| Subpackage | Modules | Purpose |
|------------|:-------:|---------|
| `analysis` | 44 | Synchronization probes: witnesses, QFI, PH, OTOC, Krylov, magic, BKT, DLA |
| `phase` | 26 | Time evolution: Trotter, VQE, ADAPT-VQE, VarQITE, AVQDS, QSVT, Floquet DTC, Lindblad |
| `hardware` | 17 | IBM Quantum runner, trapped-ion backend, GPU offload, circuit cutting, fast sparse |
| `bridge` | 12 | $K_{nm}$ → Hamiltonian, cross-repo adapters (sc-neurocore, SSGF, orchestrator) |
| `applications` | 11 | FMO photosynthesis, power grid, Josephson array, EEG, ITER, quantum EVS |
| `control` | 7 | QAOA-MPC, VQLS Grad-Shafranov, Petri nets, ITER disruption, topological optimizer |
| `mitigation` | 7 | ZNE, PEC, dynamical decoupling, Z₂ parity, CPDR, symmetry verification |
| `identity` | 6 | VQE attractor, coherence budget, entanglement witness, fingerprint |
| `qsnn` | 6 | Quantum spiking neural networks (LIF, STDP, synapses, dynamic coupling, training) |
| `crypto` | 6 | BB84, Bell tests, topology-authenticated QKD, key hierarchy |
| `gauge` | 5 | U(1) Wilson loops, vortex detection, CFT, universality, confinement |
| `qec` | 5 | Toric code, repetition code UPDE, surface code, biological surface code, error budget |
| `ssgf` | 4 | SSGF quantum integration |
| `benchmarks` | 4 | Classical vs quantum scaling, MPS baseline, GPU baseline, AppQSim |
| `tcbo` | 1 | TCBO quantum observer |
| `pgbo` | 1 | PGBO quantum bridge |
| `l16` | 1 | Layer 16 quantum director |
| `qec` | 4 | Toric code, repetition code UPDE, surface code estimation, error budget |
| `applications` | 10 | FMO photosynthesis, power grid, Josephson array, EEG, ITER, quantum EVS |
| `crypto` | 4 | BB84, Bell tests, topology-authenticated QKD |

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
- **IBM hardware campaign complete.** 33 jobs on ibm_fez (Heron r2), 176K+ shots.
  CHSH S=2.165, QBER 5.5%, 16q UPDE, dual protection confirmed.

## Documentation

- [Installation](installation.md) — pip install + dev setup
- [Quickstart](quickstart.md) — first experiment in 5 minutes
- [Research Gems](research_gems.md) — **33 analysis modules with theory and API**
- [Equations](equations.md) — every equation in the codebase
- [Architecture](architecture.md) — 107-module dependency graph
- [API Reference](api.md) — core module documentation
- [Analysis API](analysis_api.md) — 41 analysis modules
- [Phase API](phase_api.md) — 14 evolution algorithms
- [Hardware Guide](hardware_guide.md) — IBM Quantum setup
- [Bridges](bridges_api.md) — cross-repo integrations
- [Tutorials](tutorials.md) — 4-level learning path, 14 tutorials
- [Notebooks](notebooks.md) — 47 notebooks (13 core + 34 FIM investigation)

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
