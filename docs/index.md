# scpn-quantum-control

[![CI](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-quantum-control/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://github.com/anulum/scpn-quantum-control/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0%2B-6929C4.svg)](https://qiskit.org)

Quantum simulation of coupled Kuramoto oscillator networks on IBM superconducting
hardware, with 33 novel research modules probing the synchronization phase transition.

## What this package does

The classical Kuramoto model for coupled oscillators maps directly to the quantum XY
spin Hamiltonian. Superconducting qubits are native simulators of this physics: each
qubit is an oscillator on the Bloch sphere, and the XX+YY coupling between qubits
reproduces the $\sin(\theta_j - \theta_i)$ interaction of the Kuramoto model.

This package provides three things:

1. **A compiler** that takes any coupling matrix $K_{nm}$ and natural frequencies
   $\omega_i$ and produces executable Qiskit circuits for IBM hardware.

2. **33 research modules** (the "gems") implementing novel quantum probes of the
   synchronization phase transition — synchronization witnesses, topological
   diagnostics, chaos measures, computational complexity bounds, and open-system
   dynamics. 21 of these have no prior art in the literature.

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
| Novel research modules | 33 (21 with no prior art) |
| IBM hardware jobs | 9 submitted to ibm_fez (2 completed) |
| Test suite | **2,715 passing**, 13 skipped, 98% coverage |
| Python modules | 107 + 1 Rust crate |

## Package map

| Subpackage | Modules | Purpose |
|------------|:-------:|---------|
| `analysis` | 41 | Synchronization probes: witnesses, QFI, PH, OTOC, Krylov, magic, BKT, DLA |
| `phase` | 14 | Time evolution: Trotter, VQE, ADAPT-VQE, VarQITE, AVQDS, QSVT, Floquet DTC |
| `bridge` | 11 | $K_{nm}$ → Hamiltonian, cross-repo adapters (sc-neurocore, SSGF, orchestrator) |
| `control` | 5 | QAOA-MPC, VQLS Grad-Shafranov, Petri nets, ITER disruption classifier |
| `qsnn` | 5 | Quantum spiking neural networks (LIF, STDP, synapses, training) |
| `hardware` | 9 | IBM Quantum runner, trapped-ion backend, GPU offload, circuit cutting |
| `mitigation` | 4 | ZNE, PEC, dynamical decoupling, Z₂ parity post-selection |
| `gauge` | 5 | U(1) Wilson loops, vortex detection, CFT, universality, confinement |
| `identity` | 6 | VQE attractor, coherence budget, entanglement witness, fingerprint |
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
- **IBM hardware results incomplete.** 20 experiments implemented and tested
  on AerSimulator; none submitted to QPU yet (awaiting IBM Quantum budget).

## Documentation

- [Installation](installation.md) — pip install + dev setup
- [Quickstart](quickstart.md) — first experiment in 5 minutes
- [Research Gems](research_gems.md) — **33 novel modules with full theory and API**
- [Equations](equations.md) — every equation in the codebase
- [Architecture](architecture.md) — 107-module dependency graph
- [API Reference](api.md) — core module documentation
- [Analysis API](analysis_api.md) — 41 analysis modules
- [Phase API](phase_api.md) — 14 evolution algorithms
- [Hardware Guide](hardware_guide.md) — IBM Quantum setup
- [Bridges](bridges_api.md) — cross-repo integrations
- [Tutorials](tutorials.md) — 4-level learning path, 14 tutorials
- [Notebooks](notebooks.md) — 13 interactive notebooks

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
