# Preprint Outline

**Title:** Quantum simulation of coupled Kuramoto oscillators on IBM Heron r2: BKT transition, decoherence budget, and K_nm-informed ansatz advantage

**Target venue:** Physical Review Research (or Quantum Science and Technology)

**Authors:** Miroslav Šotek (ORCID 0009-0009-3560-0851)

---

## Abstract (150 words)

We present a quantum simulation framework for the Kuramoto model of coupled oscillators, mapped to the XY Hamiltonian on IBM Heron r2 superconducting hardware. Using a coupling-topology-informed variational ansatz, we demonstrate 6× faster convergence to the ground state compared to hardware-efficient alternatives. We characterize the decoherence budget at 16 qubits, showing that the XY Hamiltonian's spectral structure amplifies T2-limited fidelity loss at specific Trotter depths. We connect the synchronization transition to the Berezinskii-Kosterlitz-Thouless (BKT) universality class and show that the persistent homology threshold p_h1 = 0.72, previously empirical, is within 0.5% of the product of BKT universal constants A_HP × sqrt(2/π). We provide an open-source package (1300+ tests) with ADAPT-VQE, VarQITE, QSVT resource estimation, and benchmarks against five physical coupled-oscillator systems.

---

## I. Introduction

- Coupled oscillators: Kuramoto model, synchronization, applications (power grids, neural networks, photosynthesis, plasma)
- Quantum simulation on NISQ hardware: state of the art, IBM utility experiments (Nature 2023)
- The Kuramoto-XY mapping: dθ/dt = ω + KΣsin(Δθ) → H = -Σ K(XX+YY) - Σ ωZ
- Our contribution: physics-informed ansatz, BKT analysis, open-source framework

## II. Methods

### A. Hamiltonian construction
- K_nm coupling matrix (Paper 27 definition)
- XY Hamiltonian compilation to Qiskit SparsePauliOp
- Trotter decomposition with commutator error bounds (analytical: 4Σ|K_ij||ω_j - ω_i|)

### B. K_nm-informed ansatz
- Standard approach: EfficientSU2 (generic entanglement pattern)
- Our approach: CZ gates placed only where K_ij > threshold
- Circuit depth reduction: O(n²) → O(nnz(K))

### C. Hardware execution
- IBM Heron r2 (ibm_fez): 156 qubits, CZ error ~0.5%, T2 ~100μs
- PEA error mitigation (resilience_level=2)
- Fractional RZZ gates (native on Heron, 50-68% depth reduction)

### D. Analysis framework
- BKT observables: Fiedler eigenvalue, vortex density, Wilson loops
- Entanglement entropy scaling and CFT central charge
- Quantum Fisher Information for parameter estimation

## III. Results

### A. Ansatz advantage (HARDWARE DATA)
- Figure 1: VQE convergence — K_nm ansatz vs EfficientSU2 (6× speedup)
- 12 hardware data points on ibm_fez (February-March 2026)
- Energy relative error vs circuit depth

### B. Decoherence budget
- Figure 2: Fidelity vs Trotter depth at 4, 8, 16 qubits
- Coherence budget: max useful depth = T2 / (2 × t_gate × n_2q)
- Hardware data: noise baseline R = 0.784 (March) vs 0.805 (February)

### C. BKT analysis
- Figure 3: Phase diagram K_c vs T_eff
- Figure 4: Entanglement entropy S(n/2) vs coupling K
- Figure 5: Vortex density across synchronization transition
- Finding: p_h1 = A_HP × sqrt(2/π) = 0.717 (0.5% from 0.72)
  - Caveat: A_HP measured on square lattice, not K_nm graph
  - Significance: if confirmed, consciousness gate threshold = BKT universal

### D. Algorithm comparison
- Table 1: QSVT (O(αt)) vs Trotter-1 (O((αt)²/ε)) vs Trotter-2
- ADAPT-VQE: gradient-driven operator selection
- VarQITE: guaranteed convergence without optimizer

### E. Physical system benchmarks
- Table 2: Topology correlation ρ for FMO, IEEE 5-bus, JJA, EEG, ITER
- Finding: moderate correlations (0.2-0.4) — exponential decay is generic

## IV. Resource estimation

- Surface code: d=7 at p=0.3% → 1552 physical qubits for 16 oscillators
- Circuit cutting: 32 oscillators via 2×16 partitions on Heron
- GPU baseline: A100 beats QPU until N~33 (statevector) or N~25 (MPS at criticality)
- Honest quantum advantage boundary: N > 40 for generic dynamics

## V. Discussion

### What we claim:
- K_nm-informed ansatz provides measurable improvement (hardware-verified)
- BKT framework correctly describes the XY model synchronization transition
- p_h1 ≈ A_HP × sqrt(2/π) connects empirical threshold to BKT universals

### What we do NOT claim:
- Quantum advantage at 16 qubits (classical is faster)
- The K_nm values model any specific physical system
- "Consciousness" — we measure physical observables, not philosophical concepts
- p_h1 derivation is exact (square lattice A_HP, not graph A_HP)

### Reframe for reviewers:
- This is a NISQ benchmarking study, not a quantum advantage claim
- The K_nm ansatz is a general technique for structured Hamiltonians
- The BKT connection is standard XY physics, applied to a novel graph
- The consciousness interpretation is outside the scope of this paper

## VI. Conclusion

Open-source package: github.com/anulum/scpn-quantum-control (AGPL-3.0)
1300+ tests, Rust acceleration, 5 physical system benchmarks, 4 simulation algorithms.

## Data availability

Zenodo DOI: [to be created with v1.0.0 release]
IBM hardware data: 12 data points on ibm_fez (February-March 2026)

## References (~50 citations)

Key: Bastidas 2025, Babbush 2023, IBM Nature 2023, Kosterlitz-Thouless 1973,
Hasenbusch-Pinn 1997, Calabrese-Cardy 2004, Grimsley ADAPT-VQE 2019,
McArdle VarQITE 2019, Gilyén QSVT 2019, Huang shadows 2020
