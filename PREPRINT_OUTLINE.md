# Preprint Outline

**Title:** Quantum simulation of coupled-oscillator synchronisation on a 156-qubit superconducting processor

**Target venue:** Physical Review Research (or Quantum Science and Technology)

**Authors:** Miroslav Šotek (ORCID 0009-0009-3560-0851)

---

## Abstract (150 words)

We present a quantum simulation framework for the Kuramoto model of coupled oscillators, mapped to the XY Hamiltonian on IBM Heron r2 superconducting hardware. Using a coupling-topology-informed variational ansatz, we demonstrate 6× faster convergence to the ground state compared to hardware-efficient alternatives. We characterise the decoherence budget at 16 qubits, showing that the XY Hamiltonian's spectral structure amplifies T2-limited fidelity loss at specific Trotter depths. We connect the synchronisation transition to the Berezinskii-Kosterlitz-Thouless (BKT) universality class and report a negative result for the persistent homology threshold p_h1 = 0.72: the square-lattice expression A_HP × sqrt(2/π) is numerically close, but the K_nm graph Monte Carlo gives p_h1 ≈ 0.97, so 0.72 remains an open empirical/theoretical parameter. We provide an open-source package with ADAPT-VQE, VarQITE, QSVT resource estimation, and benchmarks against five physical coupled-oscillator systems.

---

## I. Introduction

- Coupled oscillators: Kuramoto model, synchronisation, applications (power grids, neural networks, photosynthesis, plasma)
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
- Figure 5: Vortex density across synchronisation transition
- Negative result: p_h1 = A_HP(square lattice) × sqrt(2/π) = 0.717 is a numerical coincidence, not a K_nm derivation
  - K_nm graph Monte Carlo gives A_HP ≈ 1.214 and p_h1 ≈ 0.97
  - Significance: p_h1 = 0.72 must remain an explicit open parameter until another derivation or measurement closes it

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
- Exact Hilbert-space simulation crossover: n≈11.6 from committed classical baselines plus completed ibm_fez scaling jobs
- Honest broad quantum advantage boundary remains open; Rust Kuramoto ODE baselines remain faster through n≤16

## V. Discussion

### What we claim:
- K_nm-informed ansatz provides measurable improvement (hardware-verified)
- BKT framework correctly describes the XY model synchronisation transition
- p_h1 = 0.72 remains open; the square-lattice BKT coincidence is falsified on the K_nm graph
- Exact state-vector simulation hits a measured resource boundary near n≈11.6

### What we do NOT claim:
- Quantum advantage at 16 qubits (classical is faster)
- Broad quantum advantage from the exact-simulation crossover figure
- The K_nm values model any specific physical system
- "Consciousness" — we measure physical observables, not philosophical concepts
- p_h1 = 0.72 is derived from first principles

### Reframe for reviewers:
- This is a NISQ benchmarking study, not a quantum advantage claim
- The K_nm ansatz is a general technique for structured Hamiltonians
- The BKT connection is standard XY physics, applied to a novel graph
- The consciousness interpretation is outside the scope of this paper

## VI. Conclusion

Open-source package: github.com/anulum/scpn-quantum-control (AGPL-3.0)
Rust acceleration, 5 physical system benchmarks, 4 simulation algorithms, and CI-gated tests.

## Data availability

Zenodo DOI: [to be created with v1.0.0 release]
IBM hardware data: 12 data points on ibm_fez (February-March 2026)

## References (~50 citations)

Key: Bastidas 2025, Babbush 2023, IBM Nature 2023, Kosterlitz-Thouless 1973,
Hasenbusch-Pinn 1997, Calabrese-Cardy 2004, Grimsley ADAPT-VQE 2019,
McArdle VarQITE 2019, Gilyén QSVT 2019, Huang shadows 2020
