# Preprint Methods Draft

## II. Methods

### A. Hamiltonian Construction

The Kuramoto model of N coupled oscillators with phases θ_i and natural
frequencies ω_i evolves under:

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)

Near synchronisation (small Δθ), sin(Δθ) ≈ Δθ, yielding the XY model
Hamiltonian on the coupling graph:

    H_XY = -Σ_{i<j} K_ij (X_i X_j + Y_i Y_j) - Σ_i (ω_i/2) Z_i     (1)

where X_i, Y_i, Z_i are Pauli operators on qubit i. The coupling matrix
K_ij is symmetric with exponential distance decay K_ij = K_base × exp(-α|i-j|),
with calibration anchors from [Paper 27 ref].

The Hamiltonian is compiled to a SparsePauliOp with 2 × n(n-1)/2 + n terms
(XX and YY pairs plus Z fields). The 1-norm α = Σ|c_k| determines block-encoding
normalisation for optimal algorithms.

### B. K_nm-Informed Variational Ansatz

Standard hardware-efficient ansätze (e.g., EfficientSU2) place entangling
gates between all adjacent qubits regardless of the Hamiltonian structure.
Our K_nm-informed ansatz places CZ gates only between qubits (i,j) where
K_ij exceeds a threshold K_th:

    U(θ) = Π_l [Π_{(i,j): K_ij > K_th} CZ_{ij} × Π_i Ry(θ_il)]     (2)

This reduces the CZ count from O(n) per layer (linear connectivity) to
O(nnz(K > K_th)) and aligns the entanglement topology with the coupling
structure.

**Hardware result:** On IBM Heron r2 (ibm_fez), the K_nm ansatz converges
to the ground state energy in 6× fewer VQE iterations than EfficientSU2
at matched circuit depth (12 data points, February-March 2026).

### C. Trotter Decomposition with Analytical Error Bounds

Time evolution U(t) = exp(-iHt) is approximated by first-order
Suzuki-Trotter decomposition:

    U_1(t/r)^r = [Π_k exp(-i H_k t/r)]^r + O(t²/r × ||[H_i, H_j]||)  (3)

We derive the commutator bound analytically:

    ||[H_XY, H_Z]|| ≤ 4 Σ_{i<j} |K_ij| |ω_j - ω_i|                  (4)

This bound vanishes when all frequencies are equal (pure XY, no Z field),
providing a concrete Trotter error guarantee. For SCPN default parameters
(K_base=0.45, ω range 0.48-3.78), the bound gives ε_T < 0.003 at r=10
Trotter steps for t=1.

### D. Hardware Execution

Experiments were performed on IBM Heron r2 (ibm_fez backend):
- 156 superconducting qubits (transmon), heavy-hex topology
- CZ gate error: ~0.5%, T1 ≈ 200 μs, T2 ≈ 100 μs
- Native gate set: {Rz, SX, CZ, RZZ(θ)} (fractional gates enabled)
- Error mitigation: Probabilistic Error Amplification (PEA, resilience_level=2)
- Transpilation: Qiskit 1.4.5, optimisation_level=3

Fractional RZZ gates reduce circuit depth by 50-68% for the XY coupling
terms compared to CZ decomposition.

### E. Classical Simulation Baselines

For honest comparison, we provide three classical baselines:

1. **Exact diagonalisation:** O(2^n) memory, O(2^{2n}) time. Practical for n ≤ 16.
2. **MPS tensor network:** bond dimension χ determines accuracy. At BKT criticality,
   S(n/2) ~ (c/3) log(n) with c=1, requiring χ ~ n^{1/3}.
3. **GPU statevector:** NVIDIA A100 (312 TFLOPS FP64, 80 GB HBM). Faster than QPU
   until n ≈ 33 (memory-limited) or n ≈ 25 (MPS at criticality).

### F. BKT Analysis

The XY model on a graph exhibits a Berezinskii-Kosterlitz-Thouless (BKT)
synchronisation transition characterised by:

- Fiedler eigenvalue λ_2 of the coupling-weighted Laplacian → T_BKT = (π/2) × λ_2/(2n)
- Nelson-Kosterlitz stiffness jump: ρ_s(T_BKT⁻) = (2/π) × T_BKT
- Vortex density (H1 persistence proxy) as BKT order parameter
- Wilson loop decay: area law (disordered) vs perimeter law (ordered)

### G. Dynamical Lie Algebra

The DLA of the Hamiltonian generators determines classical simulability
[Goh et al., 2025]. We compute the DLA dimension via Rust-accelerated
commutator closure (975× faster than Python):

| N | DLA dim | max su(2^N) | Fraction |
|---|---------|-------------|----------|
| 4 | 126     | 255         | 49.4%    |
| 5 | 510     | 1023        | 49.9%    |

The ~50% fraction is due to heterogeneous SCPN frequencies breaking the
SU(2) symmetry of the pure XY model. This places the system outside the
trivially classically simulable regime, though it does not by itself
establish quantum advantage.

### H. Monte Carlo Verification

To verify whether the persistent homology threshold p_h1 = 0.72 can be
derived from BKT universals, we perform Metropolis-Hastings Monte Carlo
on the classical XY model defined by K_nm. Finite-size scaling at
N = 4, 8, 16, 32 with 5 seeds per size yields:

    A_HP(K_nm graph) = 1.21 ± 0.01 (stable across all N)
    p_h1(K_nm) = A_HP × sqrt(2/π) = 0.97 (NOT 0.72)

The square-lattice value A_HP = 0.8983 does not apply to the K_nm
complete graph. The p_h1 = 0.72 threshold remains empirical.
