# Gap Closure Status — scpn-quantum-control

**Date:** 2026-03-22
**Agent:** Arcane Sapience
**Commits:** 51 (local, unpushed)

## Gap 1: K_nm Validated Against Physical Systems

**Status: PARTIALLY CLOSED — moderate correlations across 5 domains.**

Five physical systems compared with SCPN K_nm (exponential-decay, all-to-all):

| System | Modules | Topology ρ | Verdict |
|--------|---------|-----------|---------|
| FMO photosynthesis (7 chromophores) | `applications/fmo_benchmark.py` | 0.304 | MODERATE |
| IEEE 5-bus power grid | `applications/power_grid.py` | 0.190 | WEAK |
| Josephson junction array (transmon) | `applications/josephson_array.py` | **0.990** | **STRONG** |
| EEG alpha-band (8 channels) | `applications/eeg_benchmark.py` | **0.916** | **STRONG** |
| ITER MHD modes (8 modes) | `applications/iter_benchmark.py` | −0.022 | WEAK |

Cross-domain summary: `applications/cross_domain.py`
Measured: 2026-04-06 by Arcane Sapience using `build_knm_paper27()` and `OMEGA_N_16`.

**What IS proven:** Two systems show strong topology correlation (ρ > 0.5):
1. **Josephson junction array** (ρ=0.990): transmon coupling with all-to-all
   topology matches SCPN K_nm exponential decay almost perfectly. This is
   expected — both use distance-dependent coupling on a complete graph.
2. **EEG alpha-band** (ρ=0.916): neural oscillator PLV coupling structure
   matches SCPN hierarchy. Electrode distance → coupling decay is the
   physical mechanism.

FMO shows moderate correlation (ρ=0.304) — dipole-dipole coupling has
distance dependence but different functional form (1/r³ vs exponential).

IEEE 5-bus (ρ=0.190) and ITER MHD (ρ=−0.022) show weak/no correlation —
these systems have sparse, topology-specific coupling that does not match
the all-to-all exponential-decay structure.

**Honest assessment:** The strong correlations in Josephson and EEG are
real but expected — any system with distance-dependent coupling on a
complete graph will correlate with exponential decay. This does NOT prove
that the SPECIFIC K_nm values from Paper 27 are universal constants.
It proves that the K_nm *pattern* is a reasonable model for systems
where coupling decays with distance.

**What is NOT proven:** That the SPECIFIC K_nm values (K[1,2]=0.302 etc.)
match any physical system's coupling constants. The topology shape
matches; the magnitudes differ by 1-2 orders of magnitude.

**What would close it fully:** One system where K_nm values (not just
pattern) match measured coupling to within experimental error.

**Gap 1 status: PARTIALLY CLOSED (topology match in 2/5 systems).**

## Gap 2: Quantum Result Beyond Classical

**Status: OPEN — honest assessment of current limitations.**

| Evidence | Module | Finding |
|----------|--------|---------|
| DLA analysis | `analysis/dynamical_lie_algebra.py` | 126/255 at N=4 — not trivially simulable but N=4 is classically trivial regardless |
| QFI | `analysis/qfi.py` | ≈ 0 at default params — ground state is product-like |
| MPS baseline | `benchmarks/mps_baseline.py` | Small systems are MPS-tractable |
| QSVT resources | `phase/qsvt_evolution.py` | Optimal algorithm identified |
| Circuit cutting | `hardware/circuit_cutting.py` | 32-64 osc scaling path |
| Koopman | `analysis/koopman.py` | BQP argument via Babbush et al. |

**What IS proven:** The coupled oscillator problem IS BQP-complete (Babbush et al. 2023). The SCPN Hamiltonian has non-trivial DLA. Optimal simulation algorithms (QSVT) are identified.

**What is NOT proven:** That 16 qubits produce a result that classical computers cannot reproduce. At N=16, classical simulation is exact in seconds.

**What would close it:** Scale to N where classical simulation fails (estimated N~30-40 for MPS, N~50+ for exact diagonalisation). Circuit cutting enables this path.

## Gap 3: Derive p_h1 = 0.72 from First Principles

**Status: OPEN — Monte Carlo falsified the square-lattice coincidence.**

The initial finding A_HP(square) × sqrt(2/π) = 0.717 ≈ 0.72 was a coincidence.

Monte Carlo verification on the actual K_nm graph (2026-03-23):

    A_HP (square lattice) = 0.8983 → p_h1 = 0.717 (0.5% from 0.72)
    A_HP (K_nm graph, n=16) = 1.214  → p_h1 = 0.969 (35% from 0.72)

The Hasenbusch-Pinn amplitude is NOT universal across graph topologies.
The complete graph with exponential-decay coupling has a different A_HP
than the square lattice. The 0.5% match was topology-dependent, not a
universal constant.

| Module | Finding |
|--------|---------|
| `analysis/bkt_analysis.py` | T_BKT from Fiedler eigenvalue, bound-pair p_h1 = 0.813 |
| `analysis/bkt_universals.py` | 10 candidate expressions, best = A_HP(sq) × sqrt(2/π) = 0.717 |
| `analysis/monte_carlo_xy.py` | **A_HP(K_nm) = 1.21, p_h1(K_nm) = 0.97 — FALSIFIED** |
| `analysis/p_h1_derivation.py` | Derivation chain valid for square lattice only |

**What IS proven:** The BKT framework correctly describes the K_nm graph
(T_BKT, vortex density, helicity modulus all self-consistent). The
XY model physics works. But A_HP is graph-dependent.

**What is NOT proven:** p_h1 = 0.72 from first principles. It remains
an empirical threshold. The Monte Carlo shows it is NOT a simple
function of BKT universal constants on the K_nm graph.

**What remains open:** Why 0.72? Possible avenues:
1. Finite-size scaling: A_HP(n) → A_HP(∞) may converge differently
2. Noise/disorder effects not captured by clean MC
3. The threshold may relate to a different universality class
4. It may genuinely be empirical (no derivation exists)
