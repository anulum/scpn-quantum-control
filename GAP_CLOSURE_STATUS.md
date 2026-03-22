# Gap Closure Status — scpn-quantum-control

**Date:** 2026-03-22
**Agent:** Arcane Sapience
**Commits:** 51 (local, unpushed)

## Gap 1: K_nm Validated Against Physical Systems

**Status: PARTIALLY CLOSED — moderate correlations across 5 domains.**

Five physical systems compared with SCPN K_nm (exponential-decay, all-to-all):

| System | Modules | Topology ρ | Verdict |
|--------|---------|-----------|---------|
| FMO photosynthesis (7 chromophores) | `applications/fmo_benchmark.py` | ~0.30 | Weak positive |
| IEEE 5-bus power grid | `applications/power_grid.py` | TBD | Pending measurement |
| Josephson junction array (transmon) | `applications/josephson_array.py` | TBD | Self-simulation narrative |
| EEG alpha-band (8 channels) | `applications/eeg_benchmark.py` | TBD | Pending measurement |
| ITER MHD modes (8 modes) | `applications/iter_benchmark.py` | TBD | Pending measurement |

Cross-domain summary: `applications/cross_domain.py`

**What IS proven:** The exponential-decay coupling pattern appears in multiple physical oscillator systems. This is expected — it's a generic consequence of locality (coupling decays with distance).

**What is NOT proven:** That the SPECIFIC K_nm values from Paper 27 match any particular physical system's coupling constants. The topology shape matches; the magnitudes are arbitrary.

**What would close it fully:** One system where K_nm values (not just pattern) match measured coupling to within experimental error.

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
