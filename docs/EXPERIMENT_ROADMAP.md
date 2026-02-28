# Experiment Roadmap: March-June 2026

Backend: ibm_fez (Heron r2, 156 qubits)
Budget: 10 min QPU / month (free tier)
Repository: scpn-quantum-control v0.3.0

## Experiment Inventory (17 total)

| # | Experiment | Qubits | Est. QPU | Month | Science Question |
|---|-----------|--------|----------|-------|-----------------|
| 1 | `noise_baseline` | 4 | 10s | Mar | Calibration drift Feb→Mar |
| 2 | `kuramoto_4osc_zne` [1,3,5] | 4 | 90s | Mar | Raw vs mitigated (linear ZNE) |
| 3 | `kuramoto_8osc_zne` [1,3,5] | 8 | 120s | Mar | Extend mitigation to depth-233 |
| 4 | `vqe_8q_hardware` | 8 | 60s | Mar | Scale VQE from 4→8 qubits |
| 5 | `upde_16_dd` | 16 | 60s | Mar | DD on full 16-layer system |
| 6 | `kuramoto_4osc_trotter2` | 4 | 30s | Mar | Order-2 vs order-1 Trotter |
| 7 | `sync_threshold` | 4 | 50s | Apr | Kuramoto phase transition |
| 8 | `ansatz_comparison_hw` | 4 | 90s | Apr | Prove Knm ansatz wins on hardware |
| 9 | `zne_higher_order` [1..9] | 4 | 120s | Apr | Optimal extrapolation order |
| 10 | `decoherence_scaling` | 2-12 | 120s | Apr | Extract per-gate decay rate γ |
| 11 | `vqe_landscape` | 4 | 0s | Apr | Barren plateau detection (sim only) |
| 12 | `kuramoto_4osc` | 4 | 30s | — | Baseline (Feb, done) |
| 13 | `kuramoto_8osc` | 8 | 60s | — | 8-osc trajectory (Feb, done) |
| 14 | `vqe_4q` | 4 | 30s | — | VQE ground state (Feb, done) |
| 15 | `vqe_8q` | 8 | 0s | — | Statevector only (Feb, done) |
| 16 | `qaoa_mpc_4` | 4 | 20s | — | Binary MPC (Feb, done) |
| 17 | `upde_16_snapshot` | 16 | 180s | — | 16-layer snapshot (Feb, done) |


## March 2026 Plan (~370s QPU)

### Priority experiments

**1. noise_baseline** (10s) — Repeat Feb depth-5 circuit. Detects backend
drift. Compare hw_R to Feb value (0.8054). If drift > 2%, flag all
subsequent results for calibration correction.

**2. kuramoto_4osc_zne [1,3,5]** (90s) — Gate-fold at 3 noise scales,
Richardson linear extrapolation. Feb baseline: 7.3% error at depth 85.
Expected: ZNE reduces to ~3-4%. Produces the "raw vs mitigated" figure
for the paper.

**3. kuramoto_8osc_zne [1,3,5]** (120s) — Same ZNE protocol at 8 qubits.
Feb: 20% error at depth 233. Expected: ZNE reduces to ~10-12%. Key
question: does ZNE maintain effectiveness at higher depth?

**4. vqe_8q_hardware** (60s) — Statevector VQE optimization followed by
single-shot hardware energy evaluation. Feb 4q result: 0.05% error.
Expected 8q: 0.1-0.5% error (deeper ansatz, more CZ gates). Establishes
scaling trend for VQE accuracy.

**5. upde_16_dd** (60s) — XY4 dynamical decoupling on all 16-qubit UPDE
basis circuits. Compares R(DD) vs R(no-DD) vs classical. Tests whether
idle-qubit DD helps at depth ~770 where decoherence dominates.

**6. kuramoto_4osc_trotter2** (30s) — Suzuki-Trotter order 2 at identical
dt/steps as order 1. Direct comparison: does higher Trotter order gain
accuracy faster than decoherence penalty from deeper circuits?

**Total: ~370s. Buffer: ~230s for reruns or calibration.**


## April 2026 Plan (~380s QPU)

### Priority experiments

**7. sync_threshold** (50s) — Sweep K_base ∈ {0.05, 0.15, 0.30, 0.45,
0.60, 0.80} at 4 qubits. Each value = 3 circuits (Z/X/Y) at dt=0.1.
Maps the Kuramoto synchronization bifurcation on quantum hardware: below
critical coupling K_c, R stays low; above K_c, R jumps to order ~0.5+.

Science value: first quantum measurement of the Kuramoto phase transition.
Validates quantum XY ↔ classical Kuramoto correspondence at the critical
point. Publication-quality standalone result.

**8. ansatz_comparison_hw** (90s) — Three VQE ansatze (Knm-informed,
TwoLocal, EfficientSU2) optimized on Statevector, then evaluated on
hardware via Estimator. Compares hw_energy for each.

Science value: Feb sim-only benchmark showed Knm-informed converges
fastest. This proves the advantage survives real hardware noise.
Physics-informed circuit design is a hot topic — demonstrating it on
real hardware is publishable in PRX Quantum or similar.

**9. zne_higher_order [1,3,5,7,9]** (120s) — 5-point ZNE with polynomial
orders 1 (linear) and 2 (quadratic). Tests whether higher-order
extrapolation recovers more signal or overfits.

Science value: systematic ZNE study on the same circuit family.
Determines optimal extrapolation strategy for XY evolution on Heron r2.
Contributes to the error mitigation literature.

**10. decoherence_scaling** (120s) — Run 1-Trotter-step evolution at
2, 4, 6, 8, 10, 12 qubits. Records depth and R for each. Fits
R_hw = R_exact * exp(-γ * depth) to extract per-gate depolarization
rate γ.

Science value: γ is a single number that characterizes the backend for
our circuit family. Enables predictive modeling: "at depth D, expect
error E." Compares to IBM published T1/T2 and gate errors.

**11. vqe_landscape** (0s QPU, sim only) — Sample 50 random parameter
vectors for each ansatz, compute energy variance. Low variance = barren
plateau. Tests whether Knm-informed ansatz avoids barren plateaus.

Science value: barren plateaus are the #1 obstacle to VQE scaling.
Showing Knm-informed ansatz has higher variance (trainable landscape) is
publishable. Reference: McClean et al., Nature Comm. 9, 4812 (2018).

**Total: ~380s. Buffer: ~220s.**


## May 2026 Plan (Candidates)

### Experiment ideas (not yet implemented)

**A. Layer-selective qubit assignment** — On Heron r2, assign
strongly-coupled SCPN layers (L3, L4, L10) to the lowest-error physical
qubits. Compare UPDE-16 R vs default Qiskit layout.

Budget: ~120s QPU. Requires reading backend calibration data
(`backend.properties()`) to rank qubits by error rate.

**B. Readout error mitigation (M3)** — Build measurement calibration
matrix from all-0 and all-1 circuits, apply inverse to 4-osc and 8-osc
counts. Separates readout error from gate error.

Budget: ~30s QPU (calibration circuits) + reprocessing of existing data.

**C. Entanglement entropy measurement** — Compute von Neumann entropy
S(ρ_q) of per-qubit reduced density matrices during Kuramoto evolution.
Requires randomized measurements (shadow tomography) or direct
state tomography on ≤4 qubits.

Budget: ~180s QPU for 4-qubit partial tomography at 3 time steps.

**D. QAOA with ZZ coupling terms** — Extend binary MPC cost Hamiltonian
to include inter-timestep correlations (ZZ terms). Tests whether richer
cost structure improves QAOA control quality.

Budget: ~60s QPU. Requires extending `QAOA_MPC.build_cost_hamiltonian()`.

**E. Depth-optimal circuit decomposition** — Rewrite Kuramoto evolution
using native Heron r2 CZ+RZ+SX gate set directly (skip Trotter
synthesis). Hand-optimized 4-qubit circuit should be 40-60% shallower.

Budget: ~30s QPU. High implementation effort but large depth savings.


## June 2026 Plan (Candidates)

**F. Quantum phase estimation** — QPE on the XY Hamiltonian to extract
eigenvalues. 4 qubits + 3 ancilla. Tests whether QPE is practical on
NISQ for our Hamiltonian.

**G. Variational quantum simulation** — VQS (Li-Benjamin algorithm) as
alternative to Trotter. Uses ansatz evolution instead of product
formulas. Should be shallower for same accuracy.

**H. Multi-circuit quantum error correction** — Run toric code d=3 with
MWPM decoder using Knm-weighted distances. Tests whether physics-aware
decoding reduces logical error rate.


## Paper Claims Strengthened by This Roadmap

| Claim | Feb Data | March+ Extension |
|-------|----------|------------------|
| VQE 0.05% | 4q | 8q (scaling proof) |
| Decoherence curve | 12 pts, qualitative | Fit γ, R² (quantitative) |
| 16-layer UPDE | 46% error | DD + ZNE (reduced error) |
| Trotter tradeoff | Order 1 only | Order 1 vs 2 comparison |
| Error mitigation | None | ZNE linear + quadratic |
| Ansatz design | Sim only | Hardware proof |
| Phase transition | Not measured | Bifurcation curve |
| Barren plateaus | Not measured | Landscape variance |


## Hardware Notes

- **Backend**: ibm_fez, Heron r2, 156 qubits
- **Native gates**: CZ, ID, RZ, SX, X
- **Median CZ error**: ~0.5%
- **T1/T2**: ~300/200 μs
- **Coherence wall**: depth 250-400 (Feb calibration)
- **Budget**: 10 min QPU/month (free tier)
- **Scheduling**: first-come-first-served, typical queue 30s-5min
