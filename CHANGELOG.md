# Changelog

All notable changes to scpn-quantum-control are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.5] - 2026-02-28

### Added

- **Second-order Trotter**: `trotter_order=2` parameter on `QuantumKuramotoSolver` and `QuantumUPDESolver` uses SuzukiTrotter(order=2) with O(t^3/reps^2) error vs O(t^2/reps) for first-order
- **Energy tracking**: `QuantumKuramotoSolver.energy_expectation(sv)` returns <H> for paper figure data
- `test_second_order_trotter` — verifies order-2 Trotter error < order-1 on 4-oscillator system
- `test_trotter_error_decreases_with_reps` — verifies convergence: error(reps=8) < error(reps=3) < error(reps=1)
- `test_energy_expectation` — verifies <H> matches direct matrix computation

## [0.2.4] - 2026-02-28

### Fixed

- **QAOA Hamiltonian**: correct Ising encoding with identity (constant) term — h_z = -(a^2-2ab)/2, c0 = (a^2-2ab)*H/2 + H*b^2; removed spurious ZZ terms
- **Quantum Petri net**: multi-input transitions now use multi-controlled Ry (AND gating) instead of single CRy on first input only
- **Inhibitor arcs**: restructured anti-control pattern (X-CRy-X) to correctly gate output on inhibitor place emptiness
- **build_knm_paper27**: removed dead `zeta_uniform` parameter
- **VQLS**: imaginary norm threshold now configurable via `imag_tol` init parameter (default 0.1)

### Added

- `test_hamiltonian_matches_classical_cost` — verifies QAOA Hamiltonian diagonal matches brute-force cost for all bitstrings
- `test_optimal_bitstring_matches_brute_force` — verifies minimum eigenvalue bitstring equals classical optimum
- `test_multi_input_conjunctive_gating` — verifies Petri net output depends on all input places
- `test_inhibitor_blocks_when_place_occupied` — verifies anti-control suppresses output when inhibitor is |1>

## [0.2.3] - 2026-02-28

### Fixed

- **Disruption classifier**: removed dead CX parameter entries — `n_params` was `n_layers*(2*n_qubits + n_qubits-1)` but CX gates have no trainable parameters; now `n_layers*2*n_qubits` (30 vs 42 for default config)
- **QSTDP Hebbian dynamics**: `post_measured` was accepted but ignored; now implements LTP (pre+post → weight increase) and LTD (pre only → weight decrease) per Hebbian learning rule

### Added

- Test for `kuramoto_4osc_zne_experiment` on simulator
- Test for `upde_16_snapshot_experiment` on simulator (marked `@pytest.mark.slow`)
- `pytest.ini_options.addopts` skips slow/hardware tests by default

## [0.2.2] - 2026-02-28

### Fixed

- **MWPM decoder** (3 bugs):
  - `_shortest_path` now uses dual edges for plaquette (Z) syndromes — all single Z errors were failing
  - `_has_logical_error` uses seam-crossing winding number formula; shifted non-contractible cycles no longer missed
  - d=5 now correctly outperforms d=3 below threshold (Dennis et al. 2002)
- **Classical reference endianness** (`_build_initial_state`, `_expectation_pauli`): kron order reversed to match Qiskit little-endian convention; verified against Statevector evolution to 1e-6
- **Parameter-shift rule** (`q_disruption.py`): removed misleading `sin(shift)` denominator (Schuld et al., PRA 99, 032331)
- **VQLS**: assert imaginary norm < 0.1 before `np.real()` instead of silently discarding
- **QAOA_MPC**: removed dead `current_state` parameter
- **ZNE**: cache `base.inverse()` before fold loop
- **trotter_upde**: remove dead `evolve(0)` call, add `reset()`
- **qlif**: replace legacy `np.random.binomial` with seedable `rng` parameter
- **runner**: catch `TranspilerError` in DD pass fallback instead of bare `except Exception`
- **sc_to_quantum**: `measurement_to_bitstream` now accepts `rng` parameter

### Added

- `_run_vqe` helper eliminates vqe_4q/vqe_8q code duplication
- Root `__init__.py` exports all 20 public symbols
- Return type annotations on all public methods
- 9 new tests: d5-beats-d3, shifted logical cycles, single X/Z error correctness, VQE experiment, DD transpile, QSNN stochastic mode, bitstream seeded
- Citation markers on `K_base`, `K_alpha` (Paper 27, Eq. 3)
- `test_classical_evolution_matches_qiskit`: definitive Qiskit-vs-classical endianness agreement test

## [0.2.1] - 2026-02-28

### Fixed

- `q_disruption.py`: configurable `seed` parameter instead of hardcoded 42
- `qpetri.py`: threshold gating via CRy — output rotations controlled by input place qubits
- `classical.py`: sparse eigensolver path (`scipy.sparse.linalg.eigsh`) for N >= 14 or when `k_eigenvalues` specified
- `ansatz_bench.py`: replaced deprecated `TwoLocal`/`EfficientSU2` classes with `n_local`/`efficient_su2` functions (Qiskit 2.1+)
- `vqls_gs.py`: replaced deprecated `TwoLocal` class with `n_local` function

### Added

- STDP direction validation tests (gradient sign at theta=0 and theta=pi/2)
- QEC threshold measurement tests (p=0.01 vs p=0.08 success rates, d=5 single-error decoding, very low error rate high success)
- Disruption classifier seed reproducibility tests
- Petri net controlled output and multi-step bounds tests
- Bloch ball constraint test
- Sparse vs dense eigensolver agreement test

## [0.2.0] - 2026-02-28

### Added

- **hardware/noise_model.py**: `heron_r2_noise_model()` factory with Heron r2 median calibration (T1=300us, T2=200us, CZ 0.5%, readout 0.2%)
- **mitigation/zne.py**: `gate_fold_circuit()` global unitary folding (Giurgica-Tiron et al. 2020), `zne_extrapolate()` Richardson extrapolation
- **mitigation/dd.py**: `DDSequence` (XY4, X2), `insert_dd_sequence()` idle-qubit pulse insertion (Viola et al. 1999)
- **phase/trotter_error.py**: `trotter_error_norm()` exact vs Trotter Frobenius norm, `trotter_error_sweep()` 2D parameter scan
- **phase/ansatz_bench.py**: `benchmark_ansatz()` / `run_ansatz_benchmark()` — Knm-informed vs TwoLocal vs EfficientSU2 VQE comparison
- **hardware/classical.py**: `bloch_vectors_from_json()` per-qubit Bloch vector extraction from hardware result files
- **hardware/experiments.py**: `kuramoto_4osc_zne_experiment` — ZNE-mitigated 4-oscillator dynamics
- **hardware/runner.py**: `run_estimator_zne()` ZNE pipeline, `transpile_with_dd()` via Qiskit's PadDynamicalDecoupling pass, `noise_model` constructor parameter
- **scripts/plot_vqe_convergence.py**: ansatz convergence + energy gap bar chart (Figure 2)
- **scripts/plot_decoherence_curve.py**: `fit_decoherence_rate()` — exp(-gamma*depth) fit for R_hw/R_exact ratio
- **examples/05_vqe_ansatz_comparison.py**, **examples/06_zne_demo.py**
- Top-level re-exports: `OMEGA_N_16`, `build_knm_paper27`, `knm_to_hamiltonian`, `QuantumKuramotoSolver`, `QuantumUPDESolver`, `PhaseVQE`, `HardwareRunner`, `JobResult`

### Fixed

- `_run_sampler_simulator()` now uses `self._backend` instead of fresh `AerSimulator()`, respecting noise model
- Removed duplicate `AerSimulator` import in simulator path
- Removed dead `2**n_osc` expressions in `classical.py` (lines 117, 147)
- `DEFAULT_INSTANCE` reads from `SCPN_IBM_INSTANCE` env var with fallback

### Changed

- mypy scope expanded from `bridge/` to include `phase/`, `mitigation/`, `hardware/classical.py`, `hardware/runner.py`

## [0.1.0] - 2026-02-28

### Added

- **qsnn/**: Quantum LIF neuron (`qlif.py`), controlled-Ry synapse (`qsynapse.py`), parameter-shift STDP (`qstdp.py`), entangled dense layer (`qlayer.py`)
- **phase/**: Kuramoto XY Hamiltonian solver (`xy_kuramoto.py`), 16-layer Trotter UPDE (`trotter_upde.py`), VQE ground state finder (`phase_vqe.py`)
- **control/**: QAOA-MPC binary trajectory optimizer (`qaoa_mpc.py`), VQLS Grad-Shafranov solver (`vqls_gs.py`), quantum Petri net (`qpetri.py`), quantum disruption classifier (`q_disruption.py`)
- **bridge/**: Knm-to-Hamiltonian compiler (`knm_hamiltonian.py`), SPN-to-circuit converter (`spn_to_qcircuit.py`), bitstream-rotation bridge (`sc_to_quantum.py`)
- **qec/**: Toric surface code + MWPM decoder with Knm-weighted edges (`control_qec.py`)
- **hardware/**: IBM Quantum runner for ibm_fez Heron r2 (`runner.py`, `experiments.py`, `classical.py`)
- 88 unit tests, 4 example scripts, 19 hardware result files
- Hardware validation on ibm_fez: VQE 0.05% error, 12-point decoherence curve, first 16-layer SCPN quantum simulation
- CI workflow with Python 3.9-3.12 matrix, coverage, ruff lint
- Full documentation: architecture, API reference, hardware results
