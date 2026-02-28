# Changelog

All notable changes to scpn-quantum-control are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
