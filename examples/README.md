# Examples

All examples run on statevector simulation — no IBM Quantum credentials needed.

Install first:
```bash
pip install -e ".[dev]"
```

## Curated Researcher Workflows

Domain-facing workflows with provenance boundaries and deterministic
artefact paths are promoted in `docs/application_benchmarks.md`. The
examples below remain no-credential entry points; the reusable research
logic lives in `src/`, `scripts/`, and committed JSON/CSV artefacts
rather than in untracked notebooks.

Recommended researcher quick path:

- `02_kuramoto_xy_demo.py` for the Kuramoto-XY mapping.
- `05_vqe_ansatz_comparison.py` for topology-informed ansatz behaviour.
- `09_classical_vs_quantum_benchmark.py` for classical/statevector/Trotter comparisons.
- `13_iter_disruption_demo.py` for the plasma/tokamak benchmark boundary.
- `18_end_to_end_pipeline.py` for the packaged workflow path.
- `19_sync_witness_operator.py` for synchronisation observables.
- `20_quantum_persistent_homology.py` for topology diagnostics.
- `22_quantum_neuromorphic_bridge.py` for the QSNN LIF/STDP/dynamic-coupling bridge.
- `23_differentiable_api_workflow.py` for the unified differentiable API workflow.
- `24_differentiable_benchmark_reproduction.py` for local benchmark evidence reproduction.
- `25_qrng_streaming_quickstart.py` for QRNG stream-health checks.
- `27_pqc_trigger_signer_demo.py` for ML-DSA trigger signing.
- `28_pulse_to_hls_quickstart.py` for UltraScale+ pulse-code generation.
- `29_kuramoto_handbook_workflow.py` for the Phase 5 Kuramoto handbook path.
- `30_diff_first_path.py` for the canonical differentiable namespace path.

## Route by Goal

| Goal | Example |
|---|---|
| Understand the core physics | `02_kuramoto_xy_demo.py` |
| Compare ansatze and optimisation behaviour | `05_vqe_ansatz_comparison.py` |
| Inspect no-credential error mitigation | `06_zne_demo.py` |
| Compare classical and quantum execution surfaces | `09_classical_vs_quantum_benchmark.py` |
| Exercise a control-facing domain lane | `13_iter_disruption_demo.py` |
| Run the full packaged workflow | `18_end_to_end_pipeline.py` |
| Inspect synchronisation observables | `19_sync_witness_operator.py` |
| Inspect topology diagnostics | `20_quantum_persistent_homology.py` |
| Inspect differentiable API readiness | `23_differentiable_api_workflow.py` |
| Reproduce differentiable benchmark evidence | `24_differentiable_benchmark_reproduction.py` |
| Inspect QRNG stream readiness | `25_qrng_streaming_quickstart.py` |
| Inspect post-quantum trigger signing | `27_pqc_trigger_signer_demo.py` |
| Generate pulse HLS artefacts | `28_pulse_to_hls_quickstart.py` |
| Run the Kuramoto handbook workflow | `29_kuramoto_handbook_workflow.py` |
| Exercise the canonical differentiable namespace | `30_diff_first_path.py` |

Examples are onboarding aids. Reusable production logic belongs in `src/`,
scripts, committed fixtures, and release gates.

## 01_qlif_demo.py — Quantum LIF Neuron

Constructs a quantum leaky integrate-and-fire neuron that encodes membrane
potential as an Ry rotation angle. Sweeps input current levels and compares
the quantum spike rate (from Z-basis measurement) with the classical
Bernoulli expectation sin^2(theta/2). Demonstrates the probability-to-angle
bridge.

```bash
python examples/01_qlif_demo.py
```

## 02_kuramoto_xy_demo.py — Kuramoto XY Dynamics

Builds a 4-oscillator XY Hamiltonian from the SCPN coupling matrix K_nm
and evolves it via Trotter decomposition. Prints the order parameter R(t)
at each timestep, showing how synchronization emerges from the coupling
topology. This is the core quantum-classical mapping: classical Kuramoto
ODE → quantum XY spin chain.

```bash
python examples/02_kuramoto_xy_demo.py
```

## 03_qaoa_mpc_demo.py — QAOA Binary MPC

Formulates a 4-step binary coil control problem as an Ising Hamiltonian
and solves it with QAOA (p=2 layers). Compares the QAOA-selected action
sequence against brute-force enumeration. Demonstrates mapping a control
optimization problem to a combinatorial quantum algorithm.

```bash
python examples/03_qaoa_mpc_demo.py
```

## 04_qpetri_demo.py — Quantum Petri Net

Creates a 3-place, 2-transition Petri net where tokens are encoded as
qubit amplitudes and transitions fire via controlled rotations. Shows
how superposition of markings enables parallel exploration of the
reachability graph — relevant for modeling concurrent processes in
control systems.

```bash
python examples/04_qpetri_demo.py
```

## 05_vqe_ansatz_comparison.py — VQE Ansatz Benchmark

Compares three VQE ansatze on the 4-qubit Kuramoto Hamiltonian: the
K_nm-informed ansatz (entanglement topology matches coupling graph),
a generic hardware-efficient ansatz, and EfficientSU2. Reports energy,
parameter count, and convergence iterations. The K_nm-informed ansatz
converges fastest — this is the ansatz used in the 0.05% hardware result.

```bash
python examples/05_vqe_ansatz_comparison.py
```

## 06_zne_demo.py — Zero-Noise Extrapolation

Demonstrates error mitigation on a noisy Heron r2 simulator. Applies
global unitary folding at scale factors [1, 3, 5] and uses Richardson
extrapolation to estimate the zero-noise order parameter. Shows how
ZNE can partially recover the ideal result from noisy circuit execution.

```bash
python examples/06_zne_demo.py
```

## 07_crypto_bell_test.py — CHSH Bell Test

Prepares the K_nm ground state via VQE and evaluates the CHSH S-parameter
for nearest-neighbour qubit pairs. Demonstrates entanglement certification
using the coupling topology as shared secret.

```bash
python examples/07_crypto_bell_test.py
```

## 08_dynamical_decoupling.py — DD Fidelity Comparison

Compares the order parameter R with and without XY4 dynamical decoupling
on a 4-qubit Kuramoto circuit under synthetic Heron r2 noise. Shows how
DD pulse insertion partially recovers coherence on NISQ hardware.

```bash
python examples/08_dynamical_decoupling.py
```

## 09_classical_vs_quantum_benchmark.py — Classical vs Quantum

Benchmarks classical Euler ODE, exact matrix-exponential evolution, and
quantum Trotterized XY simulation at N=4, 8, 16. Reports wall-clock time,
R(t) accuracy, and VQE ground-state energy vs exact diagonalization.

```bash
python examples/09_classical_vs_quantum_benchmark.py
```

## 10_identity_continuity_demo.py — Identity Analysis

End-to-end demo of all identity modules: VQE attractor basin, coherence
budget, entanglement witness, spectral fingerprint, and binding spec.

```bash
python examples/10_identity_continuity_demo.py
```

## 11_pec_demo.py — PEC Error Mitigation

Probabilistic error cancellation: quasi-probability coefficients and
Monte Carlo sampling at varying gate error rates.

```bash
python examples/11_pec_demo.py
```

## 12_trapped_ion_demo.py — Trapped-Ion Backend

Synthetic trapped-ion noise model with all-to-all connectivity.
Transpiles and runs Kuramoto circuits under MS gate noise.

```bash
python examples/12_trapped_ion_demo.py
```

## 13_iter_disruption_demo.py — ITER Disruption Classifier

11 physics-based ITER features, synthetic data generation, and quantum
circuit classifier benchmark.

```bash
python examples/13_iter_disruption_demo.py
```

## 14_frc_pulsed_shot_qaoa_demo.py — FRC Pulsed-Shot QAOA

Builds the field-reversed-configuration pulsed-shot QAOA cost with
Magneto-Rayleigh-Taylor growth, dwell-time, and pulsed-field terms, then prints
the bounded simulator-only claim boundary for the control-facing cost path.

```bash
python examples/14_frc_pulsed_shot_qaoa_demo.py
```

## 14_quantum_advantage_demo.py — Quantum Advantage Scaling

Classical vs quantum wall-clock timing with crossover extrapolation.

```bash
python examples/14_quantum_advantage_demo.py
```

## 15_qsnn_training_demo.py — QSNN Training

Parameter-shift gradient descent on QuantumDenseLayer synapse angles.

```bash
python examples/15_qsnn_training_demo.py
```

## 16_fault_tolerant_demo.py — Fault-Tolerant UPDE

Repetition-code logical qubits with syndrome extraction at distance 3 and 5.

```bash
python examples/16_fault_tolerant_demo.py
```

## 17_snn_ssgf_bridges_demo.py — Cross-Repo Bridges

SNN adapter (spike trains → quantum → currents), SSGF adapter (geometry
→ Hamiltonian → roundtrip), and orchestrator phase mapping (18 ↔ 35).

```bash
python examples/17_snn_ssgf_bridges_demo.py
```

## 18_end_to_end_pipeline.py — Full Pipeline

K_nm → VQE → Trotter evolution → ZNE → PEC → classical comparison.
All 5 stages in one script.

```bash
python examples/18_end_to_end_pipeline.py
```

## 19_sync_witness_operator.py — Synchronisation Witness

Constructs the quantum synchronisation witness operator and evaluates
its expectation value across a coupling sweep. A negative witness value
marks entry into the synchronised phase for the demonstrated threshold.

```bash
python examples/19_sync_witness_operator.py
```

## 20_quantum_persistent_homology.py — Quantum Persistent Homology

Generates simulated X/Y-basis measurement counts from Kuramoto--XY
statevectors and extracts persistent homology diagnostics for the
synchronisation transition. Requires `ripser` for the homology backend.

```bash
python examples/20_quantum_persistent_homology.py
```

## 21_biological_qec_scpn16.py — SCPN-Topology QEC

Maps surface-code stabilisers and decoding weights onto the SCPN
16-layer coupling topology, then demonstrates a single-error syndrome
and correction path on that graph.

```bash
python examples/21_biological_qec_scpn16.py
```

## 22_quantum_neuromorphic_bridge.py — QSNN Neuromorphic Bridge

Runs the integrated QSNN bridge: vectorised quantum LIF neurons, trace-based
STDP, and dynamic recurrent coupling driven by spike coactivity and quantum
spike-probability coherence. The demo is simulator-only and prints the explicit
claim boundary; it is a framework primitive, not empirical quantum-biology
evidence by itself.

```bash
python examples/22_quantum_neuromorphic_bridge.py
```

## 23_differentiable_api_workflow.py — Differentiable API Workflow

Runs the complete no-credential differentiable tutorial path: a registered
local Phase-QNode value and parameter-shift gradient, a fail-closed diagnostic
report for an unsupported hardware route, bounded framework dependency rows, a
compiler-AD report, and a tiny bounded phase-QNN training plus gradient
verification. The workflow is functional simulator evidence, not a hardware or
performance benchmark claim.

```bash
python examples/23_differentiable_api_workflow.py
```

## 24_differentiable_benchmark_reproduction.py — Differentiable Benchmark Reproduction

Writes a temporary differentiable benchmark evidence bundle using the public
evidence writer and external-comparison rows, then prints the resulting
classification. Local runs are expected to classify as `functional_non_isolated`
and are not eligible for production performance claims. Promotion still requires
the self-hosted `isolated-benchmark` runner path documented in the benchmark
harness.

```bash
python examples/24_differentiable_benchmark_reproduction.py
```

## 25_qrng_streaming_quickstart.py — QRNG Streaming Quickstart

Runs the deterministic QRNG stream harness and reports NIST-style online
health statistics. The example is a simulator/fixture quickstart; hardware
promotion requires the separate QRNG evidence ledger.

```bash
python examples/25_qrng_streaming_quickstart.py
```

## 26_nv_magnetometry_20T_demo.py — NV Magnetometry 20 T Demo

Generates the NV-centre ODMR calibration surface used by the 20 T magnetometry
readiness lane and records the hardware-blocked calibration boundary.

```bash
python examples/26_nv_magnetometry_20T_demo.py
```

## 27_pqc_trigger_signer_demo.py — PQC Trigger Signer Demo

Signs and verifies deterministic trigger payloads with the ML-DSA trigger
signer interface, exercising the fail-closed message and key-shape contracts.

```bash
python examples/27_pqc_trigger_signer_demo.py
```

## 28_pulse_to_hls_quickstart.py — Pulse to HLS Quickstart

Renders a quantised pulse waveform into the UltraScale+ HLS C++/XDC artefact
set and reports the fixed-point range used by the generated source.

```bash
python examples/28_pulse_to_hls_quickstart.py
```

## 29_kuramoto_handbook_workflow.py — Kuramoto Handbook Workflow

Runs the deterministic six-oscillator Phase 5 handbook path through the public
`scpn_quantum_control.kuramoto` facade: RK4 integration, frequency-locking
diagnostics, stability spectrum, coherence clustering, Gaussian critical
coupling, and projected synchronising-coupling design. The output is stable
JSON for quick inspection and CI regression checks.

```bash
python examples/29_kuramoto_handbook_workflow.py
```
