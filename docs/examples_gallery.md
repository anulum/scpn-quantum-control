# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Example Gallery

# Example Gallery

Thirty task-shaped entry points, ordered from the fastest
no-credential run to hardware-evidence, integration, and release-readiness
surfaces. Each entry states what it does, when to reach for it, and the fastest
safe command. Every command here runs on the statevector simulator or on
committed artefacts; none needs IBM Quantum credentials.

Install the development extra once:

```bash
pip install -e ".[dev]"
```

The examples are onboarding aids. Reusable logic lives in `src/`, `scripts/`,
committed fixtures, and release gates — see [Onboarding](onboarding.md) for the
claim boundaries that separate simulation, method verification, hardware
evidence, and commercial readiness.

## 1. Quick simulator run

**What:** build a 4-oscillator XY Hamiltonian from `K_nm` and evolve it on the
statevector simulator, printing the order parameter at each step.
**When:** your first run, to confirm the install and see the core mapping.

```bash
python examples/02_kuramoto_xy_demo.py
```

## 2. Kuramoto-XY compile

**What:** turn a coupling matrix and frequency vector into a Trotter circuit
through the stable facade.
**When:** you want the circuit object to inspect, transpile, or run elsewhere.

```python
import numpy as np
from scpn_quantum_control import KuramotoProblem, build_knm_paper27, compile_trotter_circuit

problem = KuramotoProblem(K_nm=build_knm_paper27(L=4), omega=np.linspace(0.1, 0.4, 4))
circuit = compile_trotter_circuit(problem, time=1.0, trotter_steps=5)
print(circuit.num_qubits, circuit.depth())
```

See the [Kuramoto Core Facade](kuramoto_core_facade.md).

## 3. Synchronisation witness extraction

**What:** build synchronisation witness operators and read their expectation on
an evolved state.
**When:** you need an observable that certifies phase locking rather than a raw
amplitude.

```bash
python examples/19_sync_witness_operator.py
```

## 4. Order parameter

**What:** evolve the system and read the Kuramoto order parameter `R(t)` through
the solver.
**When:** you care about the synchronisation trajectory, not the full state.

```python
import numpy as np
from scpn_quantum_control import build_knm_paper27
from scpn_quantum_control.phase import QuantumKuramotoSolver

solver = QuantumKuramotoSolver(4, build_knm_paper27(L=4), np.linspace(0.1, 0.4, 4))
result = solver.run(t_max=1.0, dt=0.2, trotter_per_step=3)
print(result["R"])
```

## 5. Rust-accelerated Hamiltonian path

**What:** assemble the XY Hamiltonian; the optional Rust engine accelerates the
build and falls back to NumPy when the engine wheel is absent.
**When:** you build many Hamiltonians or larger systems and want the fast path.

```python
import numpy as np
from scpn_quantum_control import build_knm_paper27, knm_to_hamiltonian

hamiltonian = knm_to_hamiltonian(build_knm_paper27(L=4), np.linspace(0.1, 0.4, 4))
print(len(hamiltonian))
```

See [Pipeline Performance](pipeline_performance.md) for the measured backend
ordering.

## 6. Kuramoto handbook workflow

**What:** run the deterministic six-oscillator handbook workflow through the
public Kuramoto facade and print stable JSON diagnostics for integration,
frequency locking, stability, clusters, critical coupling, and coupling design.
**When:** you need the Phase 5 API path in one executable, no-credential command.

```bash
python examples/29_kuramoto_handbook_workflow.py
```

See the [Kuramoto Handbook](kuramoto_handbook.md) and the companion notebook
`notebooks/48_kuramoto_handbook_workflow.ipynb`.

## 7. Hardware-pack verification

**What:** verify the integrity and promotion status of committed hardware result
packs through the packaged CLI.
**When:** you want to confirm which hardware claims are artefact-backed before
quoting them.

```bash
scpn-verify-hardware-packs --help
```

See [Hardware Result Packs](hardware_result_packs.md).

## 8. Classical baseline comparison

**What:** compare the exact reference, the SciPy ODE baseline, and the
statevector Trotter route, and emit a reproducible artefact with documented
failure modes.
**When:** you need an honest classical-vs-quantum reference with a fixed claim
boundary.

```bash
python examples/09_classical_vs_quantum_benchmark.py \
    --artifact data/classical_quantum_comparison/reproducible_comparison_n8.json
```

For the system sizes this comparison can run (`n <= 16`) the classical exact
route is faster and exact; the quantum route is an evidence and falsification
surface, not a generic speed-up. See [Classical Baselines](classical_baselines.md).

## 9. Differentiable parameter-shift

**What:** walk the unified differentiable API, including parameter-shift
gradients with fail-closed framework boundaries.
**When:** you want gradients of a circuit observable for optimisation or training.

```bash
python examples/23_differentiable_api_workflow.py
```

See the [Differentiable API](differentiable_api.md).

## 10. Provider / HAL dry-run

**What:** exercise the provider hardware-abstraction layer without submitting a
job, recording capability and fail-closed boundaries.
**When:** you want to check provider readiness before any credentialled run.

```bash
scpn-provider-smoke --help
```

See [QPU Provider Readiness](qpu_provider_readiness.md).

## 11. Evidence replay

**What:** regenerate the committed methods/FIM benchmark artefacts from source,
without any IBM submission, so a ledger-promoted result can be reproduced.
**When:** you want to reproduce committed evidence from committed inputs.

```bash
scpn-bench reproduce-methods
```

See the [Hardware Status Ledger](hardware_status_ledger.md).

## 12. Probabilistic error cancellation

**What:** demonstrate PEC mitigation accounting on a local simulator route.
**When:** you need mitigation vocabulary and artefact shape before a hardware
campaign.

```bash
python examples/11_pec_demo.py
```

## 13. Trapped-ion workflow

**What:** exercise the trapped-ion-oriented compilation/demo path without a live
provider submission.
**When:** you are comparing provider-specific feasibility surfaces.

```bash
python examples/12_trapped_ion_demo.py
```

## 14. ITER disruption workflow

**What:** run the plasma-control demonstration path for ITER-style disruption
risk modelling.
**When:** you are evaluating application-plugin assumptions, not claiming
validated fusion control.

```bash
python examples/13_iter_disruption_demo.py
```

## 15. FRC pulsed-shot QAOA

**What:** build a pulsed-shot QAOA schedule for the FRC surrogate lane.
**When:** you need the control-scheduler contract before importing
SCPN-FUSION-CORE-derived calibration inputs.

```bash
python examples/14_frc_pulsed_shot_qaoa_demo.py
```

See [FRC Pulsed-Shot QAOA](frc_pulsed_qaoa.md).

## 16. Quantum-advantage boundary check

**What:** compare the current quantum route against classical baselines under
the package's explicit no-broad-advantage boundary.
**When:** you need falsification or due-diligence language rather than speed-up
marketing.

```bash
python examples/14_quantum_advantage_demo.py
```

## 17. QSNN training

**What:** run the quantum spiking-neural-network training demonstration.
**When:** you are testing differentiable-training evidence surfaces.

```bash
python examples/15_qsnn_training_demo.py
```

## 18. Fault-tolerant planning

**What:** inspect the fault-tolerant workflow scaffold and its claim boundary.
**When:** you need planning artefacts without implying deployed logical
hardware.

```bash
python examples/16_fault_tolerant_demo.py
```

## 19. SNN / SSGF bridges

**What:** demonstrate bridge contracts between quantum-control outputs and
spiking/field-model consumers.
**When:** you are checking cross-repository payload shape.

```bash
python examples/17_snn_ssgf_bridges_demo.py
```

## 20. End-to-end pipeline

**What:** run the local no-credential pipeline from model input to analysis
output.
**When:** you need a single command for integration smoke testing.

```bash
python examples/18_end_to_end_pipeline.py
```

## 21. Sync witness operator

**What:** construct and evaluate synchronisation witness operators.
**When:** you need observable evidence instead of raw state amplitudes.

```bash
python examples/19_sync_witness_operator.py
```

## 22. Quantum persistent homology

**What:** run the topology-analysis demonstration for quantum-state or network
features.
**When:** you are exploring topological diagnostics as method evidence.

```bash
python examples/20_quantum_persistent_homology.py
```

## 23. Biological QEC

**What:** exercise the SCPN-16 biological-QEC reporting path.
**When:** you need the bounded report format without promoting biological or
clinical claims.

```bash
python examples/21_biological_qec_scpn16.py
```

## 24. Quantum neuromorphic bridge

**What:** demonstrate neuromorphic bridge payloads and conversion surfaces.
**When:** you are integrating with downstream neuromorphic packages.

```bash
python examples/22_quantum_neuromorphic_bridge.py
```

## 25. Differentiable API workflow

**What:** walk the differentiable API from objective definition to evidence
record.
**When:** you need the supported user path before dropping into internals.

```bash
python examples/23_differentiable_api_workflow.py
```

## 26. Differentiable benchmark reproduction

**What:** reproduce committed differentiable benchmark evidence.
**When:** you need local regression evidence tied to the classification ledger.

```bash
python examples/24_differentiable_benchmark_reproduction.py
```

## 27. QRNG streaming quickstart

**What:** sample simulator-backed quantum random bits and run FIPS/NIST health
checks.
**When:** you need entropy stream plumbing and health-report contracts.

```bash
python examples/25_qrng_streaming_quickstart.py
```

See [Quantum Random-Number Generation](entropy_qrng.md).

## 28. NV-centre 20 T magnetometry

**What:** simulate ODMR resonances and calibration across the high-field
magnetometry range.
**When:** you need the sensing contract before attaching real calibration
evidence.

```bash
python examples/26_nv_magnetometry_20T_demo.py
```

See [NV-Centre 20 T Magnetometry](nv_magnetometry_20T.md).

## 29. PQC trigger signer

**What:** generate ML-DSA-65 keys and sign a capacitor-bank trigger payload.
**When:** you need pre-arm authorisation evidence and freshness checks.

```bash
python examples/27_pqc_trigger_signer_demo.py
```

See [Post-Quantum Trigger Signer](ml_dsa_pqc.md).

## 30. Pulse to UltraScale+ HLS

**What:** convert a pulse envelope into Vivado/Vitis HLS source and host
co-simulation files.
**When:** you need FPGA source generation without invoking Vivado.

```bash
python examples/28_pulse_to_hls_quickstart.py
```

See [Pulse -> UltraScale+ HLS Codegen](ultrascale_hls.md).

## 31. QFI/FSS differentiable evidence

**What:** run the unified differentiable QFI/FSS finite-size report and inspect
bounded BKT and inverse-size fit diagnostics.
**When:** you need small local Kuramoto-XY criticality evidence with explicit
non-hardware, non-performance, and non-thermodynamic-limit boundaries.

```bash
python examples/31_qfi_fss_differentiable_report.py
```

See [Differentiable API](differentiable_api.md).
