# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Architecture Map

# Architecture Map — capabilities, IO, backends, wiring

This page is the capability/input-output/backend map of `scpn-quantum-control`, written
for **sibling repositories** (sc-neurocore, scpn-control, scpn-fusion-core,
scpn-phase-orchestrator) and for **STUDIO design**. It complements two existing pages:
[Architecture](architecture.md) (structural dependency graph + execution pipeline) and the
[Architecture Decision Records](ARCHITECTURE_DECISIONS.md) (the *why*). This page answers:
**what can I call, what does it consume and produce, on which backend, and how honest is the
claim.**

> **Honesty contract.** Every capability below is graded `mature | functional | scaffold |
> stub | feasibility-only`, and every "closed" claim (pulse-level optimal control, lab-control
> hardware, broad quantum advantage) is marked as such. Where a surface is under active
> refactor it is labelled *volatile*. Counts and bindings drift; re-verify before public reuse.

---

## 1. System in one paragraph

`scpn-quantum-control` turns an **arbitrary coupled-oscillator network** — a coupling matrix
`K_nm` and natural frequencies `omega` — into quantum-control experiments: it compiles the
network to an XY/XXZ Hamiltonian and executable circuits, evolves/varies the state, extracts
synchronisation and quantum-information observables, optionally runs on quantum hardware through
an approval-gated provider layer, applies error mitigation, and records every result under a
five-class evidence ledger. The pipeline is **artefact-first**: each stage emits a typed Python
object or a hash-locked on-disk artefact with provenance.

---

## 2. The canonical data pipeline (INPUTS → OUTPUTS per stage)

```
K_nm,omega ─▶ PROBLEM ─▶ HAMILTONIAN ─▶ CIRCUIT/KERNEL ─▶ EXECUTION ─▶ MITIGATION ─▶ OBSERVABLES ─▶ LEDGER
```

| Stage | Entry symbol(s) | INPUT | OUTPUT | Backend |
|---|---|---|---|---|
| Problem | `build_kuramoto_problem(K_nm, omega, metadata)` · `QPUDataArtifact` · `artifact_to_kuramoto_problem` | `K_nm:(n,n) float64` square/finite/**symmetric**, `omega:(n,)`, JSON metadata | frozen read-only `KuramotoProblem` (diagonal zeroed); or hash-locked on-disk artefact adapted with scalar provenance metadata | Python |
| Hamiltonian | `compile_hamiltonian` · `compile_dense_hamiltonian` · `knm_to_sparse_matrix` · `build_sparse_hamiltonian` | `KuramotoProblem` or arrays | `SparsePauliOp` (XY, Δ=0) · dense `(2ⁿ,2ⁿ) complex128` · sparse CSC | **Rust→Qiskit** (dense Δ=0); budget-guarded |
| Circuit/kernel | `compile_trotter_circuit` · `compile_analog_program` · `compile_hybrid_program` · `phase/*` solvers | `KuramotoProblem`, time/duration, backend route | `qiskit.QuantumCircuit` · analog programme · hybrid programme · trajectory | Qiskit/hardware planners (+Rust order-parameter) |
| Execution | `hardware/runner` · `hardware/hal_*` adapters | circuit + shots + approval | raw counts + job IDs + provenance | IBM Runtime SamplerV2 / 16 provider adapters (approval-gated) |
| Mitigation | `mitigation/*` | counts / per-scale estimates | mitigated estimates **+ uncertainty** | NumPy (citation-backed) |
| Observables | `analysis/*` | counts dict / `Statevector` / dense H | metric dataclasses (R, witnesses, OTOC, DLA…) | NumPy (+Rust on hot probes) |
| Ledger | `docs/hardware_status_ledger.md`, evidence packs | observables + provenance | claim-classified, artefact-backed records | — |

The IO contract of the **programmatic entry** (the load-bearing surface a sibling/STUDIO calls):

```python
build_kuramoto_problem(K_nm, omega, metadata)        # validate → frozen KuramotoProblem
  → artifact_to_kuramoto_problem(artifact)           # validated artifact → KuramotoProblem + provenance
  → compile_hamiltonian(problem)        -> SparsePauliOp                 # XY, Δ=0
  → compile_dense_hamiltonian(problem)  -> ndarray[complex128] (2ⁿ,2ⁿ)  # Rust fast path, budgeted
  → compile_trotter_circuit(problem, time, trotter_steps, trotter_order) -> QuantumCircuit
  → compile_analog_program(problem, platform, duration)                  -> native analog plan
  → compile_hybrid_program(problem, platform, duration)                  -> analog/digital split plan
```

`K_nm` is **symmetrised** `(K+Kᵀ)/2` inside the compiler (the gate-model XY mapping is
Hermitian), so a directed/learned coupling is silently made symmetric — callers passing directed
couplings must be aware. The on-disk `QPUDataArtifact` form is *stricter* (forbids negative `K`,
gates synthetic-vs-real data, hash-locks every array).

---

## 3. Stable public surface (what siblings/STUDIO should depend on)

Two contract layers exist; **prefer the small stable one**:

- **`stable_core`** — durable, layout-independent dataclasses `Problem` / `Backend` /
  `Experiment` / `Result` + `run_stable_core_preflight` (an eligibility gate) +
  `backend_capability_matrix()`. It **compiles nothing**; it is the durable shape and the
  capability/preflight contract. Hardware submission is permitted only for `kind="qiskit"` and
  requires a `preregistration_id` — enforced at construction. This is the v1.0-stable surface
  (see [ADR-0005](ARCHITECTURE_DECISIONS.md)); the broad `__all__` (≈717 top-level symbols) is
  *workbench*, not a stability promise.
- **`kuramoto_core`** — the live compile facade (above). Stable in practice, not yet the formal
  v1.0 contract.

---

## 4. Capability lanes

Each lane: purpose · INPUTS · OUTPUTS · processing model · backends · wiring · SOTA grade.

### 4.1 Core compile + IO contracts — `kuramoto_core`, `bridge/`, `*_budget`
- **INPUTS** `K_nm`,`omega`,metadata (or `QPUDataArtifact`). **OUTPUTS** `SparsePauliOp` / dense /
  sparse CSC / Trotter circuit / physics-informed ansatz.
- **Processing** Kuramoto↔XY: `K·sin(θⱼ−θᵢ) ↔ −J(XᵢXⱼ+YᵢYⱼ)`, `ωᵢ ↔ −hᵢZᵢ`; XXZ adds `Δ·ZZ`.
- **Backends** dense Δ=0 has a **Rust fast path** → Qiskit fallback; sparse has Rust + ARPACK + U(1)
  magnetisation sectors. **Budget guards** (`require_dense_allocation`, `require_pauli_operator_budget`)
  fire before dense Hilbert allocations, Pauli-operator expansion, and the Python sparse fallback's
  full-basis COO loops — fail-closed, RAM-aware, env-overridable.
- **Provenance** `QPUDataArtifact` (hash-locked arrays, synthetic-vs-real + publication-safety gates).
- **SOTA** *mature* (strict validation, budgeted, Rust-accelerated; textbook-correct mapping).

### 4.2 Phase / evolution — `phase/` *(volatile lane)*
- **Time evolution** (*mature*): Qiskit Lie/Suzuki Trotter + native XX+YY compiler; Rust
  order-parameter path; rigorous Trotter-error spectral-norm bounds (Childs et al.).
- **Variational** (*mixed*): functional VQE; **ADAPT-VQE is layered VQE, not gradient-selected**
  (self-disclosed); VarQITE/AVQDS now use the **analytic quantum geometric tensor** — the state
  derivatives are exact via the π-shift identity ``∂_k|ψ> = ½|ψ(θ+π e_k)>`` (shared
  `phase/variational_metric.py`), with no finite-difference bias or step-size; NQS is a toy (exact
  enumeration). `avqds` is fixed-ansatz McLachlan VarQRTE and its docstring states so explicitly
  (no operator-pool growth; parameter count constant, guarded by a test).
- **Open-system** (*strong breadth*): T1/T2 Lindblad, synchronising-dissipator MCWF, collision
  model, PMP/STIRAP pulse shaping. `tensor_jump` honestly **disclaims it is not MPS** (O(2ⁿ)).
- **Tensor-network** (*scope-limited*): real quimb DMRG/TEBD but **nearest-neighbour only**;
  long-range `K_nm` omitted behind an explicit truncation gate.
- **QSVT** (*estimator only*): resource counts; phase synthesis fails closed.
- **Framework bridges**: jax / torch / pennylane / tensorflow agreement-checked against parameter-shift.
- **INPUTS** `KuramotoProblem` / `SparsePauliOp` / time. **OUTPUTS** `QuantumCircuit`, `Statevector`,
  trajectories, gradients. **Backends** Qiskit + Rust + optional jax/torch.

### 4.3 Analysis / observable probes — `analysis/` (+ `dla_parity/`, `gauge/`, `topology_control/`)
- **Substrate**: most probes rest on **exact dense diagonalisation** (`eigh`) gated by
  `dense_budget` → honestly **small-N exact** (N≲12–14 dense, N≤5–6 for 4ⁿ probes); no MPS escape
  hatch is wired (`dla_truncated_tn` is an explicit `NotImplementedError`).
- **INPUTS** counts dict (⟨X⟩,⟨Y⟩) / `Statevector` / dense H / correlators. **OUTPUTS** metric dataclasses.
- **Mature probes**: `dla_parity/` (raw-QPU-counts reproduction + falsification harness — strongest),
  `topology_control/` (constrained PH optimiser), `sync_uncertainty` (delta-method + bootstrap,
  certified coverage), OTOC, Krylov, sector-resolved SFF, QFI/criticality, entanglement
  entropy/spectrum, magic (SRE), Mpemba/NESS, DLA-parity theorem (closed-form, N≤5).
- **Functional**: Loschmidt, BKT, persistent homology (ripser), Hamiltonian learning, Berry, QRC,
  Φ (IIT), ENAQT, gauge/Wilson probes, Monte-Carlo XY.
- **Honest caveats**: `sync_order_parameter` returns the compatibility key `sync_order`
  plus `sync_order_z_magnetisation`, both backed by **Z-magnetisation, not the true X/Y
  Kuramoto R**; `is_xy_kuramoto_order_parameter = 0.0` records that claim boundary in
  result artefacts. The true R exists in `phase` but is not wired to the counts entry point.
  `shadow_tomography`
  is an O(4ⁿ) faithful simulation, **not** the measurement-efficient algorithm; several guarded
  probes fail-closed and refuse synthetic output.
- **Backends** NumPy (+Rust on OTOC/Krylov/DLA/Koopman/MC/sectors).

### 4.4 Execution substrate — `hardware/`, `mitigation/`
- **HAL** = metadata route discovery (**37 provider profiles, 16 broker adapters**) + an
  injected-adapter protocol + an **approval-gated, fail-closed** router; no SDK import at
  discovery. All 16 adapters (IBM/Aer, Braket, IonQ, IQM, Quantinuum, QuEra/Bloqade, Rigetti,
  Pasqal, Azure, qBraid, Strangeworks, D-Wave, OQC, Quandela, Cirq, PennyLane) are **real
  SDK/REST clients**, but require an injected live client + `approval_id` to fire.
- **Runner** (`runner.py`) is a separate direct-IBM path: real SamplerV2/EstimatorV2 (resilience,
  fractional RZZ) with Aer/Basic fallback.
- **Evidence**: provenance capture, 17-field preregistration job dossiers with falsification
  conditions, `_count_integrity` strict shot-conservation at every boundary, offline raw-count
  verifier.
- **Feedback loop** (*cross-shot only, fail-closed*): the classical engine proposes the next
  shot/params; budget/latency/approval gated; honest that intra-shot real-time feedback needs
  provider-side dynamic circuits (not implemented).
- **Mitigation** (*strongest cluster, citation-backed*): ZNE (**+ propagated uncertainty**), PEC,
  DD, readout-matrix, Z₂ symmetry post-selection, CPDR, GUESS, mitiq bridge.
- **CLOSED claims**: pulse-level control and analog Kuramoto are **no-submit / design-only**
  (payload + calibration-dossier construction, never executed); **no lab-control hardware** (NV
  magnetometry is simulation-only; HLS codegen emits source files only).

### 4.5 Acceleration — `accel/`, `scpn_quantum_engine` (Rust) *(volatile lane)*
- **Dispatch**: measured-fastest-first **Rust → Julia → Python** with a mandatory Python floor
  (Go/Mojo probes return `False` — not wired). See [ADR-0002](ARCHITECTURE_DECISIONS.md).
- **Rust engine** (`scpn_quantum_engine`, PyO3 0.29, rayon): **172 `#[pyfunction]` kernels**
  (verified: `#[pyfunction]`, `wrap_pyfunction!`, `add_function`, and the `.pyi` stub all = 172),
  0 `pyclass`. Real kernels for Kuramoto order parameters/gradients/Hessians, Hamiltonian
  construction, OTOC/Krylov/DLA, mitigation, and compiler-AD. *This count is
  actively growing as the accel lane adds kernels — re-verify.*

### 4.6 Differentiable programming — **two distinct surfaces** *(volatile lane)*
1. **Quantum-circuit gradients** (`phase/qnode*`, framework bridges): a **bounded, fail-closed
   parameter-shift** QNN over a fixed gate registry, with jax `custom_vjp` / TF `GradientTape`
   agreement-checked. **Not** a general AD compiler.
2. **Classical whole-program AD compiler** (`differentiable.py`, `program_ad_*`, `compiler/mlir*`):
   a **real general** whole-program automatic-differentiation compiler — NumPy `__array_function__`
   interception, SSA/effect IR, adjoint generation, fwd/rev/JVP/VJP/Hessian/Fisher, and
   **real LLVM/JIT (llvmlite) + Enzyme native-execution paths** — but **bounded** to scalar and
   static dense-linalg operand families (det/inv/solve/trace) and **fail-closed** when the
   toolchain is absent. This is the surface the external review compared to Catalyst/Enzyme.

### 4.7 Domain applications — `applications/`
- **Honest grade (review-confirmed)**: most domain "applications" (FMO, power-grid, ITER, EEG,
  Josephson) are **topology-similarity proxies, not model reproductions** — they compute the
  Spearman ρ of `K_nm` vs a reference coupling matrix + magnitude/frequency ratios; **no** FMO
  transport, swing equation, neural dynamics, Josephson device physics, or MHD dynamics is solved.
  Plugin payloads expose this coefficient as `topology_similarity_proxy`; legacy result objects keep
  `topology_correlation` only as a compatibility alias. They are honestly gated
  (`publication_safe=False`, `source_mode`).
- **Real quantum-compute apps** (*functional, exact statevector*): `quantum_kernel` (Havlíček
  QSVM), `quantum_reservoir` (Fujii–Nakajima), `eeg_classification` (structured-ansatz VQE),
  `q_disruption` (PQC + parameter-shift).

### 4.8 Control — `control/`
- QAOA-MPC, residual VQLS-GS proxy, Petri-net supervisor, ITER disruption, realtime/closed-loop.
- **Honest caveats**: `vqls_gs` keeps the historical "Grad-Shafranov" API name but emits
  `model_boundary="1d_poisson_laplacian_proxy"` and
  `is_full_grad_shafranov_equilibrium=False`; the default path falls back to a classical
  `np.linalg.solve` repair when the VQLS ansatz misses the residual tolerance. `qaoa_mpc`
  reduces to single-qubit Z (separable Ising). Closed-loop analysis is software-in-the-loop,
  fail-closed without a live ticket.

### 4.9 Crypto / QEC / QSNN / gauge — `crypto/`, `qec/`, `qsnn/`, `gauge/`, `psi_field/`
- **Standout** (*mature/production*): `crypto/ml_dsa` is a genuine from-spec **FIPS 204 ML-DSA-65**
  with KAT + ACVP tests. **Entropy**: NIST SP 800-22 QRNG harness.
- QKD modules are real Qiskit computations; their "K_nm secret" security claims are **unproven
  framework assertions** (flagged). QEC carries explicit `CLAIM_BOUNDARY` literals and roadmap
  disclaimers. QSNN = quantum spiking-neural bridge.

---

## 5. Backends and their wiring

| Backend | Where | Dispatch / wiring |
|---|---|---|
| **Rust** `scpn_quantum_engine` | dense Hamiltonian (Δ=0), order params/gradients/Hessians, OTOC/Krylov/DLA, mitigation, compiler-AD | `accel.rust_import.optional_rust_engine`; **first** in fallback; silent `AttributeError` → Python |
| **Julia** (`accel/julia`, juliacall) | order-parameter / mean-field tiers | second tier; ~20 s first-call JIT, amortised |
| **Python/NumPy** | everything; the guaranteed floor | always present |
| **Qiskit + IBM Runtime** | circuits, SamplerV2/EstimatorV2, transpile | `hardware/runner` (direct) + `hal_qiskit` (HAL) — two IBM paths |
| **16 provider adapters** | Braket/IonQ/IQM/Quantinuum/QuEra/Rigetti/Pasqal/Azure/qBraid/Strangeworks/D-Wave/OQC/Quandela/Cirq/PennyLane | real SDK/REST; **approval-gated**, injected client required |
| **Classical refs** (SciPy ODE, QuTiP, quimb MPS, exact statevector) | `stable_core.*_backend`, `benchmarks/classical_baselines` | adoption-path baselines |

Go and Mojo tiers exist as probes only (return `False`).

---

## 6. Cross-repo integration (for siblings)

`bridge/` provides the adapters; all lazy-import the sibling and fall back gracefully:

| Sibling | Adapter | Direction |
|---|---|---|
| sc-neurocore | `bridge/snn_adapter`, `snn_backward`, `spn_to_qcircuit` | spike trains ↔ Ry angles / quantum dense layer |
| scpn-control | `bridge/control_plasma_knm` | plasma-native `K`/`omega` → problem |
| scpn-fusion-core | `bridge/fusion_core_frc` | FRC equilibrium → pulsed-shot QAOA surrogate |
| scpn-phase-orchestrator | `bridge/orchestrator_adapter`, `orchestrator_feedback`, `phase_artifact`, `scpn_upde_edge` | orchestrator state ⇄ `UPDEPhaseArtifact`; `knm.scpn-upde` K_nm/omega edge; quantum R → advance/hold/rollback |
| SSGF | `bridge/ssgf_adapter`, `ssgf_w_adapter` | geometry `W` ↔ Hamiltonian |

A provider-neutral `qpu_compute_types` schema family (content-addressed, idempotency-keyed) exists
for an eventual multi-provider broker but is **not yet wired** to a producer/consumer.

---

## 7. Honest scope boundaries (read before any claim)

- **Executed**: Hamiltonian compilation, Trotter/VQE/analysis on simulators, IBM Runtime
  execution (approval-gated), error mitigation, the evidence ledger, ML-DSA signing, QRNG.
- **Bounded**: the classical whole-program AD compiler (scalar + static dense-linalg, fail-closed);
  tensor-network evolution (nearest-neighbour only); analysis (small-N exact).
- **Feasibility-only / no-submit (CLOSED claims)**: pulse-level optimal control, analog/neutral-atom
  execution, real-time intra-shot feedback, FPGA/HLS deployment, NV-magnetometry hardware.
- **Not present**: lab-control instrumentation; broad quantum advantage (classical solvers are
  faster and more accurate at the reachable sizes n ≤ 16).
- **Domain applications**: mostly topology-similarity proxies, not model reproductions — do not
  cite as solved physics.

For the **per-component SOTA audit, the verified over-claims/gaps, and the systematic per-lane
plan**, see the internal audit record (kept private under `docs/internal/`).
