<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts & Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control — Reproducibility Record -->

# REPRODUCIBILITY RECORD — SCPN Quantum Control Frontier Campaign 2026

**Document version:** 2.0
**Date:** 2026-04-26
**Author:** Miroslav Šotek / SCPN Quantum Control project
**SPDX:** AGPL-3.0-or-later

---

## 1. Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| Qiskit | 2.4.0 |
| qiskit-ibm-runtime | 0.46.1 |
| NumPy | 2.2.6 (pip) |
| mitiq | 1.0.0 |
| cirq-core | 1.6.1 |
| matplotlib | 3.10.9 (pip) |
| scpn-quantum-control | v0.9.6 (this repo) |

Install the exact environment:
```bash
pip install qiskit==2.4.0 qiskit-ibm-runtime==0.46.1 numpy==2.2.6 mitiq==1.0.0
pip install -e .    # installs scpn-quantum-control from this repo
```

### mpl_toolkits / mitiq system fix (Linux only)

On Ubuntu 24.x, the system `python3-matplotlib` apt package installs an incompatible
`mpl_toolkits` at `/usr/lib/python3/dist-packages/mpl_toolkits/` that is incompatible
with pip matplotlib ≥ 3.8. This causes `mitiq` (which depends on `cirq`) to fail to import
with `ModuleNotFoundError: No module named 'matplotlib.tri.triangulation'`.

**Fix applied 2026-04-26:**
```bash
# 1. Force-reinstall pip matplotlib to get compatible mpl_toolkits
pip install --upgrade matplotlib --force-reinstall --break-system-packages

# 2. Copy pip-installed mplot3d into system location (overwrite stale version)
sudo mkdir -p /usr/lib/python3/dist-packages/mpl_toolkits/mplot3d
sudo cp -a ~/.local/lib/python3.12/site-packages/mpl_toolkits/mplot3d/. \
           /usr/lib/python3/dist-packages/mpl_toolkits/mplot3d/

# 3. Clear pyc caches
sudo find /usr/lib/python3/dist-packages/mpl_toolkits -name "*.pyc" -delete
sudo find /usr/lib/python3/dist-packages/mpl_toolkits -name "__pycache__" -exec rm -rf {} +
```

After this fix: `import mitiq; from mitiq import zne` imports cleanly.

---

## 2. Hardware

| Field | Value |
|-------|-------|
| Provider | IBM Quantum (ibm_cloud channel) |
| Backend alias | `ibm_heron_r2` → resolved to `ibm_fez` |
| Architecture | IBM Heron r2 |
| Qubit count | 156 |
| Native gate set | CZ, RZ, SX, X |
| Shots per job | min(requested, 4000) |

---

## 3. Circuit Construction — `StructuredAnsatz.from_kuramoto`

**File:** `src/scpn_quantum_control/control/structured_ansatz.py`

The Trotterised Kuramoto-XY circuit is built for an N-qubit system as follows:

1. **Initial state:** Hadamard on all qubits → uniform phase superposition
2. **Scale coupling:** `K_scaled = K_nm × coupling_scale` (applied once before the loop)
3. **Trotter loop** (repeated `trotter_depth` times, `dt = time_step`):
   - **Frequency term:** `RZ(2·ωᵢ·dt, i)` for each qubit i
   - **XY coupling:** `RZZ(2·K_scaled[i,j]·dt, i, j)` for all pairs with |K_scaled[i,j]| > 1e-8
   - **FIM feedback (optional):** `RZ(λ_fim·dt, i)` — `λ_fim` is a **concrete float** (not a Qiskit
     Parameter) to prevent name-collision in Qiskit ≥ 2.x

### Parameters

| Parameter | Batches 1–4 | Batch 5+ |
|-----------|------------|---------|
| `trotter_depth` | 8 | 8 |
| `time_step` | 0.1 | 0.1 |
| `lambda_fim` | 0.0 | 0.0 |
| `coupling_scale` | **1.0** (implicit, no parameter existed) | **2.0** (default) |

> **Note on `coupling_scale`:** Added 2026-04-26. Default `2.0` doubles K_nm before circuit
> construction, pushing the system toward the Kuramoto synchronisation threshold. Stored in
> `ansatz.params` for reproducibility. All batches 1–4 used `coupling_scale=1.0` (parameter
> did not exist; K_nm was used unscaled).

---

## 4. Parameter Files

Frontier `.npy` files are a local transport cache, not publication
source material. They must be generated from the bridge/provenance
contract in `scripts/frontier_campaign_2026/generate_params.py`.

Default behaviour is fail-closed:

```bash
python scripts/frontier_campaign_2026/generate_params.py --output-dir params
```

The command raises if a bridge is unavailable or does not provide
`omega`. It does not silently invent matrices.

Deterministic smoke-test arrays require explicit opt-in:

```bash
python scripts/frontier_campaign_2026/generate_params.py \
  --output-dir params \
  --allow-synthetic \
  --seed 42
```

Synthetic runs write `PARAMETER_PROVENANCE.json` with
`source_mode="synthetic"` entries. These files are suitable for interface
tests only and are not publication-safe QPU inputs.

---

## 5. Error Mitigation — All Batches

### Batch 1 — 2026-04-26 (T4: 80 jobs, T7: 9 jobs, T1: 1 orphan)
| Setting | Value |
|---------|-------|
| `optimization_level` | 1 |
| Dynamical decoupling | **None** |
| ZNE | **None** |
| `coupling_scale` | 1.0 (not yet added) |

### Batch 2 — 2026-04-26 (T1: 3 jobs, T4: 80 jobs, T7: 9 jobs)
| Setting | Value |
|---------|-------|
| `optimization_level` | **3**, `seed_transpiler=42` |
| Dynamical decoupling | **Skipped** — DD `PassManager` not yet in code |
| ZNE | **None** |
| `coupling_scale` | 1.0 (not yet added) |

### Batch 3 — 2026-04-26 (T1: 3 jobs, T4: 80 jobs, T7: 9 jobs)
| Setting | Value |
|---------|-------|
| `optimization_level` | **3**, `seed_transpiler=42` |
| Dynamical decoupling | **Skipped** — `ALAPScheduleAnalysis` import missing from `_run_blocking` scope |
| ZNE | **None** |
| `coupling_scale` | 1.0 (not yet added) |

### Batch 4 — 2026-04-26 (T1: 3 jobs, T4: 80 jobs, T7: 9 jobs = 92 total)
| Setting | Value |
|---------|-------|
| `optimization_level` | **3**, `seed_transpiler=42` |
| Dynamical decoupling | ✅ **X–X sequence, verified** — `PassManager([ALAPScheduleAnalysis(durations, target), PadDynamicalDecoupling(durations, [XGate(),XGate()], target)]).run(isa_qc)` — circuit depth 1922 → 1988 confirmed on ibm_fez |
| ZNE | **None** (mitiq broken at time of run) |
| `coupling_scale` | 1.0 (not yet added) |

> **DD API note (Qiskit 2.4.0):** `PadDynamicalDecoupling` and `ALAPScheduleAnalysis` must be
> chained via `PassManager.run()` — calling them directly on a circuit raises `AttributeError`.
> `spacing` must be `list[float]` not a string. Neither pass accepts `backend` as a positional arg.
> Three iterations of debugging were required to establish the correct pattern.

### Batch 5 — 2026-04-26 (T1: 3 jobs, T4: 80 jobs, T7: 9 jobs = 92 total) ← CURRENT
| Setting | Value |
|---------|-------|
| `optimization_level` | **3**, `seed_transpiler=42` |
| Dynamical decoupling | ✅ **X–X sequence** (same verified PassManager pattern as batch 4) |
| ZNE | ✅ **Richardson extrapolation, scale factors [1, 2, 3], `fold_global`** |
| `coupling_scale` | ✅ **2.0** (default; K_nm doubled before circuit construction) |

**ZNE implementation (mitiq 1.0.0, verified API):**
```python
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory

def _zne_executor(circ):
    counts = sampler.run([circ]).result()[0].data.meas.get_counts()
    return SyncOrderParameter()(counts=counts)["sync_order"]  # scalar float

zne_sync_order = zne.execute_with_zne(
    isa_qc,
    _zne_executor,
    factory=RichardsonFactory([1, 2, 3]),
    scale_noise=fold_global,
)
```
ZNE runs 3 circuits (scale=1,2,3), extrapolates `sync_order` to zero-noise limit.
A final scale=1 run collects full counts for `DLAParityWitness`, `IntegratedInformationPhi`.
Result JSON includes `zne_applied=True`, `zne_scale_factors=[1,2,3]`, `zne_factory="RichardsonFactory"`.

---

## 6. Cross-Batch Results — T4 `SyncOrderParameter` (80 jobs each)

| Batch | `coupling_scale` | Mitigation | sync mean | sync std | Δ vs B1 |
|-------|-----------------|-----------|-----------|----------|---------|
| 1 | 1.0 | opt_level=1, no DD, no ZNE | 0.1409 | 0.0241 | baseline |
| 2 | 1.0 | opt_level=3, no DD, no ZNE | 0.0867 | 0.0071 | −38% |
| 3 | 1.0 | opt_level=3, no DD, no ZNE | 0.0867 | 0.0071 | −38% (reproducible) |
| 4 | 1.0 | opt_level=3 + DD X–X, no ZNE | **0.0066** | **0.0037** | **−95%** |
| 5 | **2.0** | opt_level=3 + DD X–X + ZNE | _pending_ | _pending_ | — |

**Key finding batch 4:** DD reduces spurious sync_order by 95% relative to batch 1. Without DD,
idle qubit dephasing and crosstalk create false correlations that inflate the magnetisation signal.
With DD, sync_order approaches the physical noise floor (~0.007), consistent with a disordered
Kuramoto system below the synchronisation threshold at these coupling strengths.

**Batch 5 prediction:** `coupling_scale=2.0` doubles all RZZ angles; combined with ZNE Richardson
extrapolation, sync_order should increase above batch 4 if the system is genuinely pushed toward
the phase-coherent regime (Kuramoto critical point ~K_c ≈ 2σ_ω/π).

---

## 7. Observables — Scientific Classification

| Class | Computes from real counts | Description |
|-------|--------------------------|-------------|
| `SyncOrderParameter` | ✅ YES | Kuramoto order parameter from bitstring marginals |
| `DLAParityWitness` | ✅ YES | Odd/even Hamming-weight parity asymmetry |
| `IntegratedInformationPhi` | ✅ YES (proxy) | Normalised Shannon entropy; saturates ~1.0 at NISQ shot counts |
| `QuantumFisherInformation` | ⚠️ PROXY | Analytic estimate from sync_order + dla_asymmetry |
| `ThermodynamicWitness` | ⚠️ PROXY | Model estimate using `kwargs.get("work", 1.2)` |
| `LogicalSyncWitness` | ⚠️ PROXY | Model estimate using `kwargs.get("logical_fidelity", 0.92)` |

> **Publication guidance:** Only ✅ observables are attributable to real QPU measurements.
> ⚠️ PROXY observables must be labelled as model estimates.

---

## 8. Reproducing Results

```bash
# 1. Set credentials
export SCPN_IBM_TOKEN="<token>"
export SCPN_IBM_INSTANCE="<instance>"

# 2. Generate source-backed parameter files
python3 scripts/frontier_campaign_2026/generate_params.py --output-dir params

# Optional smoke-test cache only; not publication-safe
python3 scripts/frontier_campaign_2026/generate_params.py \
  --output-dir params \
  --allow-synthetic \
  --seed 42

# 3. Run campaign (batch 5 settings: coupling_scale=2.0, DD, ZNE)
python3 scripts/frontier_campaign_2026/run_credible_tests.py

# 4. Retrieve results
python3 scripts/retrieve_all_jobs.py
```

Independent job verification:
```python
import os
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(
    channel="ibm_cloud",
    token=os.environ["SCPN_IBM_TOKEN"],
    instance=os.environ["SCPN_IBM_INSTANCE"],
)
job = service.job("<job_id_from_result_json>")
print(job.result())
```

---

## 9. Known Limitations

1. **Shot cap:** `min(shots, 4000)` — higher shot requests are silently capped.
2. **N=160 skipped:** IBM Heron r2 has 156 qubits; T1 N=160 point is always skipped.
3. **Phi saturation:** `IntegratedInformationPhi` returns ~0.9997–1.0000 for all runs (near-uniform bitstring distribution over 12–20 qubits at 4000 shots). Not a meaningful IIT measurement.
4. **T2 skipped:** `scpneurocore.bridge.load_live_stream` not implemented; live SCNeuroCore loop test not run.
5. **ZNE job_id:** When ZNE succeeds, `job_id` is set to `"zne_mitigated"` — individual scale-factor job IDs are managed internally by mitiq and not exposed in result JSON.
6. **ZNE cost:** Each ZNE run submits 3 circuits (scale=1,2,3) plus 1 final run = 4× IBM job cost vs unmitigated. T4 ZNE = 80 × 4 = 320 IBM jobs.
7. **Batch 1 baseline:** Run without DD or ZNE; inflated sync_order (0.14) is dominated by noise artefacts, not physical synchronisation.
