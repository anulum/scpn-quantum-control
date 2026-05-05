# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Status Ledger

# Hardware Status Ledger

This page is the public index for hardware and simulator evidence. Summary pages
may quote selected results, but every quoted campaign should point back here or
to a campaign-specific artifact with backend, date, mitigation state, job count,
and raw-data path.

## Status Snapshot — 2026-05-05

| Area | Current public status | Canonical source |
|---|---|---|
| Package line | Version `0.9.6`, Python `>=3.10`, Qiskit `>=2.2,<3.0`. | `pyproject.toml`, `CHANGELOG.md` |
| Generic compiler entry point | `scpn_quantum_control.kuramoto_core` validates arbitrary `K_nm`/`omega` problems and compiles Hamiltonians, dense matrices, Trotter circuits, and order-parameter measurements. | `docs/kuramoto_core_facade.md` |
| Core-package licence boundary | Possible future lightweight core split is documented, but no permissive relicensing has occurred. | `docs/core_package_boundary.md` |
| Baseline hardware campaign | `ibm_fez` Heron r2 baseline artifacts are retained as legacy QPU evidence. Quote only values that name a committed raw artifact or retrieval file; do not use this campaign as proof of broad advantage. | `results/ibm_hardware_2026-03-28/`, `results/march_2026/`, `results/IBM_HARDWARE_COMPLETE_AUDIT_2026-03-30.md` |
| DLA parity campaign | `ibm_kingston` Heron r2 Phase 1 campaign is the strongest promoted hardware dataset: raw counts, job IDs, integrity checks, and a reproduction harness are committed. | `data/phase1_dla_parity/`, `docs/dla_parity.md`, `scripts/run_dla_parity_suite.py` |
| Simulator claims | BKT, OTOC, Floquet, MBL, FIM, and classical comparison material remain simulator or classical-baseline claims unless a hardware artifact is named. | `results/SIMULATOR_RESULTS.md`, `results/classical_baselines_2026-03-30.json` |
| Quarantined / unpromoted IBM output | Any frontier, V2, queued-job, placeholder, or aggregate-only IBM output is not promoted until it has raw counts, retrieval manifest, analysis code, and an explicit ledger row. | `results/ibm_hardware_v2_2026-03-29/`, `results/ibm_runs/jobs.json`, `docs/internal/gemini_chat_export_full_2026-04-25.md` |

README, `docs/index.md`, and `docs/results.md` should treat this dated snapshot
as the source of truth for public status wording. If a campaign value changes,
update this table first, then refresh the summary pages.

## Claim Classes

| Class | Meaning | Required evidence |
|---|---|---|
| Theory | Analytic statement or theorem. | Derivation, assumptions, and testable prediction. |
| Simulator | Classical, tensor-network, statevector, or noisy simulator result. | Script/notebook path, seed policy, package versions, and output artifact. |
| Hardware, unmitigated | Raw QPU measurement without post-run mitigation beyond standard transpilation. | Backend, job IDs, shots, circuit family, and raw counts. |
| Hardware, mitigated | QPU result after mitigation such as ZNE, symmetry checks, or dynamical decoupling. | Raw counts, mitigation parameters, and unmitigated comparator. |
| Falsification or noise-limited | Negative or bounded result that constrains a claim. | Same evidence as the corresponding simulator or hardware class. |

## Public Hardware Campaigns

| Campaign | Backend | Date | Evidence class | Public artifacts | Current use |
|---|---|---:|---|---|---|
| Baseline IBM roadmap experiments | `ibm_fez` Heron r2 | 2026-03 | Legacy hardware artifacts; quote artifact-backed rows only | `results/ibm_hardware_2026-03-28/`, `results/march_2026/`, `results/IBM_HARDWARE_COMPLETE_AUDIT_2026-03-30.md` | Baseline Bell, QKD, VQE, ZNE, Trotter, and UPDE observations, with no broad-advantage promotion. |
| DLA parity Phase 1 | `ibm_kingston` Heron r2 | 2026-04 | Promoted hardware dataset with raw-count reproducer | `data/phase1_dla_parity/`, `paper/phase1_dla_parity_short_paper.md`, `docs/dla_parity.md`, `scripts/run_dla_parity_suite.py` | DLA parity asymmetry paper and result figures. |
| Simulator and classical baselines | local CPU/GPU where noted | 2026-03 onward | Simulator | `results/*_2026-03-*.json`, `results/SIMULATOR_RESULTS.md`, `results/classical_baselines_2026-03-30.json` | BKT, OTOC, Floquet, MBL, FIM, and classical comparison material. |

## Quarantined / Unpromoted IBM Artifacts

The April 2026 internal incident trail records placeholder and fake-count
handling in frontier IBM workflows. Therefore the files below must not be used
as public proof until they are independently re-retrieved or reproduced from
raw IBM counts and promoted here.

| Artifact family | Current status | Reason |
|---|---|---|
| `results/ibm_hardware_v2_2026-03-29/` | Unpromoted aggregate-only evidence. | Contains job IDs and aggregate metrics, but lacks the raw-count retrieval trail and reproduction harness required for a promoted claim. |
| `results/ibm_hardware_2026-03-29/dla_parity_*.json` | Superseded / unpromoted. | The audit identifies a circuit-depth artifact in this March DLA parity attempt; use the April `data/phase1_dla_parity/` dataset instead. |
| `results/ibm_runs/jobs.json` and frontier queue outputs | Quarantined. | Internal logs document queued-job placeholders and fake all-zero fallback counts in related workflows. |
| Any "400 jobs", large-N frontier, multi-QPU, or live-loop claim | Not promoted. | Requires raw counts, retrieval manifest, analysis script, and a new ledger row before citation. |

## Evidence Rules

- Do not quote a numerical result unless the source artifact is named.
- Do not mix simulator and hardware numbers in one table without the evidence
  class column.
- Do not promote internal or pending campaign output into public claims until
  its raw counts, retrieval manifest, and analysis script are committed.
- If a later mitigation pass changes a conclusion, keep the older value as a
  historical row and add the newer value with its mitigation state.
- Negative results stay in the ledger. They are evidence, not cleanup debt.

## Current Gaps

- `docs/results.md` remains a public gallery and technical summary. This ledger
  is the canonical status index; further result-page edits should keep detailed
  provenance here rather than duplicating claim-state decisions.
- Later frontier batches must be added only after their raw artifacts,
  retrieval manifests, and analysis scripts are reviewed and committed.
- Do not cite aggregate-only IBM JSON, queued-job JSON, or placeholder-derived
  results as hardware validation.
- Exact repository-wide test counts belong in CI summaries or release notes.
  Public overview pages should describe the CI-gated suite and coverage target
  rather than carrying static counts that drift between commits.

## Roadmap State — 2026-04-30

| Queue | State | Next gate |
|---|---|---|
| High-impact execution TODO | Complete locally. Dependency hygiene, core facade, documentation ergonomics, baselines, maintenance, frontier-track scaffolding, and CI timing-gate stabilization are checked off. | Keep CI green on `main`; add only scoped follow-up work. |
| Scientific gaps | Partially closed. The EEG PLV K_nm validation artifacts now cover the full 109-subject PhysioNet EEGMMIDB baseline eyes-open and eyes-closed cohorts, with a derived condition comparison. The first physical-unit measured-system control, IEEE 5-bus, is committed and does not close K_nm physical validation; broad quantum advantage remains open; `p_h1 = 0.72` is an explicit open empirical/theoretical parameter. | Additional physical-unit measured coupling candidates with null models; provenance-rich advantage benchmark tables; TCBO or first-principles p_h1 reproduction. |
| Hardware experiments | March/April evidence is narrowed to the legacy `ibm_fez` baseline artifacts and the promoted raw-count `ibm_kingston` Phase 1 DLA parity dataset. Phase 2 DLA parity expansion is ready but blocked on promo/credits. | IBM credit/promo availability plus preregistered run manifest; no frontier promotion without raw-count review. |
| Strategic roadmap | All 53 post-v1.0 differentiation tracks remain deferred / CEO-gated. | Activate one track explicitly before implementation. |

### 2026-05-03 Scientific-Gap Roadmap Progress

- Gap A item 1 status: **DONE** — added a measured non-EEG physical-unit
  coupling candidate (IEEE 5-bus power grid) with uncertainty propagation and
  null-model comparison in the existing candidate control pipeline.
- Gap A item 2 status: **DONE** — applied the measured-system comparison scan to
  all current candidate systems in
  `data/public_application_benchmarks/{eeg_alpha_plv_8ch,friston_fep_6node,ieee5bus_power_grid,iter_mhd_8mode}.json`
  with topology/magnitude/decision outputs and no closed candidate.
- Gap A item 3 status: **DONE** — claim scope is constrained to condition-specific
  EEG PLV evidence for any closure that is not backed by a locked physical-unit
  magnitude match plus null-model pass.
- Gap B item 1 status: **DONE** — scaling benchmark rows now carry backend,
  machine, command, dependency, and commit provenance in the committed
  `run_scaling_benchmark` and crossover-point loaders.
- Gap B item 2 status: **DONE** — separate exact-simulation crossover from
  observable-level broad-advantage claims in the public benchmark wording.
- Gap B item 3 status: **DONE** — ran the classical/Rust/GPU benchmark matrix
  for the quantum-advantage comparison and saved it as
  `results/classical_rust_gpu_matrix_2026-05-03.json`.
- Next pending item: **Gap B close-out review** — apply the matrix evidence to
  the next broad-advantage evidence gate and any remaining observable-level
  claim wording.

## EEG Condition Comparison — 2026-04-30

The K_nm physical-validation data now includes a matched 109-subject PhysioNet
EEGMMIDB baseline eyes-closed cohort (`S001R02` through `S109R02`) and a derived
eyes-closed-minus-eyes-open comparison artifact.

| Metric | Value | Source |
|---|---:|---|
| Eyes-open mean edge PLV | `0.545586` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Eyes-closed mean edge PLV | `0.600050` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Mean closed-minus-open delta | `0.054463` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Median closed-minus-open delta | `0.061094` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Mean absolute edge delta | `0.066249` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Largest absolute edge delta | `0.135901` on edge `(2, 3)` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Pearson r across edge medians | `0.963961` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |

This closes the condition-control EEG PLV comparison gate. It does not close
the physical-unit measured-coupling gate: PLV remains dimensionless, and public
claims must continue to describe this as condition-specific alpha-band EEG
coupling evidence rather than measured physical K_nm magnitudes.

## Measured-System Control — 2026-04-30

The first physical-unit measured-system artifact is the IEEE 5-bus power-grid
swing-equation coupling matrix. It records raw public benchmark constants,
conversion units, and propagated input-rounding uncertainty.

| Metric | Value | Source |
|---|---:|---|
| Matched edges | `10` | `data/knm_physical_validation/measured_couplings_power_grid_ieee5bus.json` |
| Spearman topology correlation vs five-layer K_nm | `0.190394` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Pearson topology correlation vs five-layer K_nm | `0.226144` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Direct RMSE | `0.219843` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Direct relative RMSE versus mean absolute measured coupling | `261.335679` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Best scale through origin | `0.003925` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Scaled RMSE | `0.000894` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Maximum direct absolute error | `0.300747` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Weighted adjacency spectrum Pearson | `0.826329` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Weighted adjacency spectrum RMSE | `0.439359` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Weighted Laplacian spectrum Pearson | `0.829195` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Weighted Laplacian spectrum RMSE | `0.966608` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Kuramoto threshold-proxy ratio measured/canonical | `244.239277` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Node-label null Spearman empirical p | `0.363636` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Node-label null Pearson empirical p | `0.264463` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Node-label null RMSE empirical p | `0.264463` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Edge-value null Spearman empirical p | `0.291189` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Edge-value null Pearson empirical p | `0.254577` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |
| Edge-value null RMSE empirical p | `0.254577` | `data/knm_physical_validation/power_grid_ieee5bus_knm_comparison.json` |

Decision: this is a useful measured-system control and a negative result for
the exact-magnitude K_nm promotion gate. It also does not beat the node-label
or edge-value permutation null gates, and the spectral/critical-response
diagnostics do not rescue the match. Physical validation remains open until a
measured-system candidate with units, uncertainty, and preregistered null models
passes the promotion criteria.
