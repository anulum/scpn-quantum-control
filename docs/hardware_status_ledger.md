# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Status Ledger

# Hardware Status Ledger

Result-pack integrity layer: promoted IBM raw-count datasets are now indexed in
[`hardware_result_packs.md`](hardware_result_packs.md), with the canonical
manifest at `data/hardware_result_packs/manifest.json` and an offline verifier
at `scripts/verify_hardware_result_packs.py`. The result-pack layer preserves
artefact integrity and claim boundaries; it does not broaden any claim in this
ledger.

This page is the public index for hardware and simulator evidence. Summary pages
may quote selected results, but every quoted campaign should point back here or
to a campaign-specific artefact with backend, date, mitigation state, job count,
and raw-data path.

## Status Snapshot — 2026-05-06

| Area | Current public status | Canonical source |
|---|---|---|
| Package line | Version `0.9.10`, Python `>=3.10`, Qiskit `>=2.2,<3.0`. | `pyproject.toml`, `CHANGELOG.md` |
| Generic compiler entry point | `scpn_quantum_control.kuramoto_core` validates arbitrary `K_nm`/`omega` problems and compiles Hamiltonians, dense matrices, Trotter circuits, and order-parameter measurements. | `docs/kuramoto_core_facade.md` |
| Core-package licence boundary | Possible future lightweight core split is documented, but no permissive relicensing has occurred. | `docs/core_package_boundary.md` |
| Baseline hardware campaign | `ibm_fez` Heron r2 baseline artefacts are retained as legacy QPU evidence. Quote only values that name a committed raw artefact or retrieval file; do not use this campaign as proof of broad advantage. | `results/ibm_hardware_2026-03-28/`, `results/march_2026/`, `results/IBM_HARDWARE_COMPLETE_AUDIT_2026-03-30.md` |
| DLA parity campaigns | `ibm_kingston` Heron r2 Phase 1, Phase 2 A+G, Phase 2 B-C, and popcount-control campaigns are promoted hardware datasets: raw counts, public run labels, integrity checks, and reproduction harnesses are committed. Raw IBM job identifiers are retained only in the private mapping. | `data/phase1_dla_parity/`, `data/phase2_dla_parity/`, `data/phase2_scaling_bc/`, `data/phase2_popcount_control/`, `docs/publication/publication_phase2_package_2026-05-05.md`, `scripts/analyse_phase2_dla_parity.py`, `scripts/analyse_phase2_scaling_bc.py`, `scripts/analyse_phase2_popcount_control.py` |
| SCPN/FIM hardware campaign | `ibm_kingston` Heron r2 pilot and repeated follow-up are promoted as a negative/falsification result for the simple digital `lambda=4` hardware-protection hypothesis on the tested circuit family. | `data/scpn_fim_hamiltonian/`, `docs/campaigns/scpn_fim_claim_boundary_2026-05-05.md`, `scripts/analyse_fim_ibm_pilot.py`, `scripts/analyse_fim_ibm_repeated_followup.py`, `scripts/analyse_fim_readout_matrix_mitigation.py` |
| Simulator claims | BKT, OTOC, Floquet, MBL, FIM, and classical comparison material remain simulator or classical-baseline claims unless a hardware artefact is named. | `results/SIMULATOR_RESULTS.md`, `results/classical_baselines_2026-03-30.json` |
| Quarantined / unpromoted IBM output | Any frontier, V2, queued-job, placeholder, or aggregate-only IBM output is not promoted until it has raw counts, private retrieval map, analysis code, and an explicit ledger row. | `results/ibm_hardware_v2_2026-03-29/`, `results/ibm_runs/jobs.json`, `docs/internal/chat_exports/2026-04-25_gemini_chat_export_full.md` |

README, `docs/index.md`, and `docs/results.md` should treat this dated snapshot
as the source of truth for public status wording. If a campaign value changes,
update this table first, then refresh the summary pages.

## Claim Classes

| Class | Meaning | Required evidence |
|---|---|---|
| Theory | Analytic statement or theorem. | Derivation, assumptions, and testable prediction. |
| Simulator | Classical, tensor-network, statevector, or noisy simulator result. | Script/notebook path, seed policy, package versions, and output artefact. |
| Hardware, unmitigated | Raw QPU measurement without post-run mitigation beyond standard transpilation. | Backend, public run labels, shots, circuit family, and raw counts. |
| Hardware, mitigated | QPU result after mitigation such as ZNE, symmetry checks, or dynamical decoupling. | Raw counts, mitigation parameters, and unmitigated comparator. |
| Falsification or noise-limited | Negative or bounded result that constrains a claim. | Same evidence as the corresponding simulator or hardware class. |

## Public Hardware Campaigns

| Campaign | Backend | Date | Evidence class | Public artefacts | Current use |
|---|---|---:|---|---|---|
| Baseline IBM roadmap experiments | `ibm_fez` Heron r2 | 2026-03 | Legacy hardware artefacts; quote artefact-backed rows only | `results/ibm_hardware_2026-03-28/`, `results/march_2026/`, `results/IBM_HARDWARE_COMPLETE_AUDIT_2026-03-30.md` | Baseline Bell, QKD, VQE, ZNE, Trotter, and UPDE observations, with no broad-advantage promotion. |
| DLA parity Phase 1 | `ibm_kingston` Heron r2 | 2026-04 | Promoted hardware dataset with raw-count reproducer | `data/phase1_dla_parity/`, `paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity_short_paper.md`, `docs/dla_parity.md`, `scripts/run_dla_parity_suite.py` | DLA parity asymmetry paper and result figures. |
| DLA parity Phase 2 reduced A+G | `ibm_kingston` Heron r2 | 2026-05 | Promoted hardware replication dataset with raw-count reproducer | `data/phase2_dla_parity/`, `scripts/analyse_phase2_dla_parity.py`, `results/ibm_phase2_preregistration_2026-05-05.json` | High-statistics `n=4` DLA parity replication plus readout baseline only; no `n=6-12`, GUESS, or broad-advantage claim. |
| DLA parity Phase 2 B-C scaling | `ibm_kingston` Heron r2 | 2026-05 | Promoted mixed scaling dataset with raw-count reproducer | `data/phase2_scaling_bc/`, `scripts/analyse_phase2_scaling_bc.py`, `docs/campaigns/ibm_phase2_scaling_bc_manifest_2026-05-05.md` | Mixed `n=6,8` scaling evidence: `n=8` positive middle-depth sign, `n=6` negative at significant depths; no monotone scaling claim. |
| DLA parity Phase 2 popcount control | `ibm_kingston` Heron r2 | 2026-05 | Promoted hardware control dataset with raw-count reproducer | `data/phase2_popcount_control/`, `scripts/analyse_phase2_popcount_control.py`, `docs/publication/publication_phase2_package_2026-05-05.md` | Excitation count and state choice materially contribute; no DLA-parity-only causal claim. |
| SCPN/FIM pilot and repeated follow-up | `ibm_kingston` Heron r2 | 2026-05 | Promoted negative/falsification hardware result | `data/scpn_fim_hamiltonian/`, `docs/campaigns/scpn_fim_claim_boundary_2026-05-05.md`, `scripts/analyse_fim_ibm_repeated_followup.py` | Digital `lambda=4` implementation increases leakage/decreases retention for the tested circuit family; no coherence-protection claim. |
| Simulator and classical baselines | local CPU/GPU where noted | 2026-03 onward | Simulator | `results/*_2026-03-*.json`, `results/SIMULATOR_RESULTS.md`, `results/classical_baselines_2026-03-30.json` | BKT, OTOC, Floquet, MBL, FIM, and classical comparison material. |

## Quarantined / Unpromoted IBM Artifacts

The April 2026 internal incident trail records placeholder and fake-count
handling in frontier IBM workflows. Therefore the files below must not be used
as public proof until they are independently re-retrieved or reproduced from
raw IBM counts and promoted here.

| Artifact family | Current status | Reason |
|---|---|---|
| `results/ibm_hardware_v2_2026-03-29/` | Unpromoted aggregate-only evidence. | Contains public run labels and aggregate metrics, but lacks the raw-count private retrieval trail and reproduction harness required for a promoted claim. |
| `results/ibm_hardware_2026-03-29/dla_parity_*.json` | Superseded / unpromoted. | The audit identifies a circuit-depth artefact in this March DLA parity attempt; use the April `data/phase1_dla_parity/` dataset instead. |
| `results/ibm_runs/jobs.json` and frontier queue outputs | Quarantined. | Internal logs document queued-job placeholders and fake all-zero fallback counts in related workflows. |
| Any "400 jobs", large-N frontier, multi-QPU, or live-loop claim | Not promoted. | Requires raw counts, private retrieval map, analysis script, and a new ledger row before citation. |

## Evidence Rules

- Do not quote a numerical result unless the source artefact is named.
- Do not mix simulator and hardware numbers in one table without the evidence
  class column.
- Do not promote internal or pending campaign output into public claims until
  its raw counts, private retrieval map, and analysis script are committed.
- If a later mitigation pass changes a conclusion, keep the older value as a
  historical row and add the newer value with its mitigation state.
- Negative results stay in the ledger. They are evidence, not cleanup debt.

## Current Gaps

- `docs/results.md` remains a public gallery and technical summary. This ledger
  is the canonical status index; further result-page edits should keep detailed
  provenance here rather than duplicating claim-state decisions.
- Later frontier batches must be added only after their raw artefacts,
  private retrieval maps, and analysis scripts are reviewed and committed.
- Do not cite aggregate-only IBM JSON, queued-job JSON, or placeholder-derived
  results as hardware validation.
- Exact repository-wide test counts belong in CI summaries or release notes.
  Public overview pages should describe the CI-gated suite and coverage target
  rather than carrying static counts that drift between commits.

## Roadmap State — 2026-04-30

| Queue | State | Next gate |
|---|---|---|
| High-impact execution TODO | Complete locally. Dependency hygiene, core facade, documentation ergonomics, baselines, maintenance, frontier-track scaffolding, and CI timing-gate stabilization are checked off. | Keep CI green on `main`; add only scoped follow-up work. |
| Scientific gaps | Partially closed. The EEG PLV K_nm validation artefacts now cover the full 109-subject PhysioNet EEGMMIDB baseline eyes-open and eyes-closed cohorts, with a derived condition comparison. The first physical-unit measured-system control, IEEE 5-bus, is committed and does not close K_nm physical validation; broad quantum advantage remains open; `p_h1 = 0.72` is an explicit open empirical/theoretical parameter. | Additional physical-unit measured coupling candidates with null models; provenance-rich advantage benchmark tables; TCBO or first-principles p_h1 reproduction. |
| Hardware experiments | March/April/May evidence is narrowed to legacy `ibm_fez` baseline artefacts, promoted raw-count `ibm_kingston` DLA Phase 1/2 datasets, and the promoted SCPN/FIM negative result. | Further QPU work needs a preregistered manifest, depth/shot gates, QPU-time estimate, and explicit approval; no frontier promotion without raw-count review. |
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
- Gap B close-out review status: **DONE 2026-05-05** — the validation readiness gate records the Phase 1 raw-count reproducer pass, a current-commit classical/Rust matrix smoke check, and the promotion rules for future broad-advantage or hardware claims.
- Phase 2 preregistration status: **DONE 2026-05-05** — `docs/campaigns/ibm_phase2_preregistered_manifest_2026-05-05.md` and `results/ibm_phase2_preregistration_2026-05-05.json` record the QPU-minimised A+G first live command, dry-run circuit inventory, abort criteria, evidence path, and promotion gates.
- 2026-05-05 live attempt status: **ABORTED / CANCELLED** — `ibm_kingston` job `ibm-run-ca8b9612732b84dc` was cancelled after live hardware transpilation exceeded the reduced dry-run depth budget; IBM metadata reported `0` quantum seconds and `0` usage seconds. This job is quarantined and not evidence.
- 2026-05-05 reduced A+G hardware run status: **DONE / PROMOTED** — `ibm_kingston` jobs `ibm-run-7da8644af35021fb` and `ibm-run-6f9990bba1d90a12` completed with 612 raw-count circuits. The committed reproducer reports Fisher chi2 `140.671952`, Fisher p `3.773718e-20`, and 6/10 significant depths at `p < 0.05`.
- 2026-05-05 B-C scaling run status: **DONE / PROMOTED AS MIXED** — `ibm_kingston` job `ibm-run-1f46ebd0da8912ff` completed with 280 raw-count circuits and IBM-reported usage `305` quantum seconds. The committed reproducer reports `n=6` Fisher p `1.883218e-07` with negative significant depths and `n=8` Fisher p `2.675193e-04` with positive middle-depth sign.
- Phase 2 publication package status: **DONE / PROMOTED** —
  `docs/publication/publication_phase2_package_2026-05-05.md` records the promoted
  A+G, B-C, and popcount-control artefacts plus conservative claim boundaries.

## EEG Condition Comparison — 2026-04-30

The K_nm physical-validation data now includes a matched 109-subject PhysioNet
EEGMMIDB baseline eyes-closed cohort (`S001R02` through `S109R02`) and a derived
eyes-closed-minus-eyes-open comparison artefact.

| Metric | Value | Source |
|---|---:|---|
| Eyes-open mean edge PLV | `0.545586` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Eyes-closed mean edge PLV | `0.600050` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Mean closed-minus-open delta | `0.054463` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Median closed-minus-open delta | `0.061094` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Mean absolute edge delta | `0.066249` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Largest absolute edge delta | `0.135901` on edge `(2, 3)` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| Pearson r across edge medians | `0.963961` | `data/knm_physical_validation/baseline_open_closed_comparison.json` |
| EEG PLV versus eight-layer K_nm Spearman | `0.767804` | `data/knm_physical_validation/eeg_alpha_plv_knm_comparison.json` |
| EEG PLV versus eight-layer K_nm Pearson | `0.715303` | `data/knm_physical_validation/eeg_alpha_plv_knm_comparison.json` |

This closes the condition-control EEG PLV comparison gate. It does not close
the physical-unit measured-coupling gate: PLV is an association observable, and
the measured-system promotion audit now blocks `phase_locking_value` units from
promotion as calibrated K_nm magnitudes. Public claims must continue to
describe this as condition-specific alpha-band EEG coupling evidence rather
than measured physical K_nm magnitudes.

## Measured-System Control — 2026-04-30

The first physical-unit measured-system artefact is the IEEE 5-bus power-grid
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

## Measured-System Control Expansion — 2026-05-18

The physical-unit candidate set now also includes the IEEE 14-bus public
benchmark as a voltage-weighted branch-admittance control. The artefact records
the public branch reactances, solved voltage magnitudes, derived
`K_ij = V_i V_j / X_ij` matrix, and propagated input-rounding uncertainty for
all 91 pairwise bus edges.

| Metric | Value | Source |
|---|---:|---|
| Matched pairwise bus edges | `91` | `data/knm_physical_validation/measured_couplings_power_grid_ieee14bus.json` |
| Non-zero public branch edges | `20` | `data/knm_physical_validation/measured_couplings_power_grid_ieee14bus.json` |
| Spearman topology correlation vs fourteen-layer K_nm | `0.406186` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Pearson topology correlation vs fourteen-layer K_nm | `0.346893` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Direct RMSE | `4.248824` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Direct relative RMSE versus mean absolute measured coupling | `2.571947` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Best scale through origin | `12.545508` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Scaled RMSE | `3.759167` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Maximum direct absolute error | `24.528498` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Weighted adjacency spectrum Pearson | `0.832010` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Weighted adjacency spectrum RMSE | `15.106532` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Weighted Laplacian spectrum Pearson | `0.774822` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Weighted Laplacian spectrum RMSE | `27.322728` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Critical-response relative difference | `0.942248` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Node-label null mode | seeded sampled permutations (`4096`) | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Node-label null Spearman empirical p | `0.000244` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |
| Edge-value null Spearman empirical p | `0.000244` | `data/knm_physical_validation/power_grid_ieee14bus_knm_comparison.json` |

Decision: this expands the measured-system candidate inventory, but it does not
close physical K_nm validation. Public case14 supplies branch reactances and
voltage magnitudes, not measured per-bus inertia constants for every load bus,
so the artefact is a non-promotional control candidate unless a future gate
adds the missing dynamic-system measurements and passes the null-model,
uncertainty, magnitude, and critical-response checks.
