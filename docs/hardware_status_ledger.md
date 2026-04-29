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

## Status Snapshot — 2026-04-29

| Area | Current public status | Canonical source |
|---|---|---|
| Package line | Version `0.9.6`, Python `>=3.10`, Qiskit `>=2.2,<3.0`. | `pyproject.toml`, `CHANGELOG.md` |
| Generic compiler entry point | `scpn_quantum_control.kuramoto_core` validates arbitrary `K_nm`/`omega` problems and compiles Hamiltonians, dense matrices, Trotter circuits, and order-parameter measurements. | `docs/kuramoto_core_facade.md` |
| Core-package licence boundary | Possible future lightweight core split is documented, but no permissive relicensing has occurred. | `docs/core_package_boundary.md` |
| Baseline hardware campaign | `ibm_fez` Heron r2 baseline campaign supports Bell, QKD, ZNE, VQE, Trotter, and 16-qubit UPDE result figures. | `results/ibm_hardware_2026-03-28/`, `results/HARDWARE_RESULTS.md` |
| DLA parity campaign | `ibm_kingston` Heron r2 Phase 1 campaign supports the DLA parity asymmetry result. | `data/phase1_dla_parity/`, `docs/dla_parity.md` |
| Simulator claims | BKT, OTOC, Floquet, MBL, FIM, and classical comparison material remain simulator or classical-baseline claims unless a hardware artifact is named. | `results/SIMULATOR_RESULTS.md`, `results/classical_baselines_2026-03-30.json` |
| Pending frontier output | Later frontier and multi-QPU batches are not promoted to public claims until raw artifacts, retrieval manifests, and analysis scripts are reviewed and committed. | This ledger |

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
| Baseline IBM roadmap experiments | `ibm_fez` Heron r2 | 2026-03 | Hardware, unmitigated and mitigated sub-runs | `results/ibm_hardware_2026-03-28/`, `results/march_2026/`, `results/HARDWARE_RESULTS.md` | Bell, QKD, VQE, ZNE, Trotter, and UPDE baseline figures. |
| DLA parity Phase 1 | `ibm_kingston` Heron r2 | 2026-04 | Hardware, unmitigated campaign with simulator comparator | `data/phase1_dla_parity/`, `paper/phase1_dla_parity_short_paper.md`, `docs/dla_parity.md` | DLA parity asymmetry paper and result figures. |
| Simulator and classical baselines | local CPU/GPU where noted | 2026-03 onward | Simulator | `results/*_2026-03-*.json`, `results/SIMULATOR_RESULTS.md`, `results/classical_baselines_2026-03-30.json` | BKT, OTOC, Floquet, MBL, FIM, and classical comparison material. |

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
- Exact repository-wide test counts belong in CI summaries or release notes.
  Public overview pages should describe the CI-gated suite and coverage target
  rather than carrying static counts that drift between commits.

## Roadmap State — 2026-04-29

| Queue | State | Next gate |
|---|---|---|
| High-impact execution TODO | Complete locally. Dependency hygiene, core facade, documentation ergonomics, baselines, maintenance, and frontier-track scaffolding are checked off. | Commit/push current Figure 17 and documentation batch, then let CI verify. |
| Scientific gaps | Partially closed. K_nm physical validation still needs measured-system coupling magnitudes; broad quantum advantage remains open; `p_h1 = 0.72` is an explicit open empirical/theoretical parameter. | New measured-system data or a separate derivation/measurement campaign. |
| Hardware experiments | March/April promoted campaigns complete. Phase 2 DLA parity expansion is ready but blocked on promo/credits. | IBM credit/promo availability plus preregistered run manifest. |
| Strategic roadmap | All 53 post-v1.0 differentiation tracks remain deferred / CEO-gated. | Activate one track explicitly before implementation. |
