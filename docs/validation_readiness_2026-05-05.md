# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Validation readiness gate, 2026-05-05

# Validation Readiness Gate — 2026-05-05

This gate records what is ready for new validation runs and what remains
blocked. It exists to prevent aggregate-only, placeholder, queued-job, or
fallback-count artefacts from being promoted as hardware validation.

## Scope

The current promoted hardware evidence remains limited to:

- April 2026 Phase 1 DLA parity on `ibm_kingston`, backed by raw counts, job
  identifiers, integrity checks, and the committed reproduction harness.
- Legacy March 2026 `ibm_fez` rows only where a committed artefact is named;
  these are not broad-advantage or frontier validation.

No new IBM validation run may be promoted until a preregistered manifest exists
before submission and the count-to-statistic reproduction path is committed.

## Local readiness checks run on 2026-05-05

### Phase 1 DLA parity raw-count reproducer

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/run_dla_parity_suite.py --verify-integrity
```

Result:

| Check | Value |
|---|---:|
| Dataset circuits | `342` |
| Sub-phase runs | `4` |
| Backend | `ibm_kingston` |
| Fisher statistic | `chi2=123.4001`, `df=16` |
| Significant depths at 0.05 | `7 / 8` |
| Peak asymmetry | `+17.48%` at depth `6` |
| Mean asymmetry | `+9.25%` |
| Published claims checked | `57`, all within tolerance |
| Classical reference | `qutip`, `max_abs_leakage=0.000e+00` |
| Classical zero-leakage invariant | true within `1e-10` |

Decision: the Phase 1 DLA parity dataset is reproducible from committed raw
counts and remains the promoted raw-count hardware dataset.

### Broad-advantage guardrail smoke check

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/anulum/.local/bin/python scripts/bench_quantum_advantage_classical_matrix.py --sizes 4,6 --output /tmp/scpn_quantum_advantage_guardrail_smoke.json
```

Result:

| n | dim | classical ODE ms | exact dense ms | exact sparse ms | Rust ODE ms | GPU |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 16 | `11.159962` | `0.533185` | `1.245445` | `0.008856` | unavailable |
| 6 | 64 | `41.192475` | `47.946798` | `5.254019` | `0.060972` | unavailable |

Decision: the matrix path executes on the current commit without overwriting
committed artefacts. This is a smoke check only; the committed May 3 matrix
remains the provenance artefact for the broader `n=4,6,8,10,12` evidence gate.

## Promotion rules for the next IBM validation run

A new IBM result is not promotable unless all items below are satisfied:

| Gate | Requirement |
|---|---|
| Preregistered manifest | Observable, backend, circuit family, shots, depths, reps, mitigation state, abort criteria, and statistical tests are committed before submission. |
| Submission record | Every submitted circuit batch records backend, job identifier, transpilation settings, shots, and timestamp. |
| Retrieval record | Raw counts are retrieved from IBM and stored with a manifest that binds each count set to a job identifier and circuit identifier. |
| Reproducer | A committed script recomputes every promoted statistic from raw counts and exits non-zero on tolerance or invariant failure. |
| Ledger row | `docs/hardware_status_ledger.md` names the evidence class and artefact paths before any summary page cites the result. |
| No fallbacks | All-zero counts, queued placeholders, aggregate-only JSON, and submit-now-retrieve-later files are quarantined, not evidence. |

## Current readiness decision

The project is prepared for local validation and preregistration work. It is not
prepared to submit or promote a new IBM hardware validation run until the Phase
2 manifest is committed and the IBM credit or promo window is live.

## Phase 2 preregistration update

The Phase 2 preregistered manifest is now recorded in
`docs/ibm_phase2_preregistered_manifest_2026-05-05.md`, with the machine-readable
inventory in `results/ibm_phase2_preregistration_2026-05-05.json`.

The first live command is narrowed to blocks A+G only so the run spends the
minimum QPU time needed for high-statistics `n=4` replication plus readout
baseline. Blocks B-F are deferred until the primary raw counts are retrieved
and reviewed.

This closes the preregistration document gate only. IBM submission remains
blocked until the credit window is live and the exact reduced
`--confirm-promo-active --skip B C D E F` command is approved immediately
before execution.

## 2026-05-05 live-attempt update

The first reduced A+G live attempt was cancelled after the hardware
transpilation path produced deeper circuits than the reduced dry-run manifest.
IBM job `ibm-run-ca8b9612732b84dc` is recorded as `CANCELLED`, with IBM metadata
reporting `0` quantum seconds and `0` usage seconds.

This cancelled job is quarantined execution evidence only. It supplies no raw
counts and closes no validation claim. The follow-up script now performs a
hardware-backend transpilation precheck before submission and requires the live
max depth to remain within the manifest budget.

## 2026-05-05 completed run update

The reduced A+G Phase 2 run completed on `ibm_kingston`:

- Main job: `ibm-run-7da8644af35021fb`, 600 circuits, 4096 shots, IBM-reported
  quantum seconds `660`.
- Readout job: `ibm-run-6f9990bba1d90a12`, 12 circuits, 8192 shots, IBM-reported
  quantum seconds `27`.
- Raw-count dataset: `data/phase2_dla_parity/phase2_reduced_ag_2026-05-05T121357Z.json`.
- Reproducer: `scripts/analyse_phase2_dla_parity.py --verify-integrity`.

The verified reproducer reports Fisher chi2 `140.671952`, Fisher p
`3.773718e-20`, and 6/10 depths significant at Welch p < 0.05. This promotes
only the reduced `n=4` DLA parity replication plus readout baseline. It does
not promote `n=6-12` scaling, GUESS calibration, broad quantum advantage, or
frontier claims.

## 2026-05-05 B-C scaling continuation update

The preregistered B-C continuation completed on `ibm_kingston`:

- Job: `ibm-run-1f46ebd0da8912ff`, 280 circuits, 4096 shots, IBM-reported quantum
  seconds `305`.
- Raw-count dataset: `data/phase2_scaling_bc/phase2_scaling_bc_2026-05-05T124722Z.json`.
- Reproducer: `scripts/analyse_phase2_scaling_bc.py`.

The verified reproducer reports mixed scaling evidence:

- `n=6`: Fisher chi2 `46.531552`, p `1.883218e-07`, 2/4 significant depths,
  with negative significant asymmetry at depths 8 and 20.
- `n=8`: Fisher chi2 `29.420107`, p `2.675193e-04`, 3/4 significant depths,
  with positive middle-depth sign.

This promotes only mixed `n=6,8` scaling evidence. It falsifies a simple
monotone scaling story and does not promote D-E larger scaling, GUESS
calibration, broad quantum advantage, frontier, multi-QPU, or live-loop claims.
