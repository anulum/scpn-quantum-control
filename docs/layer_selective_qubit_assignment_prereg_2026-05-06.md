<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Layer-Selective Qubit Assignment Preregistration -->

# Layer-Selective Qubit Assignment Preregistration

Date: 2026-05-06

This preregistration defines a no-outcome-data qubit-assignment protocol for
Kuramoto-XY and DLA-parity circuits. It does not submit IBM jobs, reserve
backend time, or authorise QPU spend.

## Scientific Question

Can assigning the strongest Hamiltonian interaction layers to the lowest-error
available physical qubit pairs reduce compiled depth, two-qubit gate burden, or
leakage variance relative to the provider default layout and generic SABRE
layout?

## Claim Boundary

Supported after successful execution and analysis:

- compiled-resource comparison between default, SABRE, and layer-selective
  assignment;
- hardware-count comparison for the same circuit family if an approved QPU run
  is later executed;
- evidence that coupling-aware layout selection does or does not reduce
  layout-driven variance for the sampled backend and calibration window.

Blocked even after a positive result:

- backend-general layout optimality;
- quantum advantage;
- universal noise reduction;
- causality from layout alone without state/readout controls;
- reuse of the same physical layout after calibration drift without a fresh
  readiness pass.

## Assignment Principle

Inputs collected before any outcome data:

- logical coupling matrix `K_ij`;
- circuit layer decomposition or ordered interaction pairs;
- backend coupling map;
- live calibration metadata where available: readout error, two-qubit error,
  gate duration, `T1`, and `T2`;
- transpiler seed list fixed before submission.

Logical pair priority:

```text
priority(i, j) = |K_ij| * layer_weight(i, j)
```

Physical pair cost:

```text
cost(q_a, q_b) =
    w_2q * two_qubit_error(q_a, q_b)
  + w_ro * mean_readout_error(q_a, q_b)
  + w_t1 * inverse_T1_penalty(q_a, q_b)
  + w_path * shortest_path_length(q_a, q_b)
```

Default weights:

| Weight | Value | Rationale |
|--------|-------|-----------|
| `w_2q` | `0.55` | two-qubit error is the dominant compiled-circuit risk |
| `w_ro` | `0.20` | parity and state-retention observables are readout-sensitive |
| `w_t1` | `0.15` | amplitude damping is a known excitation-count confound |
| `w_path` | `0.10` | routing distance predicts SWAP pressure |

The assignment is selected by minimizing the weighted logical-pair-to-physical
pair cost before outcome counts exist. Ties are broken by lower transpiled
two-qubit gate count, then lower depth, then fixed lexical ordering of physical
qubit indices.

## Comparator Layouts

Every promoted comparison must include:

- provider or transpiler default layout;
- SABRE layout with fixed seed list;
- layer-selective layout from the preregistered score above.

Optional additional comparators:

- best connected low-error window ignoring `K_ij`;
- randomized connected windows from the state/layout randomisation protocol.

## Offline Readiness Matrix

Default no-QPU readiness scope:

| Field | Value |
|-------|-------|
| `n` | `4` |
| Circuit families | DLA parity A+G, popcount controls, GUESS folded circuits |
| Depths | `6, 8, 10, 14` |
| Layout methods | default, SABRE, layer-selective |
| Transpiler seeds | `0, 1, 2, 3, 4` |
| Backend class | Heron-class live backend or saved calibration snapshot |

Readiness output must report per-family and pooled:

- depth summary;
- two-qubit gate count summary;
- SWAP count or routing overhead where available;
- selected physical qubits;
- selected high-priority logical edges and their physical realization;
- calibration timestamp and backend name.

## Hardware Follow-Up Scope

If offline readiness supports a hardware comparison and QPU approval is granted,
use the smallest block that can falsify the layout benefit:

| Field | Value |
|-------|-------|
| `n` | `4` |
| States | `0011`, `0001`, `0101`, `0010` |
| Depths | `6, 10, 14` |
| Layout methods | default, layer-selective |
| Repetitions | `6` per state/depth/layout |
| Shots | `4096` |
| Readout states | the four prepared states per selected layout |
| Readout shots | `8192` |

Circuit count:

- main circuits: `4 states x 3 depths x 2 layouts x 6 reps = 144`;
- readout circuits: `4 states x 2 layouts = 8`;
- total circuits: `152`.

Expected IBM-reported QPU time: `4-10` minutes.

Budget ceiling: `12` IBM-reported QPU minutes.

## Live Readiness Gates

Before any hardware submission:

- confirm selected backend is account-visible and operational;
- retrieve calibration metadata immediately before layout selection;
- run the default, SABRE, and layer-selective transpilation pass from committed
  code only;
- reject if layer-selective layout increases max two-qubit gate count by more
  than 10 % versus default;
- reject if layer-selective layout increases max depth by more than 10 % versus
  default;
- reject if the selected physical qubits have readout or two-qubit errors worse
  than the backend median unless the routing reduction is explicitly recorded;
- record circuit count, shot count, estimated QPU minutes, selected qubits,
  backend name, calibration timestamp, and all transpiler seeds;
- get explicit approval immediately before submission.

## Analysis Plan

Offline primary endpoints:

- compiled depth delta versus default;
- two-qubit gate-count delta versus default;
- high-priority edge physical cost delta versus default;
- stability across transpiler seeds.

Hardware primary endpoints after approved execution:

- parity leakage and exact-state retention by layout method;
- state/readout-corrected leakage delta where exact-state calibration exists;
- depth-normalized leakage comparison so layout gains are not confused with
  circuit-depth changes.

Report both signs:

- a positive result means the layer-selective rule reduced resource burden or
  leakage for the sampled backend/window;
- a negative result means default/SABRE routing is sufficient or the heuristic
  is not useful for this circuit family.

## Falsification Rules

The layer-selective claim is rejected or downgraded if:

- offline resource metrics worsen against default or SABRE;
- gains appear only for one transpiler seed;
- selected qubits have calibration outliers that explain the result;
- exact-state readout correction removes the leakage benefit;
- benefit does not survive depth-normalized analysis.

## Output Artefacts

Expected paths after offline readiness:

- `data/phase3_layer_layout/layer_selective_readiness_<backend>_<date>.json`;
- `data/phase3_layer_layout/layer_selective_transpile_rows_<date>.csv`;
- `docs/phase3_layer_layout_readiness_<date>.md`.

Expected paths after approved hardware execution:

- `data/phase3_layer_layout/layer_selective_counts_<backend>_<timestamp>.json`;
- `data/phase3_layer_layout/layer_selective_summary_<date>.json`;
- `data/phase3_layer_layout/layer_selective_metrics_<date>.csv`;
- `docs/phase3_layer_layout_manifest_<date>.md`.

Every artefact must include backend, calibration snapshot, selected layouts,
transpiler seeds, depth and gate summaries, raw counts where applicable,
SHA256 hashes, and reproduction commands.

## Submission Boundary

This preregistration is complete. Hardware execution remains blocked until
offline readiness artefacts, backend selection, budget confirmation, and
explicit approval are completed in a separate task.
