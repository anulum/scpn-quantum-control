<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
(c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
(c) Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
scpn-quantum-control -- Paper 0 K_nm replay contract
-->

# GOTM-SCPN Paper 0 K_nm preregistered replay contract

This page defines the public contract for the first downstream replay derived
from GOTM-SCPN Paper 0: The Foundational Framework. It is a contract for
offline reproducibility and promotion safety, not a claim that K_nm has been
validated as a measured physical coupling law.

## Contract surface

- Replay schema: `paper0_knm_preregistered_replay_v1`
- Contract schema: `paper0_knm_preregistered_replay_contract_v1`
- Replay artefact: `data/paper0_knm_preregistered_replay.json`
- Human report: `docs/paper0_knm_preregistered_replay.md`
- Contract exporter: `scripts/export_paper0_knm_replay_contract.py`
- Comparator: `scripts/compare_paper0_knm_preregistered_replay.py`
- Release gate: `scpn-bench paper0-knm-preregistered-replay-gate`

## Required replay keys

The replay JSON must preserve these top-level keys:

- `schema`
- `paper`
- `status`
- `claim_boundary`
- `reproducibility`
- `inputs`
- `primary_candidate`
- `negative_control`
- `gates`
- `promotion_decision`
- `next_required_artifacts`

The replay status must remain
`blocked_non_closing_preregistered_replay` until a later reviewed promotion
gate records calibrated coupling magnitudes, per-edge uncertainty, a matched
null battery, and a frozen manifest.

## Locked input manifest

The replay input manifest is closed over exactly these repository-local inputs:

| Manifest key | Path |
|---|---|
| `primary_candidate` | `data/public_application_benchmarks/eeg_alpha_plv_8ch.json` |
| `negative_control` | `data/public_application_benchmarks/ieee5bus_power_grid.json` |
| `negative_measured_couplings` | `data/knm_physical_validation/measured_couplings_power_grid_ieee5bus.json` |

Each input entry must expose a repository-relative `path` and `sha256`. The
comparator recomputes those digests from the current checkout and fails if a
digest is stale, forged, missing, or points outside the repository.

## Required diagnostics

Each primary and negative-control matrix diagnostic block must expose:

- `pearson_upper`
- `spearman_upper`
- `frobenius_relative_error`
- `density`
- `candidate_edge_count`
- `reference_edge_count`
- `shared_edge_count`

Each permutation-null block must expose:

- `seed`
- `permutations`
- `observed_pearson_upper`
- `null_mean_pearson_upper`
- `null_std_pearson_upper`
- `observed_vs_null_z`
- `two_sided_empirical_p`

These metrics are audit diagnostics only. They cannot by themselves promote a
dimensionless synchronisation matrix into a calibrated measured-system
coupling matrix.

## Fail-closed promotion contract

The replay must keep these gate states unless a later promotion artefact is
reviewed and replaces this contract:

| Gate | Required state |
|---|---|
| `named_system` | `pass` |
| `units_and_normalisation` | `blocked_primary_dimensionless_plv` |
| `pairwise_uncertainty` | `blocked_primary_missing_per_edge_uncertainty` |
| `negative_control` | `pass_non_promotional_sparse_control_remains_non_closing` |
| `qpu_submission` | `blocked_no_qpu_preregistration_lane` |
| `claim_promotion` | `blocked_measured_system_gate_open` |

The `promotion_decision` block must preserve:

- `decision`: `do_not_promote`
- `hardware_submission_authorised`: `false`
- `claim_promotion_authorised`: `false`
- At least four required evidence items before reconsideration.
- At least four falsifiers.

Any change that weakens these requirements is a contract failure, even when the
generated JSON and committed JSON are byte-aligned.

## Export command

The normative contract can be emitted without touching the replay artefacts:

```bash
PYTHONPATH=src ./.venv-linux/bin/python scripts/export_paper0_knm_replay_contract.py
```

Use `--output-json <path>` only for temporary review artefacts or release
packaging. Do not treat an exported contract as measured-system evidence.

## Replay-check command

The same command can validate an existing replay JSON against the fail-closed
contract:

```bash
PYTHONPATH=src ./.venv-linux/bin/python scripts/export_paper0_knm_replay_contract.py \
  --check-replay data/paper0_knm_preregistered_replay.json
```

The check fails if the replay changes schema, removes required fields, weakens
the locked input paths, changes required gate states, authorises hardware
submission, authorises claim promotion, or drops the required evidence and
falsifier lists. This is intentionally stricter than a JSON-byte comparison:
the contract is a semantic guard for the Paper 0 pathway boundary.
