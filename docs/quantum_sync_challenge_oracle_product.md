# Quantum Sync Challenge oracle product

Claim-governed **synchronisation challenge oracle** façade over ambient sync
witnesses, objectives, and coupling recovery. Refuse invent-green quantum
advantage, unvalidated leaderboard ranks, and hardware execution without ticket.

Module: `scpn_quantum_control.quantum_sync_challenge_oracle_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `quantum_sync_challenge_oracle_product.v1` |
| Families | F1–F4 synthetic + FH hardware schema-only |
| Unknown family | Fail closed |
| Advantage invent-green | Refuse |
| Leaderboard rank without validation | Refuse |
| Hardware execution | Schema residual (S32.10) |
| Anti-cheat | SHA-256 instance digests (family+seed+schema) |

## Problem families

| family_id | support_status |
|---|---|
| `F1_all_to_all_kuramoto` | synthetic_deterministic |
| `F2_sparse_ring_xy` | synthetic_deterministic |
| `F3_cluster_sync` | synthetic_deterministic |
| `F4_noisy_finite_shot` | noisy_sim |
| `FH_hardware_gated` | hardware_gated (schema only) |

## Quick start

```python
from scpn_quantum_control.quantum_sync_challenge_oracle_product import (
    assert_quantum_sync_challenge_oracle_product_integrity,
    build_quantum_sync_challenge_oracle_product_registry,
    compute_instance_digest,
    decide_challenge_path,
    materialise_demo_oracle_probe,
)

reg = assert_quantum_sync_challenge_oracle_product_integrity(
    build_quantum_sync_challenge_oracle_product_registry()
)
assert reg["invent_green_advantage_policy"] is False

digest = compute_instance_digest("F1_all_to_all_kuramoto")
assert len(digest) == 64

assert decide_challenge_path(
    "F1_all_to_all_kuramoto", invent_green_advantage=True
).allowed is False

probe = materialise_demo_oracle_probe()
assert probe.witness_all_passed is True
assert probe.invent_green_advantage is False
```

## Public API contracts

### Catalogue discovery

| API | Contract |
|---|---|
| `list_problem_family_ids()` | Return family identifiers in stable catalogue order. |
| `list_metric_ids()` | Return the bounded challenge metric identifiers. |
| `list_baseline_ids()` | Return baseline identifiers without executing them. |
| `get_problem_family(family_id)` | Resolve one family and reject blank or unknown identifiers. |
| `iter_problem_families(support_status=...)` | Return all families or an immutable support-filtered view. |

### Provenance and eligibility

`compute_instance_digest()` binds the product schema, family identifier, and
non-negative seed into a deterministic SHA-256 digest. It rejects unknown
families and negative seeds rather than manufacturing an instance identity.

`decide_challenge_path()` is the required fail-closed decision point. Synthetic
families may proceed only within the declared claim boundary. Invented quantum
advantage, unvalidated leaderboard rank, hardware execution without a ticket,
and the schema-only hardware family return explicit blockers.

### Witness materialisation and registry integrity

| API | Contract |
|---|---|
| `materialise_oracle_probe(family_id)` | Compose the selected family with the ambient synchronisation witness suite and reject empty evidence. |
| `materialise_demo_oracle_probe()` | Run the deterministic F1 local demonstration without provider or hardware work. |
| `map_quantum_sync_challenge_oracle_public_surfaces()` | Emit deterministic descriptors for ambient and product API surfaces. |
| `build_quantum_sync_challenge_oracle_product_registry()` | Build the schema-tagged family, metric, baseline, policy, and surface catalogue. |
| `assert_quantum_sync_challenge_oracle_product_integrity(payload=None)` | Reject missing, blank, duplicate, count-drifted, or invent-green registry state. |

The public data records validate identifiers, support and outcome enums,
digests, witness counts, blockers, no-submit flags, and bounded claim text at
construction time; their `to_dict()` methods return JSON-ready payloads.

## Residuals (honest)

- **S32.4–S32.5** — full classical/quantum baseline runners
- **S32.6–S32.7** — full anti-cheat recompute + leaderboard export depth
- **S32.8–S32.9** — static HTML + CLI
- **S32.10–S32.12** — hardware package template, invitation docs, seal design

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl32_quantum_sync_challenge_oracle.md`
- Ambient: `phase.synchronisation_witness`, `synchronisation_objectives`, `coupling_time_series_recovery`
- BL-34 dashboard, BL-47 no-submit, BL-52/53 honesty

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
