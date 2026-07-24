# Quantum Sync Challenge oracle product (BL-32)

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
