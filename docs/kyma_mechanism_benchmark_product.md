# KYMA / KYMA v2 mechanism benchmark product (BL-73)

**Public, preregistered, mechanism-only** sync-learning honesty surface. Design
constants come from **teacher dynamics only** (prereg §5) — never student
held-out accuracy. Refuse post-hoc retuning and invent-green advantage without
protocol id.

Module: `scpn_quantum_control.kyma_mechanism_benchmark_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `kyma_mechanism_benchmark_product.v1` |
| Design freeze | Ambient `kyma_v2.design` grids + targets + digest |
| Student held-out design | Refuse |
| Post-hoc retune | Refuse |
| Advantage invent-green | Refuse (need protocol id / BL-65) |
| Unknown suite | Fail closed |

## Suites

| suite_id | role |
|---|---|
| `kyma_v1` | Honest NEGATIVE baseline |
| `kyma_v2` | Corrected mechanism-only design |

Protocol: `KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21`

## Quick start

```python
from scpn_quantum_control.kyma_mechanism_benchmark_product import (
    assert_kyma_mechanism_benchmark_product_integrity,
    build_kyma_mechanism_benchmark_product_registry,
    decide_kyma_path,
    get_frozen_design_constants,
    materialise_demo_mechanism_certificate_probe,
)

reg = assert_kyma_mechanism_benchmark_product_integrity(
    build_kyma_mechanism_benchmark_product_registry()
)
assert reg["invent_green_advantage_policy"] is False
assert len(get_frozen_design_constants().content_digest) == 64

assert decide_kyma_path("kyma_v2", invent_green_advantage=True).allowed is False

probe = materialise_demo_mechanism_certificate_probe()
assert probe.meets_realise_target is True
assert probe.design_from_student_held_out is False
```

## Residuals (honest)

- **S73.3** — full classical ML baseline harness depth + multi-seed evidence
- **S73.4** — optional BL-32 family registration wire
- **S73.6** — hermetic kit entry (BL-55)

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl73_kyma_public_mechanism_benchmark.md`
- Ambient: `benchmarks.kyma`, `benchmarks.kyma_v2.design`, `task`, `teacher`
- Campaigns: `docs/campaigns/kyma_v2_composition_probe_2026-07-21.md`

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
