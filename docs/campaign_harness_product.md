# Campaign harness productisation (BL-99)

Reusable **hardware-campaign harness templates** with prereg digests and BL-47
no-submit default. Dry-run probes only. Refuse invent-green live QPU submit and
unattested claim promotion.

Module: `scpn_quantum_control.campaign_harness_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `campaign_harness_product.v1` |
| Default mode | dry_run / no_submit |
| Live submit | Owner ticket residual; would_live refused |
| Unattested claims | Refuse (S99.4 residual) |
| Prereg mutation after freeze | Refuse |
| Unknown harness | Fail closed |

## Harnesses

| harness_id | ambient |
|---|---|
| `appqsim_protocol` | `benchmarks.appqsim_protocol` |
| `iqm_layout_transfer` | `benchmarks.iqm_layout_transfer_benchmark` |
| `closed_loop_publication` | `benchmarks.closed_loop_publication_run` |
| `benchmark_harness_registry` | `benchmark_harness.registry` |

## Quick start

```python
from scpn_quantum_control.campaign_harness_product import (
    assert_campaign_harness_product_integrity,
    build_campaign_harness_product_registry,
    decide_campaign_path,
    materialise_demo_campaign_probe,
)

reg = assert_campaign_harness_product_integrity(
    build_campaign_harness_product_registry()
)
assert reg["no_submit_default_policy"] is True

assert decide_campaign_path(
    "appqsim_protocol", invent_green_live_submit=True
).allowed is False

probe = materialise_demo_campaign_probe()
assert probe.invent_green_live_submit is False
assert probe.attestation_slot_present is False
```

## Residuals (honest)

- **S99.4** — BL-55 hermetic kit + BL-48 attestation sealing slots
- Full multi-size IQM campaign execution remains ticketed residual

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl99_campaign_harness_productisation.md`
- Ambient: appqsim, iqm_layout_transfer, closed_loop_publication, benchmark_harness
- BL-47 no-submit · BL-65 advantage protocol · BL-67 control compose

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
