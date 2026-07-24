# Fixture-driven visualisation dashboard product (BL-34)

Versioned **static panel catalogue + fixture probes** for order-parameter/energy,
gradient norms, and related panel families. `live_qpu=false` honesty; no SaaS.

Module: `scpn_quantum_control.visualisation_dashboard_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `visualisation_dashboard_product.v1` |
| Default panel | `order_parameter_energy_loss` |
| Live QPU stream | Refused |
| Always-on SaaS | Refused |
| Secrets in export | Scanned and refused |
| Blank/unknown panel | Fail closed |

Claim boundary:

> Fixture-driven visualisation dashboard product surface only; catalogues
> static panel families and materialises local report probes from synthetic
> or allowed fixtures; live_qpu=false honesty; refuses invent-green live QPU
> streaming and always-on SaaS dashboard; does not claim full multi-panel
> CLI suite or BL-32 embeds (S34.3–S34.10 residual)

## Public API

```python
from scpn_quantum_control.visualisation_dashboard_product import (
    assert_visualisation_dashboard_product_integrity,
    build_visualisation_dashboard_product_registry,
    decide_visualisation_path,
    list_visualisation_panel_ids,
    materialise_demo_static_report_probe,
    scan_export_for_secrets,
)

assert "order_parameter_energy_loss" in list_visualisation_panel_ids()
reg = assert_visualisation_dashboard_product_integrity(
    build_visualisation_dashboard_product_registry()
)
probe = materialise_demo_static_report_probe()
assert probe.live_qpu is False
assert probe.series_point_count > 0
assert len(probe.fixture_digest_sha256) == 64

assert decide_visualisation_path(request_live_qpu_stream=True).allowed is False
assert scan_export_for_secrets('{"x":1}').clean is True
```

## Catalogue (S34.0)

| ID | Kind |
|---|---|
| `order_parameter_energy_loss` | series |
| `gradient_norm` | series |
| `coupling_heatmap` | catalogue |
| `witness_summary` | catalogue |
| `bitstring_saved_pack` | catalogue |
| `refuse_live_qpu_stream` | refuse |

## Bounded product status

Shipped: S34.0 design freeze · S34.1 panel models + secrets scan · S34.2
order-parameter/energy + gradient-norm materialised demo · refuse live QPU/SaaS ·
docs / API map.

Open residual: S34.3–S34.6 remaining panel bodies · S34.7 CLI bundle writer ·
S34.8 BL-32 embeds · S34.10 notebook widgets.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
