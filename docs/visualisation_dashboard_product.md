# Fixture-driven visualisation dashboard product

Versioned **static panel catalogue + fixture probes** for order-parameter/energy,
gradient norms, and related panel families. `live_qpu=false` honesty; no SaaS.

Module: `scpn_quantum_control.visualisation_dashboard_product`

This page documents a bounded static-report facade, not a live monitoring
service. The catalogue describes supported and residual panel families while
the materialised probe operates only on a deterministic local fixture.

## Contract discovery

| Function | Contract |
|---|---|
| `list_visualisation_panel_ids()` | Returns every stable panel id in catalogue order. |
| `get_visualisation_panel(panel_id)` | Resolves one exact row; blank and unknown ids raise `ValueError`. |
| `iter_visualisation_panels(...)` | Filters deterministically by panel kind and/or support posture. |
| `map_visualisation_dashboard_public_surfaces()` | Identifies the product owner and the ambient status-only facade. |

Discovery is static and local. It opens no fixture file, reads no credential,
contacts no provider, and starts no dashboard process.

## Public value objects

- `VisualisationPanelRow` maps a stable panel id to its kind, public owner,
  symbol, support posture, and `live_qpu=False` boundary.
- `PathEligibilityDecision` records an allowed/refused outcome, reason, ordered
  blockers, and the shared non-promotional claim boundary.
- `SecretsScanResult` records whether an export string is clean and the matched
  detector labels when it is not.
- `MaterialisedStaticReportProbe` records panel ids, series counts, the
  canonical fixture digest, honesty flags, and demo provenance.

All records are immutable slot-backed dataclasses with validated construction
and JSON-ready `to_dict()` mappings. A probe is local fixture evidence only; it
is not proof of a live UI, general data compatibility, or hardware execution.

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `visualisation_dashboard_product.v2` |
| Demo fixture schema | `visualisation_demo_fixture.v2` |
| Default panel | `order_parameter_energy_loss` |
| Live QPU stream | Refused |
| Always-on SaaS | Refused |
| Secrets in export | Scanned and refused |
| Blank/unknown panel | Fail closed |

## Eligibility and export safety

`decide_visualisation_path()` permits only an explicitly fixture-driven static
path. It returns a structured refusal when live-QPU streaming, an always-on
SaaS dashboard, or a non-fixture path is requested. Multiple blockers are
de-duplicated in first-seen order.

`scan_export_for_secrets()` accepts text only and checks bounded API-key,
token, bearer-token, and `sk-...` forms. A dirty scan contains detector labels,
not recovered secret values. Treat the scanner as a fail-closed product guard,
not as a complete data-loss-prevention system.

Claim boundary:

> This fixture-driven visualisation dashboard product catalogues static panel
> families and materialises local report probes from synthetic or explicitly
> allowed fixtures. It sets live_qpu=false and refuses live QPU streaming and
> always-on SaaS dashboard claims. Remaining panel bodies, a command-line
> bundle writer, challenge-result embeds, and notebook widgets remain outside
> the current product.

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

## Static report materialisation

`materialise_demo_static_report_probe()` builds order-parameter, energy/loss,
and gradient-norm series from the module's deterministic synthetic fixture. It
requires the exact v2 fixture schema, non-empty equal-length order/energy lists,
and non-empty gradient norms, serialises the fixture as canonical sorted compact
JSON, computes its lowercase SHA-256 digest, and refuses a dirty export scan.

The returned bundle names only the two materialised demo panels. Catalogue-only
coupling, witness, and saved-bitstring rows do not become implemented panel
bodies through this probe.

## Panel catalogue

| ID | Kind |
|---|---|
| `order_parameter_energy_loss` | series |
| `gradient_norm` | series |
| `coupling_heatmap` | catalogue |
| `witness_summary` | catalogue |
| `bitstring_saved_pack` | catalogue |
| `refuse_live_qpu_stream` | refuse |

## Registry integrity

`build_visualisation_dashboard_product_registry()` emits schema
`visualisation_dashboard_product.v2`, the full panel catalogue, public surface
map, default id, counts, policy note, and shared claim boundary.

Always validate transported or stored payloads through
`assert_visualisation_dashboard_product_integrity()`. It rejects:

- missing, empty, non-list, non-mapping, blank, duplicate, missing, or extra rows;
- unknown panel kinds or missing symbol names;
- any row with `live_qpu=True` or a relaxed registry policy;
- loss of the default or explicit refuse panel; and
- schema, claim-boundary, policy, canonical-row, public-surface, default-id, or
  count drift.

## Failure handling and operational non-effects

Treat `ValueError` as a caller-contract, fixture, export-safety, or transported
registry failure. Treat `RuntimeError` from catalogue construction as
repository corruption.

This product performs no network access, credential lookup, provider or QPU
discovery, hardware submission, live streaming, SaaS deployment, HTML bundle
write, browser launch, notebook mutation, or evidence promotion. The ambient
`differentiable_dashboard` reference is a status/capability pointer only.

## Bounded product status

Shipped: design freeze · panel models and secrets scan · order-parameter/energy
and gradient-norm materialised demo · live QPU/SaaS refusal · documentation and
API map.

Open product work: remaining panel bodies · command-line bundle writer ·
challenge-result embeds · notebook widgets.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
