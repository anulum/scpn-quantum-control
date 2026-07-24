# Advanced witnesses product (BL-44)

Fail-closed product surface for **estimator-aware scientific diagnostics**:
Krylov complexity, out-of-time-order correlators (OTOC), and classical shadows,
with uncertainty and provenance — beyond energy and scalar order parameters.

## Module

`scpn_quantum_control.advanced_witnesses_product`

Schema: `advanced_witnesses_product.v1`

## Ambient (inventory-first)

| Ambient | Role |
|---|---|
| `analysis.krylov_complexity.krylov_complexity` | Operator Lanczos Krylov complexity K(t) |
| `analysis.otoc.compute_otoc` | OTOC F(t), Lyapunov / scrambling-time estimates |
| `analysis.shadow_tomography.classical_shadow_estimation` | Classical-shadow Pauli estimators |
| `phase.synchronisation_witness.harmonic_order_parameter` | BL-18 compose (order parameter) |

Do **not** reimplement these modules; the product is a thin façade.

## Capabilities

- `krylov_complexity` — bounded unitary Krylov diagnostic
- `otoc_probe` — supported gate-model / Kuramoto-XY OTOC probe
- `classical_shadows` — Pauli shadow estimator with shadow-norm bound
- `small_tomography_cap` — tomography only under product qubit cap
- `ambient_inventory` — metadata inventory of ambient modules
- `bl18_sync_compose` — compose harmonic order parameter (BL-18)

## Hard caps

| Cap | Value |
|---|---:|
| `MAX_WITNESS_QUBITS` | 6 |
| `MAX_DEMO_SHADOW_SHOTS` | 200 |
| `MIN_SHADOW_SHOTS` | 16 (below → `support_status=under_sampled`) |

## Refuse invent-green

- OTOC quantum **advantage** claims
- Topology / topological-phase **certification**
- Live **QPU** witness campaigns
- Unrestricted shadow tomography without support profile
- Silent green on **under-sampled** shadows

## Entry points

```python
from scpn_quantum_control.advanced_witnesses_product import (
    assert_advanced_witnesses_product_integrity,
    build_advanced_witnesses_product_registry,
    decide_witness_path,
    materialise_demo_krylov_probe,
    materialise_demo_otoc_probe,
    materialise_demo_shadow_probe,
    materialise_bl18_order_parameter_compose,
)

reg = assert_advanced_witnesses_product_integrity(
    build_advanced_witnesses_product_registry()
)
assert decide_witness_path("otoc", invent_green_otoc_advantage=True).allowed is False
k = materialise_demo_krylov_probe()
o = materialise_demo_otoc_probe()
s = materialise_demo_shadow_probe()
r = materialise_bl18_order_parameter_compose()
```

## Residuals (honest)

- **S44.6** — full suite evidence artefact depth (beyond single probes)
- **S44.7** — optional BL-32 metric registration + BL-34 panel hooks

## Claim boundary

Advanced witnesses product surface only; catalogues Krylov/OTOC/classical-shadow
estimators with uncertainty and provenance over ambient `analysis/*`; small-system
probes with hard qubit/shot caps; refuse invent-green OTOC advantage, topology
certification, live QPU witness campaigns, and unrestricted shadow tomography;
compose BL-18 order parameters; residual S44.7 / S44.6 open honestly.

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
