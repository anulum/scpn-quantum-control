# Cloud-native deployment boundary product (BL-101)

Documented, fail-closed **cloud deploy patterns** for workers/batch. Dry-run
manifest generation only (Kubernetes + Docker Compose). No secret leakage, no
always-on QPU, no live cluster create.

Module: `scpn_quantum_control.cloud_native_deployment_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `cloud_native_deployment_product.v1` |
| Ambient generator | `deployment.cloud_native.generate_cloud_manifests` |
| Secret-like env | Refuse (ambient + product pre-check) |
| Always-on QPU | Refuse (BL-47 / BL-95 compose) |
| Live cluster create | Refuse |
| Credential loading | Refuse |
| Unknown pattern | Fail closed |

## Patterns

| pattern_id | role |
|---|---|
| `batch_worker` | Offline batch worker |
| `stable_core_gate` | Stable-core contract gate job |
| `offline_research` | Low-replica research packaging demo |

## Threat model (fail-closed)

`secret_leakage` · `always_on_qpu` · `live_cluster_create` · `credential_loading` · `unbounded_cost`

## Quick start

```python
from scpn_quantum_control.cloud_native_deployment_product import (
    assert_cloud_native_deployment_product_integrity,
    build_cloud_native_deployment_product_registry,
    decide_deploy_path,
    materialise_demo_deploy_dry_run_probe,
)

reg = assert_cloud_native_deployment_product_integrity(
    build_cloud_native_deployment_product_registry()
)
assert reg["allows_always_on_qpu_policy"] is False

assert decide_deploy_path(
    "batch_worker", invent_green_always_on_qpu=True
).allowed is False

probe = materialise_demo_deploy_dry_run_probe()
assert "deployment.yaml" in probe.file_names
assert probe.invent_green_live_cluster is False
```

## Residuals (honest)

- **S101.3** — fuller enterprise packaging / ops runbooks depth
- Ambient generator remains the implementation spine

## Related

- Pack: `docs/internal/differentiable_programming/p3_strategic/bl101_cloud_native_deployment_boundary.md`
- Ambient: `scpn_quantum_control.deployment.cloud_native`
- BL-47 hardware-safe · BL-95 QPU compute · BL-99 campaign harness

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
