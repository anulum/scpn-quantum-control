# Cloud-native deployment boundary product

Documented, fail-closed **cloud deploy patterns** for workers/batch. Dry-run
manifest generation only (Kubernetes + Docker Compose). No secret leakage, no
always-on QPU, no live cluster create.

Module: `scpn_quantum_control.cloud_native_deployment_product`

## Rules

| Rule | Behaviour |
|---|---|
| Product schema | `cloud_native_deployment_product.v2` |
| Ambient generator | `deployment.cloud_native.generate_cloud_manifests` |
| Secret-like env | Refuse (ambient + product pre-check) |
| Always-on QPU | Refuse (hardware-safe no-submit and dry-run compute planning) |
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

## Public API contracts

### Pattern and threat discovery

| API | Contract |
|---|---|
| `list_deployment_pattern_ids()` | Return stable deployment-pattern identifiers in catalogue order. |
| `list_threat_ids()` | Return the bounded cloud threat-model identifiers. |
| `get_deployment_pattern(pattern_id)` | Resolve one immutable pattern row; blank and unknown identifiers raise `ValueError`. |
| `iter_deployment_patterns(kind=None)` | Return every pattern or a stable kind-filtered tuple. |

`DeploymentPatternRow` validates identity, execution kind, image command,
replica posture, QPU/secret/cluster prohibitions, evidence pointers, and claim
boundary. `ThreatModelRow` validates each threat, mitigation, severity, and
fail-closed state. Both records provide JSON-ready `to_dict()` payloads.

### Eligibility and manifest dry runs

| API | Contract |
|---|---|
| `decide_deploy_path(pattern_id, ...)` | Allow bounded manifest generation and return explicit blockers for always-on QPU, live cluster creation, secret injection, or credential loading. |
| `compute_spec_digest(name, image, command, replicas=1, env=None)` | Bind a validated deployment specification to a deterministic SHA-256 digest without resolving credentials. |
| `materialise_deploy_dry_run_probe(pattern_id, ...)` | Compose the ambient manifest generator, reject secret-like environment keys, and return content-addressed filenames without applying them. |
| `materialise_demo_deploy_dry_run_probe()` | Materialise the deterministic batch-worker dry run. |

`PathEligibilityDecision` keeps outcome, permission, reason, and blockers
mutually consistent. `MaterialisedDeployDryRunProbe` validates its pattern,
manifest digest, file set, ambient claim boundary, and false live-cluster,
always-on-QPU, and secret flags.

### Registry evidence and integrity

| API | Contract |
|---|---|
| `map_cloud_native_deployment_public_surfaces()` | Emit deterministic product and ambient generator descriptors with pattern and claim metadata. |
| `build_cloud_native_deployment_product_registry()` | Build schema-tagged patterns, threats, surfaces, policy flags, counts, and residual-work evidence. |
| `assert_cloud_native_deployment_product_integrity(payload=None)` | Reject empty, malformed, blank, invalid-kind/severity, duplicate, count-drifted, or permissive QPU/secret/cluster state. |

These APIs generate and validate deployment files only. They do not create a
cluster, load credentials, submit QPU work, expose secrets, apply manifests, or
claim enterprise operations evidence.

## Open operations depth

- Fuller enterprise packaging and operations runbooks remain open
- Ambient generator remains the implementation spine

## Related

- Ambient: `scpn_quantum_control.deployment.cloud_native`
- Hardware-safe execution · dry-run QPU compute planning · campaign harness

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
