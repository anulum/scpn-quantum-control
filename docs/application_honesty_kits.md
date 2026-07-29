# Domain Application Honesty Kits

Domain-facing examples are useful only when their evidence boundary survives
copying into notebooks, proposals, and downstream integrations. BL-63 makes
that boundary executable. Each honesty kit states what the software route can
demonstrate, what data provenance it admits, and which claims remain forbidden.

The kits are policy-bearing software records, not certificates of scientific
or operational validity. They do not promote a synthetic fixture, curated
matrix, structural proxy, or passing unit test into domain evidence.

## Quick start

```python
from scpn_quantum_control.applications import (
    build_application_honesty_audit_report,
    get_domain_application_honesty_kit,
)

kit = get_domain_application_honesty_kit("eeg_like_synthetic")
assert kit.synthetic_only
assert not kit.publication_safe
assert "clinical classification" in kit.claims_forbidden

report = build_application_honesty_audit_report()
assert report.passed
print(report.content_digest())
```

This code reads only versioned packaged application artifacts. It performs no
network access, private-data discovery, provider submission, or hardware work.

## Policy schema

`DomainApplicationHonestyKit` is immutable. Its fields have distinct jobs:

| Field | Meaning |
|---|---|
| `kit_id` | Stable machine identifier used for lookup and evidence. |
| `domain_tag` | Non-promotional label for reports and interfaces. |
| `support_status` | `bounded_research` or `simulation_only`. |
| `data_origin` | `synthetic` or `curated_public`. |
| `synthetic_only` | Exact shorthand that must agree with `data_origin`. |
| `dataset_ids` | Packaged public artifacts governed by the kit; forbidden for synthetic-only kits. |
| `source_modules` | Implementing import paths, not a claim of domain fidelity. |
| `allowed_uses` | Narrow software demonstrations admitted by the kit. |
| `caveats` | Scientific and operational limitations that must travel with the result. |
| `claims_forbidden` | Explicit claims the kit never authorises. |
| `forecasting_tags` | BL-37 generator tags; always simulation-only. |
| `publication_safe` | Always `False`: the kit itself is not domain-publication evidence. |

Construction fails closed on blank or duplicate policy text, enum bypasses,
inconsistent synthetic flags, packaged datasets on synthetic-only routes, and
duplicate forecasting tags.

## Built-in kits

### Public power-grid benchmark

`power_grid_public_benchmark` governs the packaged IEEE 5-bus constants. It
supports artifact-custody checks, Kuramoto compilation, and structural-proxy
software tests. It does not support live-grid control, transient-stability
prediction, utility deployment, or quantum-advantage claims.

The related BL-37 tag is `grid_like_sim`. That tag still denotes a generated
forecasting configuration, even though this separate kit also governs one
curated public benchmark.

### Illustrative Josephson simulation

`josephson_illustrative_simulation` admits generated coupling topologies and
explicitly labelled nominal literature parameters. It is simulation-only.
An XY structural analogy is not measured-device validation, hardware
self-simulation, calibration guidance, or fabrication evidence.

### Synthetic EEG-like software route

`eeg_like_synthetic` admits generated PLV-shaped inputs and the BL-37
`eeg_like_sim` tag. It deliberately names no packaged dataset. EEG-like means
only that a generator has a documented matrix shape and feature convention; it
does not mean clinical classification, neural-dynamics reproduction, diagnosis,
therapy, or human-subject generalisation.

The separately packaged `eeg_alpha_plv_8ch` artifact remains a small curated
public-literature benchmark matrix. It is audited by the dataset privacy API,
but it is not silently relabelled as synthetic or promoted by this kit.

### ITER-inspired disruption simulation

`iter_disruption_inspired_simulation` governs opt-in generated disruption
features and local advisory-contract exercises. Its related BL-37 tag is
`plasma_like_sim`. It does not validate ITER prediction, facility operation,
real-time controller admission, or plasma-operation guidance.

Measured fusion archives remain outside this kit and require their own custody,
licence, validation, and downstream control-admission gates.

## Dataset privacy audit

`audit_application_benchmark_privacy()` validates every artifact in
`data/public_application_benchmarks/` against its descriptor:

- stable source identity and domain;
- exact curated source mode;
- an exact privacy/licence boundary embedded in artifact metadata;
- `contains_personal_data=False` for the packaged bytes;
- publication-safe artifact custody and array hashes.

The audit covers EEG, plasma, power-grid, and FEP packaged rows. It does not
claim that arbitrary third-party plugin inputs are public or safe. Plugin
authors remain responsible for their own raw-data governance.

```python
from scpn_quantum_control.applications import audit_application_benchmark_privacy

for row in audit_application_benchmark_privacy():
    print(row.dataset_id, row.privacy_classification, row.passed)
```

A mismatch raises immediately. The API does not return an ambiguous
"mostly passed" catalogue.

## Deterministic evidence

Regenerate the committed report with:

```bash
PYTHONPATH=src:oscillatools/src .venv-linux/bin/python scripts/run_application_honesty_audit.py
```

Verify byte-for-byte freshness without writing:

```bash
PYTHONPATH=src:oscillatools/src .venv-linux/bin/python scripts/run_application_honesty_audit.py --check
```

The JSON payload includes the full kits, privacy rows, artifact hashes, global
claim boundary, and a SHA-256 content digest. The Markdown file is a compact
human-readable view of the same report. Neither file is measured-domain,
clinical, facility, hardware, or advantage evidence.

## Extending the registry

A new kit should be added only with all of the following:

1. one stable, non-promotional identifier;
2. an exact data-origin decision;
3. source modules and any governed packaged dataset identifiers;
4. positive allowed uses plus caveats and forbidden claims;
5. privacy metadata for every new packaged artifact;
6. focused tests for failure paths, JSON stability, and evidence freshness;
7. documentation and generated capability inventory updates.

Do not reuse a simulation tag for measured data. Do not mark curated data as
synthetic. Do not use `publication_safe=False` as a substitute for concrete
forbidden claims. Provider, QPU, clinical, facility, operational, and advantage
promotion remain separate evidence programmes.

## Complete API reference

The [Application Honesty API](api/application_honesty_kits.md) documents every
public enum, record, lookup, audit, digest, and renderer directly from source
docstrings.
