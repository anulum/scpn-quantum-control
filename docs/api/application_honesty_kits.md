# Application Honesty API

`scpn_quantum_control.applications.honesty_kits` is the public policy
surface for immutable domain claim boundaries and deterministic evidence.
`scpn_quantum_control.applications.dataset_catalog` owns the complementary
packaged-artifact privacy audit.

These APIs validate software metadata and versioned local artifacts only. They
do not certify domain fidelity, clinical use, facility prediction, operational
control, hardware performance, or quantum advantage. For the guided workflow
and extension rules, read [Domain Application Honesty Kits](../application_honesty_kits.md).

## Honesty-kit policy records

::: scpn_quantum_control.applications.honesty_kits
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_signature_annotations: true
      separate_signature: true

## Packaged-dataset privacy records

::: scpn_quantum_control.applications.dataset_catalog.ApplicationBenchmarkDescriptor
    options:
      show_root_heading: true
      show_source: false

::: scpn_quantum_control.applications.dataset_catalog.ApplicationBenchmarkPrivacyAudit
    options:
      show_root_heading: true
      show_source: false

::: scpn_quantum_control.applications.dataset_catalog.audit_application_benchmark_privacy
    options:
      show_root_heading: true
      show_source: false
