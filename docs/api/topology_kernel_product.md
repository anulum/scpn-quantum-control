# Topology Kernel Product API

`scpn_quantum_control.topology_kernel_product` is the public facade for
bounded topology-aware fidelity kernels, custody-checked kernel ridge
classification, deterministic graph controls, and frozen synthetic evidence.

All quantum values use exact local statevectors. Public records hold defensive,
read-only arrays and SHA-256 custody. Identifier, topology, shape, finiteness,
and resource mismatches raise rather than being silently coerced. No API on
this page submits a provider job or claims hardware execution, quantum
advantage, independent generalisation, or application-domain fitness.

For equations, a complete workflow, frozen metrics, scientific sources, and
non-claims, read [Topology-Aware Quantum Kernel](../topology_aware_quantum_kernel.md).

## Configuration and immutable records

::: scpn_quantum_control.topology_kernel_product.schema
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Fidelity and classical kernels

::: scpn_quantum_control.topology_kernel_product.kernels
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Kernel ridge classification

::: scpn_quantum_control.topology_kernel_product.classifier
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Synthetic task and graph controls

::: scpn_quantum_control.topology_kernel_product.synthetic
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Evidence contracts and rendering

::: scpn_quantum_control.topology_kernel_product.evidence
    options:
      show_root_heading: true
      show_source: false
      members_order: source
