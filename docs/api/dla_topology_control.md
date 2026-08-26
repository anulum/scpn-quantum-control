# DLA Topology Control API

The `scpn_quantum_control.dla_topology_control` package is the public facade for
finite parity-sector derivatives, fixed-active-set topology-ledger
sensitivities, a synthetic projected-gradient task, and deterministic evidence
custody.

All returned arrays are copied and read-only. Unsupported non-smooth or
discrete branches raise instead of returning approximate or invented
derivatives. The package performs no provider submission, QPU or hardware
execution, external actuation, deployment, or error-correction action.

For equations, support tables, examples, evidence, and scientific sources,
read [DLA and Topology-Constrained Differentiable Control](../dla_topology_constrained_control.md).

## Support and parity contracts

::: scpn_quantum_control.dla_topology_control.schema
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Parity-sector projection

::: scpn_quantum_control.dla_topology_control.parity
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Topology projection sensitivity

::: scpn_quantum_control.dla_topology_control.projection
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Synthetic protected objective

::: scpn_quantum_control.dla_topology_control.objectives
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Projected-gradient loop

::: scpn_quantum_control.dla_topology_control.optimizer
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Evidence contracts

::: scpn_quantum_control.dla_topology_control.evidence
    options:
      show_root_heading: true
      show_source: false
      members_order: source
