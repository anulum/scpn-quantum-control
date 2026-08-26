# Chimera Control API

The `scpn_quantum_control.chimera_control` package is the public facade
for finite synthetic chimera generation, nested coherence observables,
analytic hierarchy targets, unapplied phase proposals, topology-constraint
composition, and deterministic evidence custody.

All arrays returned by chimera-control custody objects are copied and
read-only. The API is local and simulator-only: it performs no provider
submission, QPU or hardware execution, external actuation, deployment, or
biological inference.

For a task-oriented walkthrough and measured evidence, read
[Chimera and Multiscale Synchronisation Control](../chimera_multiscale_control.md).

## Hierarchy and target contracts

::: scpn_quantum_control.chimera_control.schema
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Synthetic generator

::: scpn_quantum_control.chimera_control.synthetic
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Multiscale observables

::: scpn_quantum_control.chimera_control.observables
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Differentiable targets and proposals

::: scpn_quantum_control.chimera_control.objectives
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Topology-constraint bridge

::: scpn_quantum_control.chimera_control.topology
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Evidence contracts

::: scpn_quantum_control.chimera_control.evidence
    options:
      show_root_heading: true
      show_source: false
      members_order: source
