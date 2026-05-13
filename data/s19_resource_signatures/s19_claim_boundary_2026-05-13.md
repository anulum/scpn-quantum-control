# S19 Resource-Signature Claim Boundary

- Schema: `scpn_s19_resource_signature_scan_v1`
- Row count: `2`
- Status: simulator-only, not submitted to hardware.
- Separation: this does not edit any submitted manuscript.

## Boundary

Simulator-only finite-size S19 scan. This is not a hardware claim, not a quantum-advantage claim, and not an edit to any submitted manuscript.

## Paper 0 Source Boundary

The Paper 0 topology source boundary is attached as provenance only. The
source boundary is not a numeric coupling matrix and is not provider-ready.
Every simulated row must carry a separate numeric-topology label.

| topology | numeric topology label | Paper 0 topology claim |
|---|---|---:|
| ring | synthetic_control.ring | False |

This is not a hardware claim, not a thermodynamic-limit claim, and not a
quantum-advantage claim. It is an offline finite-size scan used to decide
whether an S19 paper lane deserves larger simulation and later IBM gating.

## Alignment Summary

| n | topology | omega profile | onset estimate | entropy max K | Schmidt min K | magic max K | pairing max K | Krylov max K | mean distance |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | ring | disorder_seed_11 | 0.2 | 1 | 1 | 0.5 | 1 | 1 | 0.7 |

## Ensemble Alignment

| n | topology | boundary | realisations | extremum score | extremum CI95 | curvature score | curvature CI95 | best control | onset-control | control status | failing observables |
|---:|---|---|---:|---:|---|---:|---|---:|---:|---|---|
| 4 | ring | periodic_ring | 1 | 0 | 0--0 | n/a | n/a | n/a | n/a | no_off_onset_control | none |
