# Adaptive Branching Readiness

This is the S8 no-submit readiness surface for mid-circuit adaptive
branching. It records branch policies and backend prerequisites, with
no hardware submission and no adaptive-advantage claim.

## Why this file exists

This page is used by teams deciding whether to move from simulation to
controlled feedback control experiments. It captures the exact branch-policy
surface the runtime can execute and the prerequisites that must be present before
adaptive control can be promoted.

It is intended for:

- control teams testing mid-circuit correction loops;
- experiment teams planning conditional gates and reset-dependent branches;
- reviewers who need a clear boundary between experimental readiness and
  promoted hardware claims.

## Boundary

readiness and branch-planning artifact only; no adaptive-advantage claim and no hardware submission

## Branch Policies
- `local_order_threshold`: observe partial local Kuramoto order parameter; trigger `local_r < target_r - deadband`; action `corrective_kick`.
- `dla_parity_leakage`: observe sector leakage estimate; trigger `parity_leakage > max_parity_leakage`; action `sector_rebalance`.
- `chimera_cluster_detector`: observe cluster imbalance with low local order; trigger `cluster_imbalance > threshold and local_r below target window`; action `topology_aware_pulse`.

## Readiness

- Ready: `False`
- Required features: `['mid_circuit_measurement', 'conditional_control', 'conditional_reset', 'cross_shot_batches']`
- Missing features: `['conditional_reset', 'cross_shot_batches']`
- Branch table rows: `12`

## Falsifier

win-rate <= 50% on preregistered equal-depth open-loop comparison

## Prerequisites
- S1 cross-shot feedback plumbing remains the runner-level dependency
- target backend must declare mid-circuit measurement and conditional control
- Rust branch-table generation is a performance follow-up, not a correctness prerequisite
- hardware execution requires explicit preregistration and approval

## Gate

Regenerate and compare this readiness artefact with:

```bash
scpn-bench s8-adaptive-branching-readiness
```
