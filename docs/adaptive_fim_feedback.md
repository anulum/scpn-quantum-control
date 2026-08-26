# Adaptive FIM next-experiment proposals

Adaptive FIM feedback turns leakage and exact-state-retention counts into
conservative proposals for a future static `lambda_fim` batch. It is an offline
experimental-design surface, not a realtime controller and not evidence that
FIM feedback protects hardware coherence.

## Decision path

1. Supply disjoint leakage and exact-state-retention event counts from one shot
   block with provenance.
2. Approve the complete paired control/candidate shot plan through the
   hardware-safe no-submit budget.
3. Compute a two-sided Wilson score interval for the configured witness.
4. Decrease `lambda_fim` only when the harmful-direction confidence bound clears
   the target and deadband. Ambiguous, missing-count, and underpowered evidence
   holds the current value.
5. Export an observer record and an unapplied scalar proposal to the co-design
   ports.

The product deliberately has no increase action. The committed repeated IBM
follow-up found that `lambda_fim = 4` increased leakage and reduced retention for
the tested circuit family, so this policy does not reward larger feedback
without a new, separately preregistered policy.

```python
from scpn_quantum_control.analysis.adaptive_fim_feedback import (
    AdaptiveFIMConfig,
    FIMWitness,
    plan_adaptive_fim_schedule,
)

witness = FIMWitness.from_counts(
    leakage_events=60,
    retention_events=400,
    shots=512,
    depth=2,
    source="simulator",
    artifact_id="example-count-block",
)
plan = plan_adaptive_fim_schedule(
    4.0,
    (witness,),
    policy_id="ci_dry_run_only",
    shots_per_arm=128,
    config=AdaptiveFIMConfig(target_leakage=0.05, step_gain=4.0),
)

assert plan.allowed
assert plan.steps[0].decision == "decrease"
assert plan.budget.would_submit is False
```

`shots_per_arm` describes the future paired control/candidate batch. The
hardware-safe execution policy gates that complete plan before any interval is
evaluated or observer is created. An over-budget or hardware request returns a
refused plan with no steps and no observer records.

## Interfaces

| Interface | Role |
| --- | --- |
| `FIMWitness.from_counts(...)` | Count-bound, provenance-carrying leakage and retention witness |
| `wilson_score_interval(...)` | Closed-form binomial interval used by the decision gate |
| `propose_count_aware_lambda(...)` | One conservative decrease-or-hold proposal |
| `adaptive_count_aware_schedule(...)` | Deterministically thread proposals through committed witnesses |
| `plan_adaptive_fim_schedule(...)` | Hardware-safe budget composition and co-design observer creation |
| `codesign.adaptive_fim_proposal_port(...)` | Convert a step into an unapplied one-parameter `ControllerProposal` |
| `propose_next_lambda(...)` | Legacy point-estimate compatibility helper; not a product/evidence route |

## Evidence

Run:

```bash
PYTHONPATH=src python scripts/run_adaptive_fim_evidence.py
```

The committed evidence contains three synthetic calibration controls and an
offline replay of circuits `0`, `3`, and `7` from the already committed repeated
FIM raw-count artefact. The source SHA256 and exact job identifier are bound into
the payload. The synthetic actions are `decrease -> hold -> hold`; the selected
historical adverse witnesses produce three deterministic decreases.

This establishes input validation, uncertainty gating, deterministic replay,
budget refusal, and hardware refusal. It does **not** establish that executing a
proposed batch would improve leakage or retention. Closed-loop efficacy remains
untested.

- [Evidence JSON](https://github.com/anulum/scpn-quantum-control/blob/main/data/adaptive_fim_product/adaptive_fim_evidence.json)
- [Evidence summary](https://github.com/anulum/scpn-quantum-control/blob/main/data/adaptive_fim_product/adaptive_fim_evidence.md)
- [Historical protocol boundary](campaigns/adaptive_fim_qpu_protocol_2026-05-06.md)

## Scientific basis and limits

Adaptive experimental design chooses later experiments using information from
earlier outcomes. Ferrie, Granade, and Cory developed that structure for bounded
Hamiltonian estimation
([AIP Conference Proceedings 1443, 165](https://doi.org/10.1063/1.3703632));
Hincks and colleagues evaluated online design against fixed sweeps at matched
data volume ([arXiv:1806.02427](https://arxiv.org/abs/1806.02427)).
For binomial uncertainty, Brown, Cai, and DasGupta document the poor coverage of
the Wald interval and recommend Wilson or Jeffreys alternatives
([Statistical Science 16, 101](https://doi.org/10.1214/ss/1009213286)).

Those sources motivate sequential proposal semantics and interval-aware count
handling. They do not validate this proportional update rule. This product
claims no optimal policy, Bayesian posterior, realtime feedback, provider
submission, hardware execution, FIM protection, control stability, or quantum
advantage.
